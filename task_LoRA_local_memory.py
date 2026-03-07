import argparse
import json
import os
from string import Formatter

import torch
import torch.nn as nn
import transformers
from datasets import Dataset, load_from_disk
from peft import PeftModel, LoraConfig, get_peft_model
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_first_k_tokens


parser = argparse.ArgumentParser(description="Train group-wise local memory (user embedding) on profile records")
parser.add_argument('--model_name', type=str, default='/cfs/models/llama/llama3.1-8B')
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--cut_off', type=int, default=2048)
parser.add_argument('--max_epoch', type=int, default=3)
parser.add_argument('--prefix_len', type=int, default=8)
parser.add_argument('--task_name', type=str, default='movie_tagging')
parser.add_argument('--add_profile', action='store_true')
parser.add_argument('--access_token', type=str, default=None)
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DDP')
parser.add_argument('--group_mapping_file', type=str, default='')
parser.add_argument('--num_groups', type=int, default=5)
parser.add_argument('--task_ckpt_path', type=str, default='', help='Optional override for task LoRA ckpt (default: k0 task ckpt)')
parser.add_argument('--local_lora_r', type=int, default=8)
parser.add_argument('--local_lora_alpha', type=int, default=16)
parser.add_argument('--local_lora_dropout', type=float, default=0.05)
parser.add_argument('--preprocess_only', action='store_true')
parser.add_argument('--cache_dir', type=str, default='./cache_local')
parser.add_argument('--map_num_proc', type=int, default=8)
args = parser.parse_args()

if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
    args.local_rank = int(os.environ["LOCAL_RANK"])
if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)

is_main_process = args.local_rank in (-1, 0)
model_name = args.model_name
task_name = args.task_name
batch_size = args.batch_size
k = args.k
cutoff_len = args.cut_off
add_eos_token = False
max_epoch = args.max_epoch

mapping_file = args.group_mapping_file or f"./data/{task_name}/group_mapping_100.json"
with open(mapping_file, 'r') as f:
    mapping_data = json.load(f)

mapping_records = mapping_data.get("mapping", [])
user_to_group = {}
for row in mapping_records:
    if "id" not in row or "group" not in row:
        continue
    user_to_group[int(row["id"])] = int(row["group"])

with open(f"./data/{task_name}/user_top_100_history.json", 'r') as f:
    train = json.load(f)

with open('./prompt/prompt.json', 'r') as f:
    prompt_template = json.load(f)

format_flag = False
if args.task_name in ["movie_tagging", "news_categorize", "news_headline", "product_rating", "scholarly_title"]:
    format_flag = True

profile_by_user_id = {}
if args.add_profile:
    with open(f'./data/{task_name}/profile_user_100.json', 'r') as f:
        train_profile = json.load(f)
    for item in train_profile:
        if "id" in item:
            profile_by_user_id[int(item["id"])] = item.get("output", "")


class UserMemoryPrefixModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_users, prefix_len):
        super().__init__()
        self.base_model = base_model
        self.prefix_len = prefix_len
        self.user_embedding = nn.Embedding(num_users, hidden_size * prefix_len)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids=None, attention_mask=None, labels=None, user_index=None, **kwargs):
        if user_index is None:
            raise ValueError("user_index is required")
        if input_ids is None:
            raise ValueError("input_ids is required")

        token_embeds = self.base_model.get_input_embeddings()(input_ids)
        user_embeds = self.user_embedding(user_index).view(-1, self.prefix_len, token_embeds.size(-1))
        inputs_embeds = torch.cat([user_embeds, token_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prefix_mask = torch.ones(
            (attention_mask.size(0), self.prefix_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        if labels is not None:
            prefix_labels = torch.full(
                (labels.size(0), self.prefix_len),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)

        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


class LocalMemoryCollator:
    def __init__(self, tokenizer):
        self.inner_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )

    def __call__(self, features):
        user_index = torch.tensor([f["user_index"] for f in features], dtype=torch.long)
        stripped = []
        for f in features:
            item = dict(f)
            item.pop("user_index", None)
            stripped.append(item)

        batch = self.inner_collator(stripped)
        batch["user_index"] = user_index
        return batch


def safe_truncate_dict(record, limit=768):
    out = {}
    for key, value in record.items():
        if isinstance(value, str):
            out[key] = get_first_k_tokens(value, limit)
        else:
            out[key] = value
    return out


def format_with_required_fields(template_str, values):
    required_keys = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template_str)
        if field_name is not None and field_name != ""
    }
    format_values = {key: values.get(key, "") for key in required_keys}
    return template_str.format(**format_values)


def build_prompt_pair(idx, profile_item, user_history, profile_text):
    q = safe_truncate_dict(profile_item, limit=768)
    oppu_input_template = prompt_template[args.task_name]['OPPU_input']
    oppu_full_template = prompt_template[args.task_name]['OPPU_full']
    prompt = format_with_required_fields(oppu_input_template, q)
    full_prompt = format_with_required_fields(oppu_full_template, q)

    if k > 0 and idx != 0 and format_flag:
        visible_history_list = [safe_truncate_dict(h, limit=768) for h in user_history[:idx]]
        history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]
        tokenized_corpus = [doc.split(" ") for doc in history_list]
        bm25 = BM25Okapi(tokenized_corpus)

        retrieval_query = format_with_required_fields(prompt_template[args.task_name]["retrieval_query"], q)
        tokenized_query = retrieval_query.split(' ')
        retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)

        history_string = "".join(retrieved_history)
        prompt = history_string + "\n" + prompt
        full_prompt = history_string + "\n" + full_prompt

    if args.add_profile and format_flag:
        prompt = profile_text + "\n" + prompt
        full_prompt = profile_text + "\n" + full_prompt

    return prompt, full_prompt


def build_train_data(users, user_id_to_index):
    train_data = []
    for sample in tqdm(users, disable=not is_main_process):
        uid = int(sample['user_id'])
        profile_text = profile_by_user_id.get(uid, "") if args.add_profile else ""
        user_index = user_id_to_index[uid]

        for idx, profile_item in enumerate(sample['profile']):
            prompt, full_prompt = build_prompt_pair(idx, profile_item, sample['profile'], profile_text)
            train_data.append({
                "prompt": prompt,
                "full_prompt": full_prompt,
                "user_index": user_index,
            })
    return train_data


def create_tokenizers(tokenizer):
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = data_point['full_prompt']
        tokenized_full_prompt = tokenize(full_prompt)
        user_prompt = data_point['prompt']

        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        tokenized_full_prompt["user_index"] = data_point["user_index"]
        return tokenized_full_prompt

    return generate_and_tokenize_prompt


def resolve_task_ckpt_path(task_name, model_name):
    model_short = model_name.split('/')[-1]
    return f"./ckpt/{task_name}/k0-{task_name}-{model_short}-task_LoRA_ckpt"


def create_frozen_backbone_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=args.access_token)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=False,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if args.local_rank != -1:
        base_model = base_model.to(torch.device("cuda", args.local_rank))

    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.bos_token_id = tokenizer.bos_token_id

    task_ckpt = args.task_ckpt_path or resolve_task_ckpt_path(task_name, model_name)

    if not os.path.exists(task_ckpt):
        raise FileNotFoundError(f"Task LoRA checkpoint not found: {task_ckpt}")

    task_model = PeftModel.from_pretrained(base_model, task_ckpt, is_trainable=False)
    merged_task_model = task_model.merge_and_unload()

    merged_task_model.gradient_checkpointing_enable()
    return merged_task_model, tokenizer, task_ckpt


def print_trainable_parameters(model):
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    if is_main_process:
        ratio = 100 * trainable / total if total > 0 else 0
        print(f"trainable params: {trainable} || all params: {total} || trainable%: {ratio:.4f}")


def build_local_lora_config():
    return LoraConfig(
        r=args.local_lora_r,
        lora_alpha=args.local_lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=args.local_lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


training_arguments = transformers.TrainingArguments(
    output_dir='outputs/',
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    optim='adamw_torch',
    num_train_epochs=max_epoch,
    save_steps=1e9,
    logging_steps=10,
    learning_rate=3e-4,
    weight_decay=0.0,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type='linear',
    report_to='none',
    local_rank=args.local_rank,
    ddp_find_unused_parameters=False,
    save_strategy="no"
)

users_by_group = {g: [] for g in range(args.num_groups)}
skipped_no_mapping = 0
skipped_out_of_range = 0

for user_sample in train:
    uid = int(user_sample['user_id'])
    group_id = user_to_group.get(uid)
    if group_id is None:
        skipped_no_mapping += 1
        continue
    if group_id < 0 or group_id >= args.num_groups:
        skipped_out_of_range += 1
        continue
    users_by_group[group_id].append(user_sample)

if is_main_process:
    print(f"Loaded train users (top100): {len(train)}")
    print(f"Loaded group mappings: {len(user_to_group)} from {mapping_file}")
    print(f"Skipped users (no mapping): {skipped_no_mapping}")
    print(f"Skipped users (group out of range): {skipped_out_of_range}")

saved_paths = []
for group_id in range(args.num_groups):
    group_users = users_by_group[group_id]
    if is_main_process:
        print(f"\n========== Group {group_id} ==========")
        print(f"Users in group: {len(group_users)}")

    if len(group_users) == 0:
        if is_main_process:
            print("Skip empty group.")
        continue

    unique_user_ids = sorted({int(u['user_id']) for u in group_users})
    user_id_to_index = {uid: idx for idx, uid in enumerate(unique_user_ids)}

    group_cache_path = os.path.join(args.cache_dir, args.task_name, f"group{group_id}")

    train_dataset = None
    if os.path.exists(group_cache_path):
        if is_main_process:
            print(f"Load cached dataset: {group_cache_path}")
        train_dataset = load_from_disk(group_cache_path)
        if is_main_process:
            print(f"Cached samples in group: {len(train_dataset)}")
    else:
        train_data = build_train_data(group_users, user_id_to_index)
        if is_main_process:
            print(f"Train samples in group: {len(train_data)}")
            print(f"Trainable users in local memory: {len(unique_user_ids)}")

        if len(train_data) == 0:
            if is_main_process:
                print("Skip group with zero train samples.")
            continue

        if is_main_process:
            preview_cnt = min(2, len(train_data))
            for preview_idx in range(preview_cnt):
                print(f"[Group {group_id}] Preview sample {preview_idx}")
                print(f"prompt:\n{train_data[preview_idx]['prompt']}")
                print(f"full_prompt:\n{train_data[preview_idx]['full_prompt']}")

        _, tokenizer_for_cache, _ = create_frozen_backbone_and_tokenizer()
        generate_and_tokenize_prompt_for_cache = create_tokenizers(tokenizer_for_cache)
        train_dataset = Dataset.from_list(train_data)
        train_dataset = train_dataset.map(generate_and_tokenize_prompt_for_cache, num_proc=args.map_num_proc)
        train_dataset = train_dataset.shuffle()

        if is_main_process:
            os.makedirs(os.path.dirname(group_cache_path), exist_ok=True)
            train_dataset.save_to_disk(group_cache_path)
            print(f"Saved cached dataset: {group_cache_path}")

    if args.preprocess_only:
        if is_main_process:
            print(f"Preprocess-only mode, skip training for group {group_id}.")
        continue

    try:
        frozen_model, tokenizer, task_ckpt_path = create_frozen_backbone_and_tokenizer()
    except FileNotFoundError as e:
        if is_main_process:
            print(f"Skip group {group_id}: {e}")
        continue

    hidden_size = frozen_model.config.hidden_size
    local_lora_config = build_local_lora_config()
    lora_base_model = get_peft_model(frozen_model, local_lora_config)
    model = UserMemoryPrefixModel(
        lora_base_model,
        hidden_size=hidden_size,
        num_users=len(unique_user_ids),
        prefix_len=args.prefix_len,
    )
    if args.local_rank != -1:
        model = model.to(torch.device("cuda", args.local_rank))

    if is_main_process:
        print(f"Loaded frozen task base: {task_ckpt_path}")
        print_trainable_parameters(model)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_arguments,
        data_collator=LocalMemoryCollator(tokenizer),
    )

    trainer.train()

    os.makedirs(f"./ckpt/{args.task_name}", exist_ok=True)
    model_short = model_name.split('/')[-1]
    if args.add_profile:
        output_name = f"./ckpt/{args.task_name}/k{args.k}-{args.task_name}-{model_short}-group{group_id}-profile-local-memory.pt"
    else:
        output_name = f"./ckpt/{args.task_name}/k{args.k}-{args.task_name}-{model_short}-group{group_id}-local-memory.pt"

    if is_main_process:
        if args.add_profile:
            local_lora_output = f"./ckpt/{args.task_name}/k{args.k}-{args.task_name}-{model_short}-group{group_id}-profile-local-memory-lora"
        else:
            local_lora_output = f"./ckpt/{args.task_name}/k{args.k}-{args.task_name}-{model_short}-group{group_id}-local-memory-lora"

        model.base_model.save_pretrained(local_lora_output)

        payload = {
            "user_embedding": model.user_embedding.state_dict(),
            "user_id_to_index": user_id_to_index,
            "group_id": group_id,
            "task_name": args.task_name,
            "model_name": args.model_name,
            "k": args.k,
            "add_profile": args.add_profile,
            "base_task_ckpt": task_ckpt_path,
            "hidden_size": hidden_size,
            "prefix_len": args.prefix_len,
            "local_lora_path": local_lora_output,
            "local_lora_r": args.local_lora_r,
            "local_lora_alpha": args.local_lora_alpha,
            "local_lora_dropout": args.local_lora_dropout,
        }
        torch.save(payload, output_name)
        saved_paths.append(output_name)
        print(f"Saved group {group_id} local memory: {output_name}")
        print(f"Saved group {group_id} local LoRA: {local_lora_output}")

    del trainer
    del model
    del frozen_model
    torch.cuda.empty_cache()

if is_main_process:
    print("\n========== Summary ==========")
    for group_id in range(args.num_groups):
        print(f"Group {group_id}: {len(users_by_group[group_id])} users")
    print(f"Saved local-memory checkpoints: {len(saved_paths)}")
    for p in saved_paths:
        print(f"- {p}")
