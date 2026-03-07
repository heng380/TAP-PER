import os
import json
import argparse

import torch
import transformers
from datasets import Dataset
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

from utils import (
    get_first_k_tokens,
    print_trainable_parameters,
    extract_citation_title,
    extract_option,
    extract_movie,
    extract_news_cat,
    extract_news_headline,
    extract_product_review,
    extract_scholarly_title,
    extract_tweet_paraphrasing,
)


parser = argparse.ArgumentParser(description="Train group-wise task LoRA adapters")
parser.add_argument('--model_name', type=str, default='/cfs/models/llama/llama3.1-8B')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--max_step', type=int, default=5000)
parser.add_argument('--cut_off', type=int, default=2048)
parser.add_argument('--max_epoch', type=int, default=3)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--task_name', type=str, default='movie_tagging')
parser.add_argument('--add_profile', action='store_true')
parser.add_argument('--access_token', type=str, default=None)
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DDP')
parser.add_argument('--group_mapping_file', type=str, default='')
parser.add_argument('--num_groups', type=int, default=5)

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

with open(f"./data/{task_name}/user_others.json", 'r') as f:
    train = json.load(f)

mapping_file = args.group_mapping_file or f"./data/{task_name}/group_mapping_others.json"
with open(mapping_file, 'r') as f:
    mapping_data = json.load(f)

mapping_records = mapping_data.get("mapping", [])
user_to_group = {}
for row in mapping_records:
    if "id" not in row or "group" not in row:
        continue
    user_to_group[int(row["id"])] = int(row["group"])

if args.task_name == "movie_tagging":
    extract_article = extract_movie
elif args.task_name == "news_categorize":
    extract_article = extract_news_cat
elif args.task_name == "news_headline":
    extract_article = extract_news_headline
elif args.task_name == "product_rating":
    extract_article = extract_product_review
elif args.task_name == "scholarly_title":
    extract_article = extract_scholarly_title
elif args.task_name == "tweet_paraphrase":
    extract_article = extract_tweet_paraphrasing
else:
    extract_article = None

with open('./prompt/prompt.json', 'r') as f:
    prompt_template = json.load(f)

profile_by_user_id = {}
if args.add_profile:
    with open(f'./data/{task_name}/profile_user_others.json', 'r') as f:
        train_profile = json.load(f)
    for item in train_profile:
        if "id" in item:
            profile_by_user_id[int(item["id"])] = item.get("output", "")


def build_prompt_pair(q, profile_text, user_profile):
    if args.task_name != "citation":
        article = get_first_k_tokens(extract_article(q['input']), 768)
        prompt = prompt_template[args.task_name]['prompt'].format(article)
        full_prompt = prompt_template[args.task_name]['full_prompt'].format(
            get_first_k_tokens(extract_article(q['input']), 768), q['gold']
        )
    else:
        question = q['input']
        article = extract_citation_title(question)
        option1, option2 = extract_option(question, 1), extract_option(question, 2)
        prompt = prompt_template[args.task_name]['prompt'].format(article, option1, option2)
        full_prompt = prompt_template[args.task_name]['full_prompt'].format(article, option1, option2, q['gold'])

    if k > 0:
        visible_history_list = user_profile
        for p in visible_history_list:
            for key, value in p.items():
                p[key] = get_first_k_tokens(p[key], 368)

        history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]
        tokenized_corpus = [doc.split(" ") for doc in history_list]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = prompt_template[args.task_name]["retrieval_query_wokey"].format(article).split(' ')
        retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)

        history_string = "".join(retrieved_history)
        prompt = history_string + "\n" + prompt
        full_prompt = history_string + "\n" + full_prompt

    if args.add_profile:
        prompt = profile_text + "\n" + prompt
        full_prompt = profile_text + "\n" + full_prompt

    return prompt, full_prompt


def build_train_data(users):
    train_data = []
    for sample in tqdm(users, disable=not is_main_process):
        uid = int(sample['user_id'])
        profile_text = profile_by_user_id.get(uid, "") if args.add_profile else ""

        for q in sample['query']:
            prompt, full_prompt = build_prompt_pair(q, profile_text, sample['profile'])
            train_data.append({
                "prompt": prompt,
                "full_prompt": full_prompt,
            })
    return train_data


def build_tokenizers(tokenizer):
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

        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]
        return tokenized_full_prompt

    return generate_and_tokenize_prompt


def create_base_model_and_tokenizer():
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
        torch_dtype=torch.bfloat16
    )
    if args.local_rank != -1:
        base_model = base_model.to(torch.device("cuda", args.local_rank))

    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.bos_token_id = tokenizer.bos_token_id

    base_model.gradient_checkpointing_enable()
    return base_model, tokenizer


def create_train_model(base_model):
    default_init_lora_path = "./ckpt/{}/k{}-{}-{}-task_LoRA_ckpt".format(
        args.task_name,
        args.k,
        args.task_name,
        model_name.split('/')[-1],
    )
    if os.path.exists(default_init_lora_path):
        model = PeftModel.from_pretrained(base_model, default_init_lora_path, is_trainable=True)
    else:
        model = get_peft_model(base_model, peft_config)
    return model, default_init_lora_path


peft_config = LoraConfig(
    r=4,
    lora_alpha=4,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_arguments = transformers.TrainingArguments(
    output_dir='outputs/',
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    optim='adamw_torch',
    num_train_epochs=max_epoch,
    save_steps=1e9,
    logging_steps=1,
    learning_rate=5e-5,
    weight_decay=1e-2,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type='linear',
    report_to='none',
    local_rank=args.local_rank,
    ddp_find_unused_parameters=False,
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
    print(f"Loaded train users: {len(train)}")
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

    train_data = build_train_data(group_users)
    if is_main_process:
        print(f"Train samples in group: {len(train_data)}")

    if len(train_data) == 0:
        if is_main_process:
            print("Skip group with zero train samples.")
        continue

    base_model, tokenizer = create_base_model_and_tokenizer()
    model, init_lora_path = create_train_model(base_model)
    if is_main_process:
        if os.path.exists(init_lora_path):
            print(f"Initialize from LoRA checkpoint: {init_lora_path}")
        else:
            print(f"Base LoRA checkpoint not found, training from new adapter: {init_lora_path}")
        print_trainable_parameters(model)

    generate_and_tokenize_prompt = build_tokenizers(tokenizer)
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(generate_and_tokenize_prompt).shuffle()

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False
    trainer.train()

    os.makedirs(f"./ckpt/{args.task_name}", exist_ok=True)
    if args.add_profile:
        output_name = "./ckpt/{}/k{}-{}-{}-group{}-profile-task_LoRA_ckpt".format(
            args.task_name, args.k, args.task_name, model_name.split('/')[-1], group_id
        )
    else:
        output_name = "./ckpt/{}/k{}-{}-{}-group{}-task_LoRA_ckpt".format(
            args.task_name, args.k, args.task_name, model_name.split('/')[-1], group_id
        )

    if is_main_process:
        model.save_pretrained(output_name)
        saved_paths.append(output_name)
        print(f"Saved group {group_id} LoRA: {output_name}")

if is_main_process:
    print("\n========== Summary ==========")
    for group_id in range(args.num_groups):
        print(f"Group {group_id}: {len(users_by_group[group_id])} users")
    print(f"Saved checkpoints: {len(saved_paths)}")
    for p in saved_paths:
        print(f"- {p}")
