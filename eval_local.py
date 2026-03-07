import argparse
import json
import os

import torch
from peft import PeftModel
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.evaluation import LaMPEvaluation
from utils import (
    split_batch,
    get_first_k_tokens,
    name2taskid,
    extract_citation_title,
    extract_option,
    extract_movie,
    extract_news_cat,
    extract_news_headline,
    extract_product_review,
    extract_scholarly_title,
    extract_tweet_paraphrasing,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Group-routed local-memory inference")
    parser.add_argument('--model_name', type=str, default='/cfs/models/llama/llama3.1-8B')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--k', type=int, default=0, help='Only for optional retrieval prompt construction')
    parser.add_argument('--task_name', type=str, default='movie_tagging')
    parser.add_argument('--access_token', type=str, default=None)
    parser.add_argument('--golds_json', type=str, default='', help='Gold labels json file')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA device id')
    parser.add_argument('--group_mapping_file', type=str, default='')
    parser.add_argument('--num_groups', type=int, default=5)
    parser.add_argument('--task_ckpt_path', type=str, default='', help='Optional override for task LoRA ckpt (default: k0 task ckpt)')
    parser.add_argument('--group_mode', type=int, default=0, help='0: use group2 OPPU LoRA, 1: use group task LoRA')
    parser.add_argument('--local_memory_dir', type=str, default='', help='Optional override dir for local memory checkpoints')
    parser.add_argument('--prefix_len', type=int, default=8)
    parser.add_argument('--profile', action='store_true')
    return parser.parse_args()


def resolve_task_ckpt_path(task_name, model_name):
    model_short = model_name.split('/')[-1]
    return f"./ckpt/{task_name}/k0-{task_name}-{model_short}-task_LoRA_ckpt"


def resolve_group2_ckpt_path(task_name, model_name, group_id):
    model_short = model_name.split('/')[-1]
    return f"./ckpt/{task_name}/k0-{task_name}-{model_short}-group{group_id}-OPPU-task_LoRA_ckpt"


def resolve_group_task_ckpt_path(task_name, model_name, group_id):
    model_short = model_name.split('/')[-1]
    return f"./ckpt/{task_name}/k0-{task_name}-{model_short}-group{group_id}-task_LoRA_ckpt"


def resolve_local_memory_ckpt_path(task_name, model_name, group_id, local_memory_dir=''):
    model_short = model_name.split('/')[-1]
    base_dir = local_memory_dir or f"./ckpt/{task_name}"
    return f"{base_dir}/k0-{task_name}-{model_short}-group{group_id}-local-memory.pt"


def resolve_local_lora_ckpt_path(task_name, model_name, group_id, local_memory_dir=''):
    model_short = model_name.split('/')[-1]
    base_dir = local_memory_dir or f"./ckpt/{task_name}"
    return f"{base_dir}/k0-{task_name}-{model_short}-group{group_id}-local-memory-lora"


def load_group_backbone(model_name, task_name, group_id, cuda_id, task_ckpt_override='', group_mode=0):
    task_ckpt = task_ckpt_override or resolve_task_ckpt_path(task_name, model_name)
    if group_mode == 1:
        group_ckpt = resolve_group_task_ckpt_path(task_name, model_name, group_id)
        group_ckpt_type = "group task"
    else:
        group_ckpt = resolve_group2_ckpt_path(task_name, model_name, group_id)
        group_ckpt_type = "group2 OPPU"

    if not os.path.exists(task_ckpt):
        raise FileNotFoundError(f"Task LoRA checkpoint not found: {task_ckpt}")
    if not os.path.exists(group_ckpt):
        raise FileNotFoundError(f"{group_ckpt_type} LoRA checkpoint not found: {group_ckpt}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=False,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(f"cuda:{cuda_id}")

    task_model = PeftModel.from_pretrained(base_model, task_ckpt, is_trainable=False)
    merged_task_model = task_model.merge_and_unload()

    group_model = PeftModel.from_pretrained(merged_task_model, group_ckpt, is_trainable=False)
    merged_group_model = group_model.merge_and_unload()

    merged_group_model.eval()
    merged_group_model.config.use_cache = True

    return merged_group_model, task_ckpt, group_ckpt, group_ckpt_type


def build_inputs_with_user_prefix(model, embedding_layer, prefix_len, user_index, input_ids, attention_mask):
    token_embeds = model.get_input_embeddings()(input_ids)
    user_ids = torch.full((input_ids.size(0),), user_index, dtype=torch.long, device=input_ids.device)
    user_embeds = embedding_layer(user_ids).view(-1, prefix_len, token_embeds.size(-1))
    inputs_embeds = torch.cat([user_embeds, token_embeds], dim=1)

    prefix_mask = torch.ones(
        (attention_mask.size(0), prefix_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    merged_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
    return inputs_embeds, merged_attention_mask


def main():
    args = parse_args()
    model_name = args.model_name
    task_name = args.task_name

    if not args.golds_json:
        args.golds_json = f"./data/{task_name}/user_top_100_history_label.json"

    mapping_file = args.group_mapping_file or f"./data/{task_name}/group_mapping_100.json"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=args.access_token)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    mapping_records = mapping_data.get('mapping', [])
    user_to_group = {}
    for row in mapping_records:
        if 'id' not in row or 'group' not in row:
            continue
        user_to_group[int(row['id'])] = int(row['group'])

    with open(f"./data/{task_name}/user_top_100_history.json", 'r') as f:
        test_data = json.load(f)

    profile_by_user_id = {}
    if args.profile:
        with open(f'./data/{task_name}/profile_user_100.json', 'r') as f:
            test_profile = json.load(f)
        for item in test_profile:
            if 'id' in item:
                profile_by_user_id[int(item['id'])] = item.get('output', '')

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

    with open(args.golds_json, 'r') as f:
        golds_data = json.load(f)
    golds_dict = {item['id']: item['output'] for item in golds_data.get('golds', [])}

    users_by_group = {g: [] for g in range(args.num_groups)}
    missing_mapping_count = 0
    out_of_range_count = 0

    for i in range(len(test_data)):
        user_id = int(test_data[i].get('user_id'))
        group_id = user_to_group.get(user_id)
        if group_id is None:
            missing_mapping_count += 1
            group_id = 0
        if group_id < 0 or group_id >= args.num_groups:
            out_of_range_count += 1
            continue
        users_by_group[group_id].append(i)

    pred_all = []
    prompt_output_records = []
    missing_ckpt_groups = []
    missing_user_memory_users = 0

    for group_id in range(args.num_groups):
        group_indices = users_by_group[group_id]
        if len(group_indices) == 0:
            continue

        print(f"\n========== Group {group_id} ==========")
        print(f"Users in group: {len(group_indices)}")

        local_memory_ckpt = resolve_local_memory_ckpt_path(
            task_name=task_name,
            model_name=model_name,
            group_id=group_id,
            local_memory_dir=args.local_memory_dir,
        )
        if not os.path.exists(local_memory_ckpt):
            print(f"[WARN] Skip group {group_id}: local memory ckpt not found: {local_memory_ckpt}")
            missing_ckpt_groups.append(group_id)
            continue

        try:
            model, task_ckpt_path, group_ckpt_path, group_ckpt_type = load_group_backbone(
                model_name=model_name,
                task_name=task_name,
                group_id=group_id,
                cuda_id=args.cuda_id,
                task_ckpt_override=args.task_ckpt_path,
                group_mode=args.group_mode,
            )
            print(f"Loaded frozen task base: {task_ckpt_path}")
            print(f"Loaded frozen {group_ckpt_type} base: {group_ckpt_path}")
        except FileNotFoundError as e:
            print(f"[WARN] Skip group {group_id}: {e}")
            missing_ckpt_groups.append(group_id)
            continue

        payload = torch.load(local_memory_ckpt, map_location='cpu')
        user_id_to_index = {int(k): int(v) for k, v in payload.get("user_id_to_index", {}).items()}
        prefix_len = int(payload.get("prefix_len", args.prefix_len))
        hidden_size = int(payload.get("hidden_size", model.config.hidden_size))

        local_lora_path = payload.get("local_lora_path")
        if not local_lora_path:
            local_lora_path = resolve_local_lora_ckpt_path(
                task_name=task_name,
                model_name=model_name,
                group_id=group_id,
                local_memory_dir=args.local_memory_dir,
            )
        if os.path.exists(local_lora_path):
            model = PeftModel.from_pretrained(model, local_lora_path, is_trainable=False)
            model = model.merge_and_unload()
            model.eval()
            model.config.use_cache = True
            print(f"Loaded local LoRA: {local_lora_path}")
        else:
            print(f"[WARN] Local LoRA not found, fallback to prefix-only: {local_lora_path}")

        embedding_layer = torch.nn.Embedding(len(user_id_to_index), hidden_size * prefix_len)
        embedding_layer.load_state_dict(payload["user_embedding"])
        embedding_layer = embedding_layer.to(model.device)
        embedding_layer.eval()
        print(f"Loaded local memory: {local_memory_ckpt}")

        for idx in tqdm(group_indices):
            user_id = int(test_data[idx].get('user_id'))
            profile_text = profile_by_user_id.get(user_id, '') if args.profile else ''
            user_index = user_id_to_index.get(user_id)
            if user_index is None:
                missing_user_memory_users += 1
                print(f"[WARN] Missing user embedding for user_id={user_id}, skip")
                continue

            if args.k > 0:
                visible_history_list = test_data[idx]['profile']
                for p in visible_history_list:
                    for key, value in p.items():
                        if isinstance(value, str):
                            p[key] = get_first_k_tokens(value, 368)
                        else:
                            p[key] = value

                history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]
                tokenized_corpus = [doc.split(" ") for doc in history_list]
                bm25 = BM25Okapi(tokenized_corpus)

            test_question_list = []
            question_id_list = []

            for q in test_data[idx]['query']:
                if args.task_name == 'citation':
                    test_question = q['input']
                    query_title = extract_citation_title(test_question)
                    option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
                    test_prompt = prompt_template[args.task_name]['prompt'].format(
                        query_title,
                        option1,
                        option2,
                    )
                else:
                    test_question = q['input']
                    test_article = extract_article(test_question)
                    test_prompt = prompt_template[args.task_name]['prompt'].format(test_article)

                test_prompt = "##INSTRUCTION:\n" + test_prompt

                if args.profile:
                    test_prompt = "##USER PROFILE:\n" + profile_text + "\n" + test_prompt

                if args.k > 0 and args.task_name == 'citation':
                    tokenized_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(query_title).split(" ")
                    retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)
                    history_string = "".join(retrieved_history)
                    test_prompt = "##USER HISTORY:\n" + history_string + "\n" + test_prompt

                if args.k > 0 and args.task_name != 'citation':
                    tokenized_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(test_article).split(" ")
                    retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)
                    history_string = "".join(retrieved_history)
                    test_prompt = "##USER HISTORY:\n" + history_string + "\n" + test_prompt

                test_question_list.append(test_prompt)
                question_id_list.append(q['id'])

            test_batch_list = split_batch(test_question_list, args.batch_size)
            out_list = []

            with torch.inference_mode():
                for batch in test_batch_list:
                    inputs = tokenizer(batch, return_tensors="pt", padding=True, return_token_type_ids=False)
                    inputs = inputs.to(model.device)

                    inputs_embeds, merged_attention_mask = build_inputs_with_user_prefix(
                        model=model,
                        embedding_layer=embedding_layer,
                        prefix_len=prefix_len,
                        user_index=user_index,
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )

                    with torch.autocast(device_type="cuda"):
                        outputs = model.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=merged_attention_mask,
                            do_sample=False,
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=100,
                        )

                    out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    out_list += out_sentence

            for j in range(len(out_list)):
                output = out_list[j].replace(test_question_list[j], '')
                if args.task_name == "citation":
                    stripped = output.strip()
                    if stripped == "1":
                        output = "[1]"
                    elif stripped == "2":
                        output = "[2]"

                question_id = question_id_list[j]
                pred_all.append({
                    "id": question_id,
                    "output": output
                })

                prompt_output_records.append({
                    "id": question_id,
                    "user_id": user_id,
                    "group": group_id,
                    "prompt": test_question_list[j],
                    "output": output,
                    "gold": golds_dict.get(question_id, "")
                })

                print(output)

        del embedding_layer
        del model
        torch.cuda.empty_cache()

    print(f"[INFO] Missing mapping users: {missing_mapping_count}")
    print(f"[INFO] Out-of-range group users: {out_of_range_count}")
    print(f"[INFO] Groups skipped due to missing ckpt: {missing_ckpt_groups}")
    print(f"[INFO] Users skipped due to missing local memory embedding: {missing_user_memory_users}")

    output_file = {
        'task': name2taskid[args.task_name],
        'golds': pred_all,
        'model': model_name,
    }

    output_dir = f"./output_local/{args.k}"
    preds_dir = f"{output_dir}/preds"
    os.makedirs(preds_dir, exist_ok=True)

    if args.profile:
        preds_path = f"{preds_dir}/output-task-local-k{args.k}-{args.task_name}-{model_name.split('/')[-1]}-profile.json"
    else:
        preds_path = f"{preds_dir}/output-task-local-k{args.k}-{args.task_name}-{model_name.split('/')[-1]}.json"

    with open(preds_path, 'w') as f:
        json.dump(output_file, f, indent=4)

    if args.profile:
        prompt_output_path = f"{output_dir}/prompt-output-local-k{args.k}-{args.task_name}-{model_name.split('/')[-1]}-profile.jsonl"
    else:
        prompt_output_path = f"{output_dir}/prompt-output-local-k{args.k}-{args.task_name}-{model_name.split('/')[-1]}.jsonl"

    with open(prompt_output_path, 'w') as f:
        for record in prompt_output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.golds_json:
        evaluator = LaMPEvaluation(single_gold_json_file_addr=args.golds_json)
        results = evaluator.evaluate_task(preds_path, name2taskid[args.task_name])
        results = {key: round(value, 3) if isinstance(value, float) else value for key, value in results.items()}
        print(results)


if __name__ == "__main__":
    main()
