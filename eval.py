import argparse
import json
import os

import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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
    parser = argparse.ArgumentParser(description="Inference for LoRA")
    parser.add_argument('--model_name', type=str, default='/cfs/models/llama/llama3.1-8B')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--cut_off', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--task_name', type=str, default='movie_tagging')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--access_token', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default='', help='LoRA checkpoint path')
    parser.add_argument('--golds_json', type=str, default='', help='Gold labels json file')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA device id, e.g. 0-7')
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    task_name = args.task_name
    k = args.k

    if not args.golds_json:
        args.golds_json = f"./data/{task_name}/user_top_100_history_label.json"

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
    ).to(f"cuda:{args.cuda_id}")

    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.bos_token_id = tokenizer.bos_token_id

    if not args.ckpt_path:
        args.ckpt_path = f"./ckpt/{task_name}/k0-{task_name}-{model_name.split('/')[-1]}-task_LoRA_ckpt"

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    model = PeftModel.from_pretrained(base_model, args.ckpt_path)

    model.eval()
    model.config.use_cache = True

    with open(f"./data/{task_name}/user_top_100_history.json", 'r') as f:
        test_data = json.load(f)

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

    with open('./prompt/prompt.json', 'r') as f:
        prompt_template = json.load(f)

    if args.profile:
        with open(f'./data/{task_name}/profile_user_100.json', 'r') as f:
            test_profile = json.load(f)

    pred_all = []
    prompt_output_records = []

    with open(args.golds_json, 'r') as f:
        golds_data = json.load(f)
    golds_dict = {item['id']: item['output'] for item in golds_data.get('golds', [])}

    for i in tqdm(range(len(test_data))):
        user_id = test_data[i].get('user_id')
        if args.profile:
            profile = test_profile[i]['output']

        if k > 0:
            visible_history_list = test_data[i]['profile']
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

        for q in test_data[i]['query']:
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

            if k > 0 and args.task_name == 'citation':
                tokenized_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(query_title).split(" ")
                retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)

                history_string = "".join(retrieved_history)
                test_prompt = "##USER HISTORY:\n" + history_string + "\n" + test_prompt

            if k > 0 and args.task_name != 'citation':
                tokenized_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(test_article).split(" ")
                retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)

                history_string = "".join(retrieved_history)
                test_prompt = "##USER HISTORY:\n" + history_string + "\n" + test_prompt

            if args.profile:
                test_prompt = "##USER PROFILE:\n" + profile + "\n" + test_prompt

            test_question_list.append(test_prompt)
            question_id_list.append(q['id'])

        test_batch_list = split_batch(test_question_list, args.batch_size)
        out_list = []

        with torch.inference_mode():
            for batch_idx, batch in tqdm(enumerate(test_batch_list), total=len(test_batch_list)):
                sentences = batch
                inputs = tokenizer(sentences, return_tensors="pt", padding=True, return_token_type_ids=False)
                inputs = inputs.to(model.device)

                with torch.autocast(device_type="cuda"):
                    outputs = model.generate(
                        **inputs,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=100
                    )

                out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                out_list += out_sentence

        for i in range(len(out_list)):
            output = out_list[i].replace(test_question_list[i], '')
            if args.task_name == "citation":
                stripped = output.strip()
                if stripped == "1":
                    output = "[1]"
                elif stripped == "2":
                    output = "[2]"
            question_id = question_id_list[i]
            pred_all.append({
                "id": question_id,
                "output": output
                })

            prompt_output_records.append({
                "id": question_id,
                "user_id": user_id,
                "prompt": test_question_list[i],
                "output": output,
                "gold": golds_dict.get(question_id, "")
            })

            print(output)

    output_file = {
        'task': name2taskid[args.task_name],
        'golds': pred_all,
        'model': model_name,
    }

    output_dir = f"./output/{args.k}"
    preds_dir = f"{output_dir}/preds"
    os.makedirs(preds_dir, exist_ok=True)

    if args.profile:
        preds_path = f"{preds_dir}/output-task-k{args.k}-{args.task_name}-{model_name.split('/')[-1]}-profile.json"
    else:
        preds_path = f"{preds_dir}/output-task-k{args.k}-{args.task_name}-{model_name.split('/')[-1]}.json"

    with open(preds_path, 'w') as f:
        json.dump(output_file, f, indent=4)

    output_dir = f"./output/{args.k}"
    if args.profile:
        prompt_output_path = f"{output_dir}/prompt-output-k{args.k}-{args.task_name}-{model_name.split('/')[-1]}-profile.jsonl"
    else:
        prompt_output_path = f"{output_dir}/prompt-output-k{args.k}-{args.task_name}-{model_name.split('/')[-1]}.jsonl"
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
