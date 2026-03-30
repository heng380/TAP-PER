import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
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
    parser = argparse.ArgumentParser(description="RAG-prefix inference")
    parser.add_argument('--model_name', type=str, default='/cfs/models/llama/llama3.1-8B')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--k', type=int, default=10, help='Top-k records used for RAG-prefix')
    parser.add_argument('--task_name', type=str, default='movie_tagging')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--access_token', type=str, default=None)
    parser.add_argument('--golds_json', type=str, default='', help='Gold labels json file')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA device id')
    parser.add_argument('--task_ckpt_path', type=str, default='', help='Task-level LoRA checkpoint path')
    parser.add_argument('--rag_prefix_ckpt_path', type=str, default='', help='RAG-prefix .pt checkpoint path')
    parser.add_argument('--rag_lora_path', type=str, default='', help='RAG-prefix LoRA checkpoint path')
    parser.add_argument('--query_max_len', type=int, default=128)
    parser.add_argument('--record_max_len', type=int, default=128)
    parser.add_argument('--use_time_bias', action='store_true', help='Enable recency-based bias in RAG attention')
    parser.add_argument('--use_order_bias', action='store_true', help='Enable order-index bias in RAG attention')
    return parser.parse_args()


def resolve_task_ckpt_path(task_name, model_name):
    model_short = model_name.split('/')[-1]
    return f"./ckpt/{task_name}/k0-{task_name}-{model_short}-task_LoRA_ckpt"


def resolve_rag_prefix_ckpt_path(task_name, model_name, k, profile, use_time_bias=False, use_order_bias=False):
    model_short = model_name.split('/')[-1]
    suffix = "-profile" if profile else ""
    bias_tag = f"{'-tb' if use_time_bias else ''}{'-ob' if use_order_bias else ''}"
    return f"./ckpt/{task_name}/k{k}-{task_name}-{model_short}{suffix}{bias_tag}-rag-prefix.pt"


def resolve_rag_lora_path(task_name, model_name, k, profile, use_time_bias=False, use_order_bias=False):
    model_short = model_name.split('/')[-1]
    suffix = "-profile" if profile else ""
    bias_tag = f"{'-tb' if use_time_bias else ''}{'-ob' if use_order_bias else ''}"
    return f"./ckpt/{task_name}/k{k}-{task_name}-{model_short}{suffix}{bias_tag}-rag-prefix-lora"


class RagPrefixModule(nn.Module):
    def __init__(self, hidden_size, prefix_len, use_time_bias=False, use_order_bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.prefix_len = prefix_len
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.prefix_proj = nn.Linear(hidden_size, hidden_size * prefix_len)
        self.use_time_bias = use_time_bias
        self.use_order_bias = use_order_bias
        self.register_buffer("time_bias_lambda", torch.tensor(0.0), persistent=False)
        self.register_buffer("order_bias_lambda", torch.tensor(0.0), persistent=False)

    def _mean_pool_embeds(self, embeds, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(embeds.dtype)
        summed = (embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def build_prefix(self, embedding_layer, query_input_ids, query_attention_mask, record_input_ids, record_attention_mask, record_valid_mask, record_delta_days=None, record_order_idx=None):
        batch_size = query_input_ids.size(0)
        top_k = record_input_ids.size(1)
        rec_len = record_input_ids.size(2)

        q_embeds = embedding_layer(query_input_ids)
        q_vec = self._mean_pool_embeds(q_embeds, query_attention_mask)

        rec_input_ids_flat = record_input_ids.reshape(batch_size * top_k, rec_len)
        rec_attention_flat = record_attention_mask.reshape(batch_size * top_k, rec_len)
        rec_embeds_flat = embedding_layer(rec_input_ids_flat)
        rec_vec = self._mean_pool_embeds(rec_embeds_flat, rec_attention_flat).reshape(batch_size, top_k, self.hidden_size)

        q_expand = q_vec.unsqueeze(1).expand_as(rec_vec)
        feat = torch.cat([q_expand, rec_vec, q_expand - rec_vec, q_expand * rec_vec], dim=-1)
        scores = self.attn_mlp(feat).squeeze(-1)

        valid = record_valid_mask.to(scores.dtype)
        scores = scores.masked_fill(valid == 0, -1e4)

        if self.use_time_bias and record_delta_days is not None:
            recency_penalty = torch.log1p(record_delta_days.to(scores.dtype).clamp_min(0.0))
            scores = scores - self.time_bias_lambda.to(scores.dtype) * recency_penalty
            scores = scores.masked_fill(valid == 0, -1e4)

        if self.use_order_bias and record_order_idx is not None:
            order_penalty = torch.log1p(record_order_idx.to(scores.dtype).clamp_min(0.0))
            scores = scores - self.order_bias_lambda.to(scores.dtype) * order_penalty
            scores = scores.masked_fill(valid == 0, -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = attn * valid
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        context = (attn.unsqueeze(-1) * rec_vec).sum(dim=1)
        prefix = self.prefix_proj(context).view(batch_size, self.prefix_len, self.hidden_size)
        return prefix


def build_prefix_inputs(tokenizer, query_texts, retrieved_texts_list, rag_k, query_max_len, record_max_len, device, delta_days_list=None, order_idx_list=None):
    batch_size = len(query_texts)

    record_texts_flat = []
    valid_mask = []
    delta_days_flat = []
    order_idx_flat = []
    for i, recs in enumerate(retrieved_texts_list):
        recs = recs if isinstance(recs, list) else []
        recs = [str(r) for r in recs][:rag_k]
        deltas = [] if delta_days_list is None else delta_days_list[i]
        if not isinstance(deltas, list):
            deltas = []
        deltas = [float(x) if x is not None else 0.0 for x in deltas][:rag_k]
        orders = [] if order_idx_list is None else order_idx_list[i]
        if not isinstance(orders, list):
            orders = []
        orders = [float(x) if x is not None else 0.0 for x in orders][:rag_k]

        if len(recs) < rag_k:
            recs = recs + [""] * (rag_k - len(recs))
        if len(deltas) < rag_k:
            deltas = deltas + [0.0] * (rag_k - len(deltas))
        if len(orders) < rag_k:
            orders = orders + [0.0] * (rag_k - len(orders))

        for rec, dd, oi in zip(recs, deltas, orders):
            record_texts_flat.append(rec)
            valid_mask.append(1 if rec.strip() else 0)
            delta_days_flat.append(float(dd))
            order_idx_flat.append(float(oi))

    q_tokens = tokenizer(
        query_texts,
        padding=True,
        truncation=True,
        max_length=query_max_len,
        return_tensors="pt",
    )
    r_tokens = tokenizer(
        record_texts_flat,
        padding=True,
        truncation=True,
        max_length=record_max_len,
        return_tensors="pt",
    )

    query_input_ids = q_tokens["input_ids"].to(device)
    query_attention_mask = q_tokens["attention_mask"].to(device)
    record_input_ids = r_tokens["input_ids"].view(batch_size, rag_k, -1).to(device)
    record_attention_mask = r_tokens["attention_mask"].view(batch_size, rag_k, -1).to(device)
    record_valid_mask = torch.tensor(valid_mask, dtype=torch.long, device=device).view(batch_size, rag_k)

    return {
        "query_input_ids": query_input_ids,
        "query_attention_mask": query_attention_mask,
        "record_input_ids": record_input_ids,
        "record_attention_mask": record_attention_mask,
        "record_valid_mask": record_valid_mask,
        "record_delta_days": torch.tensor(delta_days_flat, dtype=torch.float, device=device).view(batch_size, rag_k),
        "record_order_idx": torch.tensor(order_idx_flat, dtype=torch.float, device=device).view(batch_size, rag_k),
    }


def main():
    args = parse_args()
    model_name = args.model_name
    task_name = args.task_name

    if not args.golds_json:
        args.golds_json = f"./data/{task_name}/user_top_100_history_label.json"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=args.access_token)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    task_ckpt_path = args.task_ckpt_path or resolve_task_ckpt_path(task_name, model_name)
    rag_prefix_ckpt = args.rag_prefix_ckpt_path or resolve_rag_prefix_ckpt_path(
        task_name, model_name, args.k, args.profile, args.use_time_bias, args.use_order_bias
    )
    rag_lora_path = args.rag_lora_path or resolve_rag_lora_path(
        task_name, model_name, args.k, args.profile, args.use_time_bias, args.use_order_bias
    )

    if not os.path.exists(task_ckpt_path):
        raise FileNotFoundError(f"Task LoRA checkpoint not found: {task_ckpt_path}")
    if not os.path.exists(rag_prefix_ckpt):
        raise FileNotFoundError(f"RAG-prefix checkpoint not found: {rag_prefix_ckpt}")
    if not os.path.exists(rag_lora_path):
        raise FileNotFoundError(f"RAG-prefix LoRA path not found: {rag_lora_path}")

    print(f"Loaded task LoRA path: {task_ckpt_path}")
    print(f"Loaded mediator LoRA path: {rag_lora_path}")
    print(f"Loaded RAG-prefix .pt path: {rag_prefix_ckpt}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=False,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(f"cuda:{args.cuda_id}")

    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.bos_token_id = tokenizer.bos_token_id

    task_model = PeftModel.from_pretrained(base_model, task_ckpt_path, is_trainable=False)
    model = task_model.merge_and_unload()

    rag_model = PeftModel.from_pretrained(model, rag_lora_path, is_trainable=False)
    model = rag_model.merge_and_unload()

    model.eval()
    model.config.use_cache = True

    rag_payload = torch.load(rag_prefix_ckpt, map_location='cpu')
    prefix_len = int(rag_payload.get("prefix_len", 8))
    hidden_size = int(rag_payload.get("hidden_size", model.config.hidden_size))
    query_max_len = int(rag_payload.get("query_max_len", args.query_max_len))
    record_max_len = int(rag_payload.get("record_max_len", args.record_max_len))

    use_time_bias = bool(rag_payload.get("use_time_bias", False) or args.use_time_bias)
    use_order_bias = bool(rag_payload.get("use_order_bias", False) or args.use_order_bias)
    rag_prefix = RagPrefixModule(
        hidden_size=hidden_size,
        prefix_len=prefix_len,
        use_time_bias=use_time_bias,
        use_order_bias=use_order_bias,
    )
    rag_prefix.attn_mlp.load_state_dict(rag_payload["attn_mlp"])
    rag_prefix.prefix_proj.load_state_dict(rag_payload["prefix_proj"])
    if rag_payload.get("time_bias_lambda", None) is not None:
        rag_prefix.time_bias_lambda = torch.tensor(float(rag_payload["time_bias_lambda"]))
    if rag_payload.get("order_bias_lambda", None) is not None:
        rag_prefix.order_bias_lambda = torch.tensor(float(rag_payload["order_bias_lambda"]))
    rag_prefix_dtype = model.get_input_embeddings().weight.dtype
    rag_prefix = rag_prefix.to(model.device, dtype=rag_prefix_dtype)
    rag_prefix.eval()

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

    pred_all = []
    prompt_output_records = []

    for i in tqdm(range(len(test_data))):
        user_id = int(test_data[i].get('user_id'))
        profile_text = profile_by_user_id.get(user_id, '') if args.profile else ''

        history_list = []
        bm25 = None
        if args.k > 0:
            visible_history_list = test_data[i]['profile']
            for p in visible_history_list:
                for key, value in p.items():
                    if isinstance(value, str):
                        p[key] = get_first_k_tokens(value, 368)

            history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]
            if len(history_list) > 0:
                tokenized_corpus = [doc.split(" ") for doc in history_list]
                bm25 = BM25Okapi(tokenized_corpus)

        test_prompt_list = []
        query_texts = []
        retrieved_texts_list = []
        retrieved_delta_days_list = []
        retrieved_order_idx_list = []
        question_id_list = []

        for q in test_data[i]['query']:
            if args.task_name == 'citation':
                test_question = q['input']
                query_title = extract_citation_title(test_question)
                option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
                test_prompt = prompt_template[args.task_name]['prompt'].format(query_title, option1, option2)
                retrieval_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(query_title)
            else:
                test_question = q['input']
                test_article = extract_article(test_question)
                test_prompt = prompt_template[args.task_name]['prompt'].format(test_article)
                retrieval_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(test_article)

            test_prompt = "##INSTRUCTION:\n" + test_prompt
            if args.profile:
                test_prompt = "##USER PROFILE:\n" + profile_text + "\n" + test_prompt

            retrieved_texts = []
            retrieved_delta_days = []
            retrieved_order_idx = []
            if args.k > 0 and bm25 is not None:
                top_n = min(args.k, len(history_list))
                retrieved_texts = bm25.get_top_n(retrieval_query.split(" "), history_list, n=top_n)
                if args.use_time_bias or args.use_order_bias:
                    text_to_delta = {}
                    text_to_order = {}
                    query_date = q.get("date", None)
                    query_ord = None
                    if isinstance(query_date, str) and query_date.strip():
                        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m", "%Y/%m"]:
                            try:
                                query_ord = datetime.strptime(query_date.strip(), fmt).toordinal()
                                break
                            except Exception:
                                pass
                    for rec_idx, rec in enumerate(test_data[i]['profile']):
                        rec_text = prompt_template[args.task_name]['retrieval_history'].format(**rec)
                        if args.use_time_bias:
                            rec_ord = None
                            rec_date = rec.get("date", None)
                            if isinstance(rec_date, str) and rec_date.strip():
                                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m", "%Y/%m"]:
                                    try:
                                        rec_ord = datetime.strptime(rec_date.strip(), fmt).toordinal()
                                        break
                                    except Exception:
                                        pass
                            if query_ord is not None and rec_ord is not None:
                                text_to_delta[rec_text] = float(max(0, query_ord - rec_ord))
                            else:
                                text_to_delta[rec_text] = 0.0
                        if args.use_order_bias:
                            text_to_order[rec_text] = float(rec_idx + 1)
                    if args.use_time_bias:
                        retrieved_delta_days = [text_to_delta.get(t, 0.0) for t in retrieved_texts]
                    if args.use_order_bias:
                        retrieved_order_idx = [text_to_order.get(t, 0.0) for t in retrieved_texts]

            test_prompt_list.append(test_prompt)
            query_texts.append(retrieval_query)
            retrieved_texts_list.append(retrieved_texts)
            retrieved_delta_days_list.append(retrieved_delta_days)
            retrieved_order_idx_list.append(retrieved_order_idx)
            question_id_list.append(q['id'])

        prompt_batch_list = split_batch(test_prompt_list, args.batch_size)
        query_batch_list = split_batch(query_texts, args.batch_size)
        retrieved_batch_list = split_batch(retrieved_texts_list, args.batch_size)
        retrieved_delta_batch_list = split_batch(retrieved_delta_days_list, args.batch_size)
        retrieved_order_batch_list = split_batch(retrieved_order_idx_list, args.batch_size)

        out_list = []
        with torch.inference_mode():
            for batch_idx in range(len(prompt_batch_list)):
                prompt_batch = prompt_batch_list[batch_idx]
                query_batch = query_batch_list[batch_idx]
                retrieved_batch = retrieved_batch_list[batch_idx]
                delta_batch = retrieved_delta_batch_list[batch_idx]
                order_batch = retrieved_order_batch_list[batch_idx]

                inputs = tokenizer(prompt_batch, return_tensors="pt", padding=True, return_token_type_ids=False)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                rag_inputs = build_prefix_inputs(
                    tokenizer=tokenizer,
                    query_texts=query_batch,
                    retrieved_texts_list=retrieved_batch,
                    rag_k=max(1, args.k),
                    query_max_len=query_max_len,
                    record_max_len=record_max_len,
                    device=model.device,
                    delta_days_list=delta_batch,
                    order_idx_list=order_batch,
                )

                prefix_embeds = rag_prefix.build_prefix(
                    embedding_layer=model.get_input_embeddings(),
                    **rag_inputs,
                )

                token_embeds = model.get_input_embeddings()(inputs["input_ids"])
                inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

                prefix_mask = torch.ones(
                    (inputs["attention_mask"].size(0), prefix_len),
                    dtype=inputs["attention_mask"].dtype,
                    device=inputs["attention_mask"].device,
                )
                merged_attention_mask = torch.cat([prefix_mask, inputs["attention_mask"]], dim=1)

                with torch.autocast(device_type="cuda"):
                    outputs = model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=merged_attention_mask,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=100,
                    )

                out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                out_list.extend(out_sentence)

        for j in range(len(out_list)):
            output = out_list[j].replace(test_prompt_list[j], '')
            if args.task_name == "citation":
                stripped = output.strip()
                if stripped == "1":
                    output = "[1]"
                elif stripped == "2":
                    output = "[2]"

            question_id = question_id_list[j]
            pred_all.append({"id": question_id, "output": output})

            prompt_output_records.append({
                "id": question_id,
                "user_id": user_id,
                "prompt": "[rag_prefix]\n" + test_prompt_list[j],
                "output": output,
                "gold": golds_dict.get(question_id, ""),
            })

            print(output)

    output_file = {
        'task': name2taskid[args.task_name],
        'golds': pred_all,
        'model': model_name,
    }

    output_dir = f"./output_rag/{args.k}"
    preds_dir = f"{output_dir}/preds"
    os.makedirs(preds_dir, exist_ok=True)

    if args.profile:
        preds_path = f"{preds_dir}/output-task-rag-k{args.k}-{args.task_name}-{model_name.split('/')[-1]}-profile.json"
        prompt_output_path = f"{output_dir}/prompt-output-rag-k{args.k}-{args.task_name}-{model_name.split('/')[-1]}-profile.jsonl"
    else:
        preds_path = f"{preds_dir}/output-task-rag-k{args.k}-{args.task_name}-{model_name.split('/')[-1]}.json"
        prompt_output_path = f"{output_dir}/prompt-output-rag-k{args.k}-{args.task_name}-{model_name.split('/')[-1]}.jsonl"

    with open(preds_path, 'w') as f:
        json.dump(output_file, f, indent=4)

    with open(prompt_output_path, 'w') as f:
        for record in prompt_output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    evaluator = LaMPEvaluation(single_gold_json_file_addr=args.golds_json)
    results = evaluator.evaluate_task(preds_path, name2taskid[args.task_name])
    results = {key: round(value, 3) if isinstance(value, float) else value for key, value in results.items()}
    print(results)


if __name__ == "__main__":
    main()
