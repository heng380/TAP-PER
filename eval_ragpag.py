import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
from peft import PeftModel
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
    parser = argparse.ArgumentParser(description="RAGPAG inference: trainable PAG prefix from RAGPAG + frozen RAG prefix + RAGPAG mediator LoRA")
    parser.add_argument('--model_name', type=str, default=os.environ.get('MODEL_NAME', 'meta-llama/Llama-3.1-8B'))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--k', type=int, default=0, help='Number of previous records for RAG-prefix attention; <=0 attends over all visible history')
    parser.add_argument('--task_name', type=str, default='movie_tagging')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--access_token', type=str, default=None)
    parser.add_argument('--golds_json', type=str, default='', help='Gold labels json file')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA device id')
    parser.add_argument('--task_ckpt_path', type=str, default='', help='Task-level LoRA checkpoint path')
    parser.add_argument('--ragpag_prefix_ckpt_path', type=str, default='', help='RAGPAG prefix .pt checkpoint path')
    parser.add_argument('--ragpag_lora_path', type=str, default='', help='RAGPAG mediator LoRA checkpoint path')
    parser.add_argument('--query_max_len', type=int, default=128)
    parser.add_argument('--record_max_len', type=int, default=128)
    parser.add_argument('--record_encode_batch_size', type=int, default=64)
    parser.add_argument('--use_time_bias', action='store_true', help='Enable recency-based bias in RAG attention')
    parser.add_argument('--use_order_bias', action='store_true', help='Enable order-index bias in RAG attention')
    parser.add_argument('--disable_pag', action='store_true', help='Disable PAG branch and evaluate with RAG prefix only')
    return parser.parse_args()


def resolve_task_ckpt_path(task_name, model_name):
    model_short = model_name.split('/')[-1]
    return f"./ckpt/{task_name}/k0-{task_name}-{model_short}-task_LoRA_ckpt"


def rag_k_to_tag(k):
    return "all" if int(k) <= 0 else str(int(k))


def resolve_ragpag_prefix_ckpt_path(task_name, model_name, k, profile, use_time_bias=False, use_order_bias=False, disable_pag=False):
    model_short = model_name.split('/')[-1]
    suffix = "-profile" if profile else ""
    bias_tag = f"{'-tb' if use_time_bias else ''}{'-ob' if use_order_bias else ''}{'-nopag' if disable_pag else ''}"
    return f"./ckpt/{task_name}/k{rag_k_to_tag(k)}-{task_name}-{model_short}{suffix}{bias_tag}-ragpag-prefix.pt"


def resolve_ragpag_lora_path(task_name, model_name, k, profile, use_time_bias=False, use_order_bias=False, disable_pag=False):
    model_short = model_name.split('/')[-1]
    suffix = "-profile" if profile else ""
    bias_tag = f"{'-tb' if use_time_bias else ''}{'-ob' if use_order_bias else ''}{'-nopag' if disable_pag else ''}"
    return f"./ckpt/{task_name}/k{rag_k_to_tag(k)}-{task_name}-{model_short}{suffix}{bias_tag}-ragpag-lora"


def parse_date_to_ordinal(value):
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m", "%Y/%m"]:
        try:
            return datetime.strptime(s, fmt).toordinal()
        except Exception:
            pass
    return None


class RagPrefixModule(nn.Module):
    def __init__(self, hidden_size, prefix_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.prefix_len = prefix_len
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.prefix_proj = nn.Linear(hidden_size, hidden_size * prefix_len)

    def _mean_pool_embeds(self, embeds, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(embeds.dtype)
        summed = (embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def build_prefix(
        self,
        embedding_layer,
        query_input_ids,
        query_attention_mask,
        record_embeds,
        record_valid_mask,
        record_delta_days=None,
        record_order_idx=None,
        use_time_bias=False,
        time_bias_lambda=None,
        use_order_bias=False,
        order_bias_lambda=None,
    ):
        batch_size = query_input_ids.size(0)
        q_embeds = embedding_layer(query_input_ids)
        q_vec = self._mean_pool_embeds(q_embeds, query_attention_mask)
        rec_vec = record_embeds.to(q_vec.dtype)

        q_expand = q_vec.unsqueeze(1).expand_as(rec_vec)
        feat = torch.cat([q_expand, rec_vec, q_expand - rec_vec, q_expand * rec_vec], dim=-1)
        mlp_dtype = self.prefix_proj.weight.dtype
        feat = feat.to(mlp_dtype)
        scores = self.attn_mlp(feat).squeeze(-1)

        valid = record_valid_mask.to(scores.dtype)
        scores = scores.masked_fill(valid == 0, -1e4)

        if use_time_bias and record_delta_days is not None and time_bias_lambda is not None:
            recency_penalty = torch.log1p(record_delta_days.to(scores.dtype).clamp_min(0.0))
            scores = scores - torch.as_tensor(time_bias_lambda, dtype=scores.dtype, device=scores.device) * recency_penalty
            scores = scores.masked_fill(valid == 0, -1e4)

        if use_order_bias and record_order_idx is not None and order_bias_lambda is not None:
            order_penalty = torch.log1p(record_order_idx.to(scores.dtype).clamp_min(0.0))
            scores = scores - torch.as_tensor(order_bias_lambda, dtype=scores.dtype, device=scores.device) * order_penalty
            scores = scores.masked_fill(valid == 0, -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = attn * valid
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        context = (attn.unsqueeze(-1) * rec_vec).sum(dim=1)
        prefix = self.prefix_proj(context).view(batch_size, self.prefix_len, self.hidden_size)
        return prefix


def mean_pool_token_embeddings(embedding_layer, input_ids, attention_mask):
    embeds = embedding_layer(input_ids)
    mask = attention_mask.unsqueeze(-1).to(embeds.dtype)
    summed = (embeds * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


@torch.no_grad()
def encode_history_records(tokenizer, embedding_layer, history_texts, record_max_len, device, batch_size=64):
    hidden_size = embedding_layer.weight.shape[-1]
    if not history_texts:
        return torch.empty(0, hidden_size, dtype=embedding_layer.weight.dtype)

    encoded_chunks = []
    for start in range(0, len(history_texts), batch_size):
        batch_texts = history_texts[start:start + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=record_max_len,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        pooled = mean_pool_token_embeddings(embedding_layer, input_ids, attention_mask)
        encoded_chunks.append(pooled.detach().cpu())
    return torch.cat(encoded_chunks, dim=0)


def limit_records(values, rag_k):
    if int(rag_k) > 0:
        return values[-int(rag_k):]
    return values


def build_prefix_inputs(tokenizer, query_texts, record_indices_list, record_embedding_bank, rag_k, query_max_len, device, retrieved_delta_days_list=None, retrieved_order_idx_list=None):
    batch_size = len(query_texts)

    record_indices_list = [
        limit_records([int(x) for x in indices], rag_k) if isinstance(indices, list) else []
        for indices in record_indices_list
    ]
    batch_rag_k = max(1, max((len(indices) for indices in record_indices_list), default=0))
    record_embeds = torch.zeros(
        batch_size,
        batch_rag_k,
        record_embedding_bank.shape[-1],
        dtype=record_embedding_bank.dtype,
        device=device,
    )
    valid_mask = []
    delta_days_flat = []
    order_idx_flat = []

    if retrieved_delta_days_list is None:
        retrieved_delta_days_list = [[] for _ in range(batch_size)]
    if retrieved_order_idx_list is None:
        retrieved_order_idx_list = [[] for _ in range(batch_size)]

    for row_idx, (indices, deltas, orders) in enumerate(zip(record_indices_list, retrieved_delta_days_list, retrieved_order_idx_list)):
        deltas = deltas if isinstance(deltas, list) else []
        deltas = limit_records([float(x) if x is not None else 0.0 for x in deltas], rag_k)

        orders = orders if isinstance(orders, list) else []
        orders = limit_records([float(x) if x is not None else 0.0 for x in orders], rag_k)

        if len(deltas) < len(indices):
            deltas = deltas + [0.0] * (len(indices) - len(deltas))
        if len(orders) < len(indices):
            orders = orders + [0.0] * (len(indices) - len(orders))

        for col_idx, rec_idx in enumerate(indices):
            if rec_idx < 0 or rec_idx >= record_embedding_bank.size(0):
                raise IndexError(f"record index {rec_idx} out of range for evaluation history")
            record_embeds[row_idx, col_idx] = record_embedding_bank[rec_idx].to(device)

        for col_idx in range(batch_rag_k):
            is_valid = col_idx < len(indices)
            valid_mask.append(1 if is_valid else 0)
            delta_days_flat.append(float(deltas[col_idx]) if is_valid else 0.0)
            order_idx_flat.append(float(orders[col_idx]) if is_valid else 0.0)

    q_tokens = tokenizer(
        query_texts,
        padding=True,
        truncation=True,
        max_length=query_max_len,
        return_tensors="pt",
    )
    query_input_ids = q_tokens["input_ids"].to(device)
    query_attention_mask = q_tokens["attention_mask"].to(device)
    record_valid_mask = torch.tensor(valid_mask, dtype=torch.long, device=device).view(batch_size, batch_rag_k)

    record_delta_days = torch.tensor(delta_days_flat, dtype=torch.float, device=device).view(batch_size, batch_rag_k)
    record_order_idx = torch.tensor(order_idx_flat, dtype=torch.float, device=device).view(batch_size, batch_rag_k)

    return {
        "query_input_ids": query_input_ids,
        "query_attention_mask": query_attention_mask,
        "record_embeds": record_embeds,
        "record_valid_mask": record_valid_mask,
        "record_delta_days": record_delta_days,
        "record_order_idx": record_order_idx,
    }


def build_pag_prefix_from_user_ids(user_ids, user_id_to_index, pag_embedding_layer, prefix_len, hidden_size, device, dtype):
    batch_size = user_ids.size(0)
    prefix = torch.zeros(batch_size, prefix_len, hidden_size, device=device, dtype=dtype)

    valid_positions = []
    valid_indices = []
    for pos, uid in enumerate(user_ids.tolist()):
        user_idx = user_id_to_index.get(int(uid), None)
        if user_idx is not None:
            valid_positions.append(pos)
            valid_indices.append(user_idx)

    if len(valid_positions) > 0:
        idx_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
        user_embed = pag_embedding_layer(idx_tensor).view(-1, prefix_len, hidden_size).to(dtype)
        pos_tensor = torch.tensor(valid_positions, dtype=torch.long, device=device)
        prefix[pos_tensor] = user_embed

    return prefix


def main():
    args = parse_args()
    model_name = args.model_name
    task_name = args.task_name
    rag_k_tag = rag_k_to_tag(args.k)
    rag_history_label = "all_visible_history" if args.k <= 0 else f"last_{args.k}_visible_history"

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
    ragpag_prefix_ckpt = args.ragpag_prefix_ckpt_path or resolve_ragpag_prefix_ckpt_path(
        task_name, model_name, args.k, args.profile, args.use_time_bias, args.use_order_bias, args.disable_pag
    )
    ragpag_lora_path = args.ragpag_lora_path or resolve_ragpag_lora_path(
        task_name, model_name, args.k, args.profile, args.use_time_bias, args.use_order_bias, args.disable_pag
    )

    if not os.path.exists(task_ckpt_path):
        raise FileNotFoundError(f"Task LoRA checkpoint not found: {task_ckpt_path}")
    if not os.path.exists(ragpag_prefix_ckpt):
        raise FileNotFoundError(f"RAGPAG prefix checkpoint not found: {ragpag_prefix_ckpt}")
    if not os.path.exists(ragpag_lora_path):
        raise FileNotFoundError(f"RAGPAG LoRA path not found: {ragpag_lora_path}")

    print(f"Loaded task LoRA path: {task_ckpt_path}")
    print(f"Loaded RAGPAG prefix .pt path: {ragpag_prefix_ckpt}")
    print(f"Loaded RAGPAG mediator LoRA path: {ragpag_lora_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=False,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(f"cuda:{args.cuda_id}")

    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.bos_token_id = tokenizer.bos_token_id

    task_model = PeftModel.from_pretrained(base_model, task_ckpt_path, is_trainable=False)
    model = task_model.merge_and_unload()

    ragpag_model = PeftModel.from_pretrained(model, ragpag_lora_path, is_trainable=False)
    model = ragpag_model.merge_and_unload()

    model.eval()
    model.config.use_cache = True

    ragpag_payload = torch.load(ragpag_prefix_ckpt, map_location='cpu')

    user_id_to_index = {int(k): int(v) for k, v in ragpag_payload.get("user_id_to_index", {}).items()}
    disable_pag = bool(ragpag_payload.get("disable_pag", False) or args.disable_pag)
    prefix_len = int(ragpag_payload.get("prefix_len", 8))
    hidden_size = int(ragpag_payload.get("hidden_size", model.config.hidden_size))
    query_max_len = int(ragpag_payload.get("query_max_len", args.query_max_len))
    record_max_len = int(ragpag_payload.get("record_max_len", args.record_max_len))
    use_time_bias = bool(ragpag_payload.get("use_time_bias", args.use_time_bias))
    use_order_bias = bool(ragpag_payload.get("use_order_bias", args.use_order_bias))
    time_bias_lambda = ragpag_payload.get("time_bias_lambda", None)
    order_bias_lambda = ragpag_payload.get("order_bias_lambda", None)

    pag_dtype = model.get_input_embeddings().weight.dtype
    pag_embedding = None
    if not disable_pag:
        pag_embedding = nn.Embedding(len(user_id_to_index), hidden_size * prefix_len)
        pag_embedding.load_state_dict(ragpag_payload["pag_user_embedding"])
        pag_embedding = pag_embedding.to(model.device, dtype=pag_dtype)
        pag_embedding.eval()

    rag_prefix = RagPrefixModule(hidden_size=hidden_size, prefix_len=prefix_len)
    rag_prefix.attn_mlp.load_state_dict(ragpag_payload["attn_mlp"])
    rag_prefix.prefix_proj.load_state_dict(ragpag_payload["prefix_proj"])
    rag_prefix = rag_prefix.to(model.device, dtype=pag_dtype)
    rag_prefix.eval()

    print(f"Disable PAG branch: {disable_pag}")
    print(f"Use time bias: {use_time_bias}, lambda={time_bias_lambda}")
    print(f"Use order bias: {use_order_bias}, lambda={order_bias_lambda}")

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

        history_records = []
        history_list = []
        record_embedding_bank = torch.empty(0, hidden_size, dtype=model.get_input_embeddings().weight.dtype)
        for record in test_data[i]['profile']:
            safe_record = {}
            for key, value in record.items():
                if isinstance(value, str):
                    safe_record[key] = get_first_k_tokens(value, 768)
                else:
                    safe_record[key] = value
            history_records.append(safe_record)

        history_list = [prompt_template[args.task_name]['retrieval_history'].format(**record) for record in history_records]
        if len(history_list) > 0:
            record_embedding_bank = encode_history_records(
                tokenizer=tokenizer,
                embedding_layer=model.get_input_embeddings(),
                history_texts=history_list,
                record_max_len=record_max_len,
                device=model.device,
                batch_size=args.record_encode_batch_size,
            )

        test_prompt_list = []
        query_texts = []
        record_indices_list = []
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

            record_indices = []
            retrieved_delta_days = []
            retrieved_order_idx = []
            if len(history_list) > 0:
                start_idx = 0 if args.k <= 0 else max(0, len(history_list) - args.k)
                record_indices = list(range(start_idx, len(history_list)))
                if use_time_bias or use_order_bias:
                    query_date_ord = parse_date_to_ordinal(q.get("date", None))
                    if query_date_ord is None:
                        query_date_ord = parse_date_to_ordinal(history_records[-1].get("date", None))
                    for rec_idx in record_indices:
                        rec = history_records[rec_idx]
                        if use_time_bias:
                            rec_date_ord = parse_date_to_ordinal(rec.get("date", None))
                            if query_date_ord is not None and rec_date_ord is not None:
                                retrieved_delta_days.append(float(max(0, query_date_ord - rec_date_ord)))
                            else:
                                retrieved_delta_days.append(0.0)
                        if use_order_bias:
                            retrieved_order_idx.append(float(len(history_records) - rec_idx))

            test_prompt_list.append(test_prompt)
            query_texts.append(retrieval_query)
            record_indices_list.append(record_indices)
            retrieved_delta_days_list.append(retrieved_delta_days)
            retrieved_order_idx_list.append(retrieved_order_idx)
            question_id_list.append(q['id'])

        prompt_batch_list = split_batch(test_prompt_list, args.batch_size)
        query_batch_list = split_batch(query_texts, args.batch_size)
        record_indices_batch_list = split_batch(record_indices_list, args.batch_size)
        delta_batch_list = split_batch(retrieved_delta_days_list, args.batch_size)
        order_batch_list = split_batch(retrieved_order_idx_list, args.batch_size)

        out_list = []
        with torch.inference_mode():
            for batch_idx in range(len(prompt_batch_list)):
                prompt_batch = prompt_batch_list[batch_idx]
                query_batch = query_batch_list[batch_idx]
                record_indices_batch = record_indices_batch_list[batch_idx]
                delta_batch = delta_batch_list[batch_idx]
                order_batch = order_batch_list[batch_idx]

                inputs = tokenizer(prompt_batch, return_tensors="pt", padding=True, return_token_type_ids=False)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                rag_inputs = build_prefix_inputs(
                    tokenizer=tokenizer,
                    query_texts=query_batch,
                    record_indices_list=record_indices_batch,
                    record_embedding_bank=record_embedding_bank,
                    rag_k=args.k,
                    query_max_len=query_max_len,
                    device=model.device,
                    retrieved_delta_days_list=delta_batch,
                    retrieved_order_idx_list=order_batch,
                )

                rag_prefix_embeds = rag_prefix.build_prefix(
                    embedding_layer=model.get_input_embeddings(),
                    use_time_bias=use_time_bias,
                    time_bias_lambda=time_bias_lambda,
                    use_order_bias=use_order_bias,
                    order_bias_lambda=order_bias_lambda,
                    **rag_inputs,
                )

                token_embeds = model.get_input_embeddings()(inputs["input_ids"])
                rag_prefix_embeds = rag_prefix_embeds.to(token_embeds.dtype)

                if disable_pag:
                    combined_prefix = rag_prefix_embeds
                else:
                    user_ids = torch.tensor([user_id] * len(prompt_batch), dtype=torch.long, device=model.device)
                    pag_prefix_embeds = build_pag_prefix_from_user_ids(
                        user_ids=user_ids,
                        user_id_to_index=user_id_to_index,
                        pag_embedding_layer=pag_embedding,
                        prefix_len=prefix_len,
                        hidden_size=hidden_size,
                        device=model.device,
                        dtype=rag_prefix_embeds.dtype,
                    )
                    combined_prefix = (pag_prefix_embeds + rag_prefix_embeds).to(token_embeds.dtype)

                inputs_embeds = torch.cat([combined_prefix, token_embeds], dim=1)

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
                "prompt": (
                    f"[ragpag_prefix=preencoded_{rag_history_label}_rag_only(no_pag)]\n"
                    if disable_pag
                    else f"[ragpag_prefix=pag+preencoded_{rag_history_label}_rag(joint-trained)]\n"
                ) + test_prompt_list[j],
                "output": output,
                "gold": golds_dict.get(question_id, ""),
            })

            print(output)

    output_file = {
        'task': name2taskid[args.task_name],
        'golds': pred_all,
        'model': model_name,
    }

    output_dir = f"./output_ragpag/{rag_k_tag}"
    preds_dir = f"{output_dir}/preds"
    os.makedirs(preds_dir, exist_ok=True)

    if args.profile:
        preds_path = f"{preds_dir}/output-task-ragpag-k{rag_k_tag}-{args.task_name}-{model_name.split('/')[-1]}-profile.json"
        prompt_output_path = f"{output_dir}/prompt-output-ragpag-k{rag_k_tag}-{args.task_name}-{model_name.split('/')[-1]}-profile.jsonl"
    else:
        preds_path = f"{preds_dir}/output-task-ragpag-k{rag_k_tag}-{args.task_name}-{model_name.split('/')[-1]}.json"
        prompt_output_path = f"{output_dir}/prompt-output-ragpag-k{rag_k_tag}-{args.task_name}-{model_name.split('/')[-1]}.jsonl"

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
