import argparse
import json
import os
from datetime import datetime
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


parser = argparse.ArgumentParser(description="Train RAG+PAG jointly from task LoRA: trainable RAG DIN + PAG embedding + mediator LoRA")
parser.add_argument('--model_name', type=str, default='/cfs/models/llama/llama3.1-8B')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--k', type=int, default=10, help='Top-k records for RAG-prefix attention')
parser.add_argument('--cut_off', type=int, default=2048)
parser.add_argument('--max_epoch', type=int, default=3)
parser.add_argument('--prefix_len', type=int, default=8)
parser.add_argument('--task_name', type=str, default='movie_tagging')
parser.add_argument('--add_profile', action='store_true')
parser.add_argument('--access_token', type=str, default=None)
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DDP')
parser.add_argument('--task_ckpt_path', type=str, default='', help='Optional override for task LoRA ckpt (default: k0 task ckpt)')
parser.add_argument('--ragpag_lora_r', type=int, default=4)
parser.add_argument('--ragpag_lora_alpha', type=int, default=8)
parser.add_argument('--ragpag_lora_dropout', type=float, default=0.05)
parser.add_argument('--query_max_len', type=int, default=128)
parser.add_argument('--record_max_len', type=int, default=128)
parser.add_argument('--use_time_bias', action='store_true', help='Enable recency-based bias in RAG attention')
parser.add_argument('--use_order_bias', action='store_true', help='Enable order-index bias in RAG attention')
parser.add_argument('--disable_pag', action='store_true', help='Disable PAG branch and train RAG+mediator only')
parser.add_argument('--preprocess_only', action='store_true')
parser.add_argument('--cache_dir', type=str, default='./cache_rp')
parser.add_argument('--map_num_proc', type=int, default=8)
args = parser.parse_args()

TASK_TRAIN_OVERRIDES = {
    "movie_tagging": {
        "batch_size": 1,
        "learning_rate": 1e-4,
    },
    "citation": {
        "batch_size": 4,
        "learning_rate": 2e-4,
    },
    "news_categorize": {
        "batch_size": 2,
        "learning_rate": 1e-4,
    },
    "product_rating": {
        "batch_size": 4,
        "learning_rate": 2e-4,
    },
    "news_headline": {
        "batch_size": 4,
        "learning_rate": 2e-4,
    },
    "scholarly_title": {
        "batch_size": 2,
        "learning_rate": 1e-4,
    },
    "tweet_paraphrase": {
        "batch_size": 4,
        "learning_rate": 1e-5,
    }
}

if args.task_name in TASK_TRAIN_OVERRIDES:
    override_cfg = TASK_TRAIN_OVERRIDES[args.task_name]
    args.batch_size = override_cfg.get("batch_size", args.batch_size)

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


def build_prompt_pair(idx, profile_item, user_history, profile_text):
    q = safe_truncate_dict(profile_item, limit=768)

    oppu_input_template = prompt_template[args.task_name]['OPPU_input']
    oppu_full_template = prompt_template[args.task_name]['OPPU_full']

    prompt = format_with_required_fields(oppu_input_template, q)
    full_prompt = format_with_required_fields(oppu_full_template, q)

    retrieval_query_template = prompt_template[args.task_name].get('retrieval_query', None)
    if retrieval_query_template is not None:
        query_text = format_with_required_fields(retrieval_query_template, q)
    else:
        query_text = prompt

    retrieved_texts = []
    retrieved_delta_days = []
    retrieved_order_idx = []
    if k > 0 and idx != 0 and format_flag:
        visible_history_list = [safe_truncate_dict(h, limit=768) for h in user_history[:idx]]
        history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]

        if len(history_list) > 0:
            tokenized_corpus = [doc.split(" ") for doc in history_list]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query_text.split(' ')
            top_n = min(args.k, len(history_list))
            retrieved_texts = bm25.get_top_n(tokenized_query, history_list, n=top_n)

            if args.use_time_bias or args.use_order_bias:
                text_to_delta = {}
                text_to_order = {}
                query_date_ord = parse_date_to_ordinal(profile_item.get("date", None))
                for rec_idx, rec in enumerate(visible_history_list):
                    rec_text = prompt_template[args.task_name]['retrieval_history'].format(**rec)
                    if args.use_time_bias:
                        rec_date_ord = parse_date_to_ordinal(rec.get("date", None))
                        if query_date_ord is not None and rec_date_ord is not None:
                            text_to_delta[rec_text] = float(max(0, query_date_ord - rec_date_ord))
                        else:
                            text_to_delta[rec_text] = 0.0
                    if args.use_order_bias:
                        text_to_order[rec_text] = float(rec_idx + 1)

                if args.use_time_bias:
                    retrieved_delta_days = [text_to_delta.get(t, 0.0) for t in retrieved_texts]
                if args.use_order_bias:
                    retrieved_order_idx = [text_to_order.get(t, 0.0) for t in retrieved_texts]

    if args.add_profile and format_flag:
        prompt = profile_text + "\n" + prompt
        full_prompt = profile_text + "\n" + full_prompt

    return prompt, full_prompt, query_text, retrieved_texts, retrieved_delta_days, retrieved_order_idx


def build_train_data(users):
    train_data = []
    for sample in tqdm(users, disable=not is_main_process):
        uid = int(sample['user_id'])
        profile_text = profile_by_user_id.get(uid, "") if args.add_profile else ""

        for idx, profile_item in enumerate(sample['profile']):
            prompt, full_prompt, query_text, retrieved_texts, retrieved_delta_days, retrieved_order_idx = build_prompt_pair(
                idx, profile_item, sample['profile'], profile_text
            )
            train_data.append({
                "prompt": prompt,
                "full_prompt": full_prompt,
                "user_id": uid,
                "query_text": query_text,
                "retrieved_texts": retrieved_texts,
                "retrieved_delta_days": retrieved_delta_days,
                "retrieved_order_idx": retrieved_order_idx,
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

        tokenized_full_prompt["user_id"] = data_point["user_id"]
        tokenized_full_prompt["query_text"] = data_point["query_text"]
        tokenized_full_prompt["retrieved_texts"] = data_point["retrieved_texts"]
        tokenized_full_prompt["retrieved_delta_days"] = data_point.get("retrieved_delta_days", [])
        tokenized_full_prompt["retrieved_order_idx"] = data_point.get("retrieved_order_idx", [])
        return tokenized_full_prompt

    return generate_and_tokenize_prompt


def resolve_task_ckpt_path(task_name, model_name):
    model_short = model_name.split('/')[-1]
    return f"./ckpt/{task_name}/k0-{task_name}-{model_short}-task_LoRA_ckpt"


def create_frozen_task_backbone_and_tokenizer():
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
    # Avoid DDP "marked ready twice" issues with re-entrant checkpointing in LoRA training.
    merged_task_model.gradient_checkpointing_disable()
    return merged_task_model, tokenizer, task_ckpt


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
        record_input_ids,
        record_attention_mask,
        record_valid_mask,
        record_delta_days=None,
        record_order_idx=None,
        use_time_bias=False,
        time_bias_lambda=None,
        use_order_bias=False,
        order_bias_lambda=None,
    ):
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
        mlp_dtype = self.prefix_proj.weight.dtype
        feat = feat.to(mlp_dtype)
        scores = self.attn_mlp(feat).squeeze(-1)

        valid = record_valid_mask.to(scores.dtype)
        scores = scores.masked_fill(valid == 0, -1e4)

        if use_time_bias and record_delta_days is not None and time_bias_lambda is not None:
            recency_penalty = torch.log1p(record_delta_days.to(scores.dtype).clamp_min(0.0))
            scores = scores - time_bias_lambda.to(scores.dtype) * recency_penalty
            scores = scores.masked_fill(valid == 0, -1e4)

        if use_order_bias and record_order_idx is not None and order_bias_lambda is not None:
            order_penalty = torch.log1p(record_order_idx.to(scores.dtype).clamp_min(0.0))
            scores = scores - order_bias_lambda.to(scores.dtype) * order_penalty
            scores = scores.masked_fill(valid == 0, -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = attn * valid
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        context = (attn.unsqueeze(-1) * rec_vec).sum(dim=1)

        context = context.to(self.prefix_proj.weight.dtype)
        prefix = self.prefix_proj(context).view(batch_size, self.prefix_len, self.hidden_size)
        return prefix


class RagPagPrefixModel(nn.Module):
    def __init__(self, base_model, hidden_size, prefix_len, num_users, user_id_to_index, use_time_bias=False, use_order_bias=False, disable_pag=False):
        super().__init__()
        self.base_model = base_model
        self.prefix_len = prefix_len
        self.hidden_size = hidden_size

        self.disable_pag = bool(disable_pag)
        self.user_id_to_index = {int(k): int(v) for k, v in user_id_to_index.items()}
        if not self.disable_pag:
            self.pag_user_embedding = nn.Embedding(num_users, hidden_size * prefix_len)
            nn.init.normal_(self.pag_user_embedding.weight, mean=0.0, std=0.02)
        else:
            self.pag_user_embedding = None

        self.rag_prefix_module = RagPrefixModule(
            hidden_size=hidden_size,
            prefix_len=prefix_len,
        )
        self.use_time_bias = use_time_bias
        if self.use_time_bias:
            self.time_bias_lambda = nn.Parameter(torch.empty(1))
            nn.init.normal_(self.time_bias_lambda, mean=0.0, std=0.02)
        else:
            self.register_parameter('time_bias_lambda', None)

        self.use_order_bias = use_order_bias
        if self.use_order_bias:
            self.order_bias_lambda = nn.Parameter(torch.empty(1))
            nn.init.normal_(self.order_bias_lambda, mean=0.0, std=0.02)
        else:
            self.register_parameter('order_bias_lambda', None)

    def _mean_pool_embeds(self, input_ids, attention_mask):
        embeds = self.base_model.get_input_embeddings()(input_ids)
        mask = attention_mask.unsqueeze(-1).to(embeds.dtype)
        summed = (embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def _build_pag_prefix(self, user_ids, device, dtype):
        batch_size = user_ids.size(0)
        prefix = torch.zeros(
            batch_size,
            self.prefix_len,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )

        valid_positions = []
        valid_indices = []
        for pos, uid in enumerate(user_ids.tolist()):
            user_idx = self.user_id_to_index.get(int(uid), None)
            if user_idx is not None:
                valid_positions.append(pos)
                valid_indices.append(user_idx)

        if len(valid_positions) > 0:
            idx_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
            user_embed = self.pag_user_embedding(idx_tensor).view(-1, self.prefix_len, self.hidden_size).to(dtype)
            pos_tensor = torch.tensor(valid_positions, dtype=torch.long, device=device)
            prefix[pos_tensor] = user_embed

        return prefix

    def _build_rag_prefix(
        self,
        query_input_ids,
        query_attention_mask,
        record_input_ids,
        record_attention_mask,
        record_valid_mask,
        record_delta_days=None,
        record_order_idx=None,
    ):
        return self.rag_prefix_module.build_prefix(
            embedding_layer=self.base_model.get_input_embeddings(),
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            record_input_ids=record_input_ids,
            record_attention_mask=record_attention_mask,
            record_valid_mask=record_valid_mask,
            record_delta_days=record_delta_days,
            record_order_idx=record_order_idx,
            use_time_bias=self.use_time_bias,
            time_bias_lambda=self.time_bias_lambda,
            use_order_bias=self.use_order_bias,
            order_bias_lambda=self.order_bias_lambda,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        user_id=None,
        query_input_ids=None,
        query_attention_mask=None,
        record_input_ids=None,
        record_attention_mask=None,
        record_valid_mask=None,
        record_delta_days=None,
        record_order_idx=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required")

        token_embeds = self.base_model.get_input_embeddings()(input_ids)

        if self.disable_pag or user_id is None:
            pag_prefix = torch.zeros(
                token_embeds.size(0),
                self.prefix_len,
                token_embeds.size(-1),
                dtype=token_embeds.dtype,
                device=token_embeds.device,
            )
        else:
            pag_prefix = self._build_pag_prefix(user_id, token_embeds.device, token_embeds.dtype)

        if (
            query_input_ids is None
            or query_attention_mask is None
            or record_input_ids is None
            or record_attention_mask is None
            or record_valid_mask is None
        ):
            rag_prefix = torch.zeros(
                token_embeds.size(0),
                self.prefix_len,
                token_embeds.size(-1),
                dtype=token_embeds.dtype,
                device=token_embeds.device,
            )
        else:
            rag_prefix = self._build_rag_prefix(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                record_input_ids=record_input_ids,
                record_attention_mask=record_attention_mask,
                record_valid_mask=record_valid_mask,
                record_delta_days=record_delta_days,
                record_order_idx=record_order_idx,
            )

        rag_prefix = rag_prefix.to(token_embeds.dtype)
        combined_prefix = (pag_prefix + rag_prefix).to(token_embeds.dtype)
        inputs_embeds = torch.cat([combined_prefix, token_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prefix_mask = torch.ones(
            (attention_mask.size(0), self.prefix_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        merged_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

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
            attention_mask=merged_attention_mask,
            labels=labels,
            **kwargs,
        )


class RpCollator:
    def __init__(self, tokenizer, rag_k, query_max_len=128, record_max_len=128):
        self.tokenizer = tokenizer
        self.rag_k = max(1, rag_k)
        self.query_max_len = query_max_len
        self.record_max_len = record_max_len
        self.inner_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )

    def __call__(self, features):
        user_ids = torch.tensor([int(f.get("user_id", -1)) for f in features], dtype=torch.long)
        query_texts = [str(f.get("query_text", "")) for f in features]

        record_texts_flat = []
        valid_mask = []
        delta_days_flat = []
        order_idx_flat = []
        for f in features:
            recs = f.get("retrieved_texts", [])
            if not isinstance(recs, list):
                recs = []
            recs = [str(r) for r in recs][: self.rag_k]
            deltas = f.get("retrieved_delta_days", [])
            if not isinstance(deltas, list):
                deltas = []
            deltas = [float(x) if x is not None else 0.0 for x in deltas][: self.rag_k]
            orders = f.get("retrieved_order_idx", [])
            if not isinstance(orders, list):
                orders = []
            orders = [float(x) if x is not None else 0.0 for x in orders][: self.rag_k]

            if len(recs) < self.rag_k:
                recs = recs + [""] * (self.rag_k - len(recs))
            if len(deltas) < self.rag_k:
                deltas = deltas + [0.0] * (self.rag_k - len(deltas))
            if len(orders) < self.rag_k:
                orders = orders + [0.0] * (self.rag_k - len(orders))

            for rec, dd, oi in zip(recs, deltas, orders):
                record_texts_flat.append(rec)
                valid_mask.append(1 if rec.strip() else 0)
                delta_days_flat.append(float(dd))
                order_idx_flat.append(float(oi))

        stripped = []
        for f in features:
            item = dict(f)
            item.pop("user_id", None)
            item.pop("query_text", None)
            item.pop("retrieved_texts", None)
            item.pop("retrieved_delta_days", None)
            item.pop("retrieved_order_idx", None)
            stripped.append(item)

        batch = self.inner_collator(stripped)

        query_tokens = self.tokenizer(
            query_texts,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        rec_tokens = self.tokenizer(
            record_texts_flat,
            padding=True,
            truncation=True,
            max_length=self.record_max_len,
            return_tensors="pt",
        )

        batch_size = len(features)
        batch["user_id"] = user_ids
        batch["query_input_ids"] = query_tokens["input_ids"]
        batch["query_attention_mask"] = query_tokens["attention_mask"]
        batch["record_input_ids"] = rec_tokens["input_ids"].view(batch_size, self.rag_k, -1)
        batch["record_attention_mask"] = rec_tokens["attention_mask"].view(batch_size, self.rag_k, -1)
        batch["record_valid_mask"] = torch.tensor(valid_mask, dtype=torch.long).view(batch_size, self.rag_k)
        batch["record_delta_days"] = torch.tensor(delta_days_flat, dtype=torch.float).view(batch_size, self.rag_k)
        batch["record_order_idx"] = torch.tensor(order_idx_flat, dtype=torch.float).view(batch_size, self.rag_k)
        return batch


def print_trainable_parameters(model):
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    if is_main_process:
        ratio = 100 * trainable / total if total > 0 else 0
        print(f"trainable params: {trainable} || all params: {total} || trainable%: {ratio:.4f}")


effective_lr = TASK_TRAIN_OVERRIDES.get(args.task_name, {}).get("learning_rate", 2e-4)

training_arguments = transformers.TrainingArguments(
    output_dir='./ckpt/trainer_tmp_rp/',
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    optim='adamw_torch',
    num_train_epochs=max_epoch,
    save_steps=1e9,
    logging_steps=10,
    learning_rate=effective_lr,
    weight_decay=1e-2,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type='linear',
    report_to='none',
    local_rank=args.local_rank,
    ddp_find_unused_parameters=False,
    save_strategy='no',
)

users = train
if is_main_process:
    print(f"Loaded train users (top100): {len(users)}")

cache_tag = f"k{args.k}{'-profile' if args.add_profile else ''}{'-tb' if args.use_time_bias else ''}{'-ob' if args.use_order_bias else ''}{'-nopag' if args.disable_pag else ''}"
cache_path = os.path.join(
    args.cache_dir,
    args.task_name,
    cache_tag,
)

train_dataset = None
if os.path.exists(cache_path):
    if is_main_process:
        print(f"Load cached dataset: {cache_path}")
    train_dataset = load_from_disk(cache_path)
    if is_main_process:
        print(f"Cached samples: {len(train_dataset)}")
else:
    train_data = build_train_data(users)
    if is_main_process:
        print(f"Train samples: {len(train_data)}")

    if len(train_data) == 0:
        raise ValueError("No train samples built from user records.")

    if is_main_process:
        preview_cnt = min(2, len(train_data))
        for preview_idx in range(preview_cnt):
            print(f"[RP] Preview sample {preview_idx}")
            print(f"prompt:\n{train_data[preview_idx]['prompt']}")
            print(f"full_prompt:\n{train_data[preview_idx]['full_prompt']}")
            print(f"user_id: {train_data[preview_idx]['user_id']}")
            print(f"query_text:\n{train_data[preview_idx]['query_text']}")
            print(f"retrieved_texts:\n{train_data[preview_idx]['retrieved_texts']}")

    _, tokenizer_for_cache, _ = create_frozen_task_backbone_and_tokenizer()
    generate_and_tokenize_prompt_for_cache = create_tokenizers(tokenizer_for_cache)
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(generate_and_tokenize_prompt_for_cache, num_proc=args.map_num_proc)
    train_dataset = train_dataset.shuffle()

    if is_main_process:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        train_dataset.save_to_disk(cache_path)
        print(f"Saved cached dataset: {cache_path}")

if args.preprocess_only:
    if is_main_process:
        print("Preprocess-only mode, skip training.")
    raise SystemExit(0)

frozen_model, tokenizer, task_ckpt_path = create_frozen_task_backbone_and_tokenizer()


def build_ragpag_lora_config():
    return LoraConfig(
        r=args.ragpag_lora_r,
        lora_alpha=args.ragpag_lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=args.ragpag_lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

if is_main_process:
    print(f"Loaded frozen task LoRA: {task_ckpt_path}")
    print("Initialize RAG DIN module + PAG user embedding + mediator LoRA from scratch")
    print(f"Disable PAG branch: {args.disable_pag}")
    print(f"Time bias enabled: {args.use_time_bias}")
    print(f"Order bias enabled: {args.use_order_bias}")

ragpag_lora_config = build_ragpag_lora_config()
mediator_model = get_peft_model(frozen_model, ragpag_lora_config)

prefix_len = int(args.prefix_len)
hidden_size = int(mediator_model.config.hidden_size)

unique_user_ids = sorted({int(u['user_id']) for u in users})
user_id_to_index = {uid: idx for idx, uid in enumerate(unique_user_ids)}

model = RagPagPrefixModel(
    mediator_model,
    hidden_size=hidden_size,
    prefix_len=prefix_len,
    num_users=len(unique_user_ids),
    user_id_to_index=user_id_to_index,
    use_time_bias=args.use_time_bias,
    use_order_bias=args.use_order_bias,
    disable_pag=args.disable_pag,
)

if args.local_rank != -1:
    model = model.to(torch.device("cuda", args.local_rank))

if is_main_process:
    print_trainable_parameters(model)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_arguments,
    data_collator=RpCollator(
        tokenizer,
        rag_k=args.k,
        query_max_len=args.query_max_len,
        record_max_len=args.record_max_len,
    ),
)

trainer.train()

os.makedirs(f"./ckpt/{args.task_name}", exist_ok=True)
model_short = model_name.split('/')[-1]
suffix = "-profile" if args.add_profile else ""
bias_tag = f"{'-tb' if args.use_time_bias else ''}{'-ob' if args.use_order_bias else ''}{'-nopag' if args.disable_pag else ''}"
ragpag_lora_output = f"./ckpt/{args.task_name}/k{args.k}-{args.task_name}-{model_short}{suffix}{bias_tag}-ragpag-lora"
ragpag_prefix_output = f"./ckpt/{args.task_name}/k{args.k}-{args.task_name}-{model_short}{suffix}{bias_tag}-ragpag-prefix.pt"

if is_main_process:
    model.base_model.save_pretrained(ragpag_lora_output)
    payload = {
        "task_name": args.task_name,
        "model_name": args.model_name,
        "k": args.k,
        "add_profile": args.add_profile,
        "base_task_ckpt": task_ckpt_path,
        "hidden_size": hidden_size,
        "prefix_len": prefix_len,
        "query_max_len": args.query_max_len,
        "record_max_len": args.record_max_len,
        "user_id_to_index": user_id_to_index,
        "disable_pag": args.disable_pag,
        "pag_user_embedding": ({k: v.detach().cpu() for k, v in model.pag_user_embedding.state_dict().items()} if model.pag_user_embedding is not None else None),
        "attn_mlp": {k: v.detach().cpu() for k, v in model.rag_prefix_module.attn_mlp.state_dict().items()},
        "prefix_proj": {k: v.detach().cpu() for k, v in model.rag_prefix_module.prefix_proj.state_dict().items()},
        "ragpag_lora_path": ragpag_lora_output,
        "ragpag_lora_r": args.ragpag_lora_r,
        "ragpag_lora_alpha": args.ragpag_lora_alpha,
        "ragpag_lora_dropout": args.ragpag_lora_dropout,
        "use_time_bias": args.use_time_bias,
        "time_bias_lambda": (float(model.time_bias_lambda.detach().cpu().item()) if model.time_bias_lambda is not None else None),
        "use_order_bias": args.use_order_bias,
        "order_bias_lambda": (float(model.order_bias_lambda.detach().cpu().item()) if model.order_bias_lambda is not None else None),
    }
    torch.save(payload, ragpag_prefix_output)
    print(f"Saved RAGPAG LoRA: {ragpag_lora_output}")
    print(f"Saved RAGPAG prefix module: {ragpag_prefix_output}")


del trainer
del model
del frozen_model
torch.cuda.empty_cache()
