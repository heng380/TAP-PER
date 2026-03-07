import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster users by profile text")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--train_profile", type=str, default="")
    parser.add_argument("--infer_profile", type=str, default="")
    parser.add_argument("--train_output_json", type=str, default="")
    parser.add_argument("--infer_output_json", type=str, default="")
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--model_path", type=str, default="/cfs/models/qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_paths(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    task_root = os.path.join(base_dir, "data", args.task_name)
    jsonl_root = os.path.join(task_root, "jsonl_files")

    def pick_existing(candidates):
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]

    train_candidates = [
        os.path.join(task_root, "profile_user_others_new.json"),
        os.path.join(task_root, "profile_user_others.json"),
        os.path.join(jsonl_root, "profile_user_others_new.jsonl"),
    ]
    infer_candidates = [
        os.path.join(task_root, "profile_user_100_new.json"),
        os.path.join(task_root, "profile_user_100.json"),
        os.path.join(jsonl_root, "profile_user_100_new.jsonl"),
    ]

    train_profile = args.train_profile or pick_existing(train_candidates)
    infer_profile = args.infer_profile or pick_existing(infer_candidates)
    train_output_json = args.train_output_json or os.path.join(task_root, "group_mapping_others.json")
    infer_output_json = args.infer_output_json or os.path.join(task_root, "group_mapping_100.json")
    return train_profile, infer_profile, train_output_json, infer_output_json


def _append_record(obj, ids, texts, file_path, row_desc):
    if "id" not in obj or "output" not in obj:
        raise ValueError(f"Missing 'id' or 'output' at {file_path}:{row_desc}")

    text = str(obj["output"]).strip()
    if not text:
        text = " "

    ids.append(obj["id"])
    texts.append(text)


def read_profile_data(file_path):
    ids = []
    texts = []

    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                row = line.strip()
                if not row:
                    continue
                try:
                    obj = json.loads(row)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {file_path}:{line_idx}: {e}")
                _append_record(obj, ids, texts, file_path, line_idx)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file {file_path}: {e}")

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {file_path}")

        for idx, obj in enumerate(data, start=1):
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {file_path}:{idx}")
            _append_record(obj, ids, texts, file_path, idx)

    if not ids:
        raise ValueError(f"No valid rows found in {file_path}")

    return ids, texts


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def build_embeddings(texts, tokenizer, model, batch_size, max_length, device):
    vectors = []
    model.eval()

    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.float().cpu().numpy())

    return np.concatenate(vectors, axis=0)


def build_mapping(ids, groups):
    return [{"id": user_id, "group": int(group)} for user_id, group in zip(ids, groups)]


def write_output(path, task_name, model_path, n_clusters, source_profile, mapping):
    counter = Counter([x["group"] for x in mapping])
    cluster_counts = {str(i): int(counter.get(i, 0)) for i in range(n_clusters)}

    payload = {
        "task_name": task_name,
        "model_path": model_path,
        "n_clusters": n_clusters,
        "source_profile": source_profile,
        "num_records": len(mapping),
        "cluster_counts": cluster_counts,
        "mapping": mapping,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    train_profile, infer_profile, train_output_json, infer_output_json = resolve_paths(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but CUDA is not available")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(args.device)

    train_ids, train_texts = read_profile_data(train_profile)
    infer_ids, infer_texts = read_profile_data(infer_profile)

    train_embeddings = build_embeddings(
        train_texts,
        tokenizer,
        model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    train_groups = kmeans.fit_predict(train_embeddings)

    infer_embeddings = build_embeddings(
        infer_texts,
        tokenizer,
        model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    infer_groups = kmeans.predict(infer_embeddings)

    train_mapping = build_mapping(train_ids, train_groups)
    infer_mapping = build_mapping(infer_ids, infer_groups)

    write_output(
        train_output_json,
        args.task_name,
        args.model_path,
        args.n_clusters,
        train_profile,
        train_mapping,
    )
    write_output(
        infer_output_json,
        args.task_name,
        args.model_path,
        args.n_clusters,
        infer_profile,
        infer_mapping,
    )

    print(f"Saved: {train_output_json} ({len(train_mapping)} records)")
    print(f"Saved: {infer_output_json} ({len(infer_mapping)} records)")


if __name__ == "__main__":
    main()
