"""Opt-in standalone CP-02 dense evaluation. Never imported by the MCP server.

Inputs contain public/synthetic text only. Output contains identifiers and
metrics, not input text, private paths, vectors or model payloads. Run each
model in its own interpreter to isolate resource measurements and indexes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import threading
import time

def _dcg(grades):
    return sum((2 ** grade - 1) / math.log2(i + 2)
               for i, grade in enumerate(grades[:10]))


def ndcg_at_10(grades, qrels):
    ideal = _dcg(sorted(qrels, reverse=True))
    if not ideal:
        raise ValueError("positive qrels required")
    return _dcg(grades) / ideal


def recall_at_10(found, total):
    if total <= 0:
        raise ValueError("positive qrels required")
    return found / total


def percentiles(samples):
    ordered = sorted(samples)
    return tuple(ordered[max(1, math.ceil(p * len(ordered))) - 1] for p in (.5, .95))

MODELS = {
    "jina": ("jinaai/jina-embeddings-v5-text-nano", "8a7f00aac812071b69403df470f1038ec85f8925"),
    "bge": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
    "gemma": ("google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b"),
}


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_dataset(data, *, allow_unretrievable_qrels=False):
    records, queries = data["records"], data["queries"]
    ids = [r["id"] for r in records]
    qids = [q["id"] for q in queries]
    if not ids or not qids or len(ids) != len(set(ids)) or len(qids) != len(set(qids)):
        raise ValueError("nonempty unique record/query identifiers required")
    if any(not isinstance(r["text"], str) or not r["text"].strip() for r in records + queries):
        raise ValueError("nonempty text required")
    id_set = set(ids)
    for q in queries:
        rel = q["relevant"]
        if not rel or (not allow_unretrievable_qrels and not set(rel).issubset(id_set)):
            raise ValueError("every query requires in-corpus qrels")
        if any(not isinstance(g, int) or isinstance(g, bool) or g <= 0 for g in rel.values()):
            raise ValueError("positive integer relevance grades required")
        if "candidates" in q:
            candidates = q["candidates"]
            if (not candidates and not allow_unretrievable_qrels) \
                    or len(candidates) != len(set(candidates)) \
                    or not set(candidates).issubset(id_set) \
                    or (not allow_unretrievable_qrels and not set(rel).issubset(candidates)):
                raise ValueError("invalid scene candidate evidence")
    return data


def ranking_metrics(ranked_ids, qrels):
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranking identifiers must be unique")
    hits = [i + 1 for i, ref in enumerate(ranked_ids) if ref in qrels]
    return {
        **{f"recall_at_{k}": recall_at_10(sum(h <= k for h in hits), len(qrels))
           for k in (1, 5, 10)},
        "success_at_10": float(any(h <= 10 for h in hits)),
        "mrr_at_10": 1 / hits[0] if hits and hits[0] <= 10 else 0.0,
        "ndcg_at_10": ndcg_at_10([qrels.get(ref, 0) for ref in ranked_ids[:10]],
                                 list(qrels.values())),
    }


def paired_comparison(left, right, *, samples=2000, seed=1729):
    if left["dataset_sha256"] != right["dataset_sha256"]:
        raise ValueError("paired runs require identical dataset fingerprints")
    a = {q["query_id"]: q for q in left["per_query"]}
    b = {q["query_id"]: q for q in right["per_query"]}
    if not a or a.keys() != b.keys():
        raise ValueError("paired runs require identical query IDs")
    if left["errors"] or right["errors"]:
        raise ValueError("failed runs cannot establish a complete paired comparison")
    out = {}
    for metric in ("ndcg_at_10", "recall_at_10", "mrr_at_10"):
        deltas = [a[q]["metrics"][metric] - b[q]["metrics"][metric] for q in sorted(a)]
        if not all(math.isfinite(d) for d in deltas):
            raise ValueError("nonfinite paired metrics")
        rng = random.Random(seed)
        bootstrap = sorted(statistics.mean(rng.choices(deltas, k=len(deltas)))
                           for _ in range(samples))
        out[metric] = {
            "mean_delta_left_minus_right": statistics.mean(deltas),
            "wins": sum(d > 1e-12 for d in deltas),
            "ties": sum(abs(d) <= 1e-12 for d in deltas),
            "losses": sum(d < -1e-12 for d in deltas),
            "bootstrap_95ci": [bootstrap[int(samples * .025)], bootstrap[int(samples * .975) - 1]],
            "bootstrap_samples": samples, "seed": seed,
        }
    return out


def prepare(args):
    """Normalize the complete pinned parquet task and original synthetic PoC.
    Downloading is a separate explicit step; no network client is used here.
    """
    import pyarrow.parquet as pq
    files = json.loads(Path(args.lmeb_files).read_text())
    def rows(part):
        name = next(k for k in files if f"-{part}/" in k)
        return pq.read_table(files[name]).to_pylist()
    records = [{"id": r["id"], "text": (r["title"] + "\n" if r["title"] else "") + r["text"]}
               for r in rows("corpus")]
    qrels = {}
    for r in rows("qrels"):
        if r["score"] > 0:
            qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = r["score"]
    queries = [{"id": r["id"], "text": r["text"], "relevant": qrels.get(r["id"], {})}
               for r in rows("queries")]
    japanese = json.loads(Path(args.japanese).read_text())
    datasets = {
        "lmeb_scifact": {
            "source": {"id": "mteb/LMEB_SciFact", "revision": "432cf48c879c2de792720d5c0af6900312ebe7b6", "split": "test", "scope": "full published task; not full LMEB aggregate"},
            "records": sorted(records, key=lambda r: r["id"]),
            "queries": sorted(queries, key=lambda q: q["id"]),
        },
        "japanese_smoke": {
            "source": {"id": "CPersona/poc/embedding/sample.json", "revision": "7821840f114866673022edfc5fe4b21a9b08a5e5", "source_sha256": hashlib.sha256(Path(args.japanese).read_bytes()).hexdigest(), "scope": "synthetic smoke only"},
            "records": japanese["records"],
            "queries": [{"id": f"ja-{i:02}", "text": q["text"], "relevant": {r: 1 for r in q["relevant"]}}
                        for i, q in enumerate(japanese["queries"], 1)],
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, data in datasets.items():
        validate_dataset(data)
        (out / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
        manifest[name] = {"source": data["source"], "records": len(data["records"]),
                          "queries": len(data["queries"]), "dataset_sha256": fingerprint(data)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def run(args):
    import os
    import subprocess
    import sys
    from datetime import datetime, timezone
    # Set before model/tokenizer imports. Offline acquisition is separate.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    import numpy as np
    import psutil
    import torch
    from sentence_transformers import SentenceTransformer
    from importlib.metadata import version

    if not torch.backends.mps.is_available():
        raise RuntimeError("preregistered MPS device is unavailable")
    torch.manual_seed(1729)
    stop = threading.Event()
    peaks = {"sample_interval_s": .1, "peak_rss_bytes": 0, "peak_mps_driver_bytes": 0}
    process = psutil.Process()
    def sample():
        while not stop.is_set():
            peaks["peak_rss_bytes"] = max(peaks["peak_rss_bytes"], process.memory_info().rss)
            peaks["peak_mps_driver_bytes"] = max(peaks["peak_mps_driver_bytes"], torch.mps.driver_allocated_memory())
            stop.wait(.1)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        kwargs = {"device": "mps", "local_files_only": True,
                  "trust_remote_code": args.model == "jina",
                  "model_kwargs": {"torch_dtype": torch.bfloat16}}
        if args.model == "gemma":
            kwargs["config_kwargs"] = {"audio_config": None, "vision_config": None}
        started = time.perf_counter()
        if Path(args.model_root).resolve().name != MODELS[args.model][1]:
            raise ValueError("model root must be the preregistered immutable snapshot")
        model = SentenceTransformer(args.model_root, **kwargs)
        model.to(dtype=torch.bfloat16)
        model.max_seq_length = 512
        model.eval()
        torch.mps.synchronize()
        load_s = time.perf_counter() - started
        common = {"normalize_embeddings": True, "convert_to_numpy": True,
                  "show_progress_bar": False}
        def role(kind):
            if args.model == "gemma":
                return {"prompt_name": "Document" if kind == "document" else "SearchQuery"}
            if args.model == "jina":
                return {"task": "retrieval", "prompt_name": kind}
            return {}
        def encode(texts, kind, batch):
            return np.asarray(model.encode(texts, batch_size=batch, **role(kind), **common), dtype=np.float32)
        def truncated(texts, kind):
            prompt = model.prompts.get(role(kind).get("prompt_name"), "") or ""
            lengths = [len(model.tokenizer.encode(prompt + t, truncation=False)) for t in texts]
            return {"over_512": sum(n > 512 for n in lengths), "max_tokens_with_prompt": max(lengths)}
        output = {
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2], capture_output=True,
                text=True, check=True).stdout.strip(),
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "platform": sys.platform,
            "model": MODELS[args.model][0], "revision": MODELS[args.model][1],
            "condition": {"device": "mps", "dtype": "bf16", "max_seq_length": 512,
                          "corpus_batch": 8, "query_batch": 1, "warmup": 2,
                          "normalized": True, "ranking": "cosine; stable corpus-order ties",
                          "roles": {r: role(r) for r in ("document", "query")},
                          "torch_threads": torch.get_num_threads()},
            "packages": {p: version(p) for p in ("torch", "sentence-transformers", "transformers", "peft", "numpy", "psutil")},
            "load_s": load_s, "datasets": {},
            "limitations": ["one LMEB task and Japanese synthetic smoke, not full LMEB or a representative Japanese benchmark", "standalone dense retrieval only; no CPersona hybrid/gates/reconstruct or generated answer accuracy", "no network-capture audit; local-files-only inference with telemetry/offline flags", "sampled resource maxima may miss brief peaks; no cold/warm paired baseline"],
        }
        for path in args.dataset:
            data = validate_dataset(json.loads(Path(path).read_text()))
            name = Path(path).stem
            texts = [r["text"] for r in data["records"]]
            ids = [r["id"] for r in data["records"]]
            for _ in range(2):
                encode(texts[:1], "document", 1)
                encode([data["queries"][0]["text"]], "query", 1)
            torch.mps.synchronize()
            started = time.perf_counter()
            docs = encode(texts, "document", 8)
            torch.mps.synchronize()
            corpus_s = time.perf_counter() - started
            if not np.isfinite(docs).all() or not np.allclose(np.linalg.norm(docs, axis=1), 1, atol=.01):
                raise ValueError("invalid document vectors")
            rows = []
            for q in data["queries"]:
                torch.mps.synchronize()
                started = time.perf_counter()
                vector = encode([q["text"]], "query", 1)[0]
                torch.mps.synchronize()
                embedding_ms = (time.perf_counter() - started) * 1000
                if not np.isfinite(vector).all():
                    raise ValueError("nonfinite query vector")
                rank_start = time.perf_counter()
                order = np.argsort(-(docs @ vector), kind="stable")
                top = [ids[i] for i in order[:10]]
                ranking_ms = (time.perf_counter() - rank_start) * 1000
                rows.append({"query_id": q["id"], "top10": top,
                             "metrics": ranking_metrics(top, q["relevant"]),
                             "embedding_ms": embedding_ms, "ranking_ms": ranking_ms,
                             "latency_ms": embedding_ms + ranking_ms})
            p50, p95 = percentiles([r["latency_ms"] for r in rows])
            output["datasets"][name] = {
                "source": data["source"], "dataset_sha256": fingerprint(data),
                "corpus_count": len(ids), "query_count": len(rows), "missing_qrels": 0,
                "errors": 0, "dimension": int(docs.shape[1]), "corpus_encode_s": corpus_s,
                "truncation": {"document": truncated(texts, "document"),
                               "query": truncated([q["text"] for q in data["queries"]], "query")},
                "metrics": {k: statistics.mean(r["metrics"][k] for r in rows) for k in rows[0]["metrics"]},
                "p50_ms": p50, "p95_ms": p95, "per_query": rows,
            }
            print(json.dumps({"model": args.model, "dataset": name, "queries": len(rows),
                              "corpus_encode_s": corpus_s, "p50_ms": p50}), flush=True)
        output["resources"] = peaks
        Path(args.out).write_text(json.dumps(output, indent=2))
    finally:
        stop.set()
        monitor.join(timeout=2)


def compare(args):
    results = {p.stem: json.loads(p.read_text()) for p in map(Path, args.result)}
    names = sorted(results)
    if len(names) != 3 or {r["model"] for r in results.values()} != {m[0] for m in MODELS.values()}:
        raise ValueError("three distinct preregistered model results required")
    pins = dict(MODELS.values())
    if any(r["revision"] != pins[r["model"]] for r in results.values()):
        raise ValueError("model revision differs from preregistration")
    baseline = results[names[0]]
    if any(r["packages"] != baseline["packages"]
           or r["harness_sha256"] != baseline["harness_sha256"] for r in results.values()):
        raise ValueError("paired runs require the same harness and dependency versions")
    output = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if results[left]["condition"] != results[right]["condition"]:
                # Roles are intentionally model-specific; other controls match.
                a = {k: v for k, v in results[left]["condition"].items() if k != "roles"}
                b = {k: v for k, v in results[right]["condition"].items() if k != "roles"}
                if a != b:
                    raise ValueError("paired runs have different execution conditions")
            output[f"{left}_minus_{right}"] = {
                dataset: paired_comparison(results[left]["datasets"][dataset], results[right]["datasets"][dataset])
                for dataset in results[left]["datasets"]}
    Path(args.out).write_text(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--lmeb-files", required=True)
    p.add_argument("--japanese", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(action=prepare)
    p = sub.add_parser("run")
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--model-root", required=True)
    p.add_argument("--dataset", action="append", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(action=run)
    p = sub.add_parser("compare")
    p.add_argument("--result", action="append", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(action=compare)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
