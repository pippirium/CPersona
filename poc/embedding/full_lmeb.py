"""Normalize all pinned LMEB retrieval tasks, preserving candidate scenes.

Public input text stays in an operator-selected scratch directory. The
manifest contains counts, relative task identifiers and fingerprints only.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import threading
import time
from poc.embedding.compare import fingerprint, validate_dataset

REVISION = "f9d4294be4a24b8c16d6bfb59d56a6fec4bd95c0"
INSTRUCTIONS_SHA256 = "f3f83069f3a3818a89c28de320051521bc37c6b8314d110b8911356eb9adfaa8"
ENCODE_CHUNK_SIZE = 1024
INSTRUCTION_DATASET_KEYS = {
    "Covid-QA": ("CovidQA", "Covid-QA"),
    "ESG-Reports": ("ESGReports", "ESG-Reports"),
    "SciFact": ("LMEB_SciFact", "LMEB_SciFact"),
}


def read_rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def canonicalize_records(rows):
    """Drop only byte-semantically identical records with a repeated ID."""
    records = {}
    raw_count = duplicate_count = 0
    for row in rows:
        raw_count += 1
        record = {"id": str(row["id"]),
                  "text": (row.get("title", "") + "\n" if row.get("title") else "") + row["text"]}
        previous = records.get(record["id"])
        if previous is not None:
            if previous != record:
                raise ValueError("conflicting source rows share a corpus identifier")
            duplicate_count += 1
        else:
            records[record["id"]] = record
    return sorted(records.values(), key=lambda r: r["id"]), raw_count, duplicate_count


def candidate_scene(query_id, scenes):
    matches = [v for k, v in scenes.items()
               if query_id == str(k) or query_id.startswith(str(k) + "_")]
    if len(matches) != 1:
        raise ValueError("query must identify exactly one published candidate scene")
    return matches[0]


def instruction_for_task(task, instructions):
    parts = task.split("/")
    if len(parts) < 2:
        raise ValueError("LMEB task path must include its dataset")
    dataset = parts[1]
    dataset_key, instruction_key = INSTRUCTION_DATASET_KEYS.get(
        dataset, (dataset, "MemGovern" if dataset == "MemGovern" else parts[-1]))
    try:
        return instructions[dataset_key][instruction_key]
    except KeyError as exc:
        raise ValueError("pinned task instruction is missing for an LMEB task") from exc


def query_input(query, model_key, instructions):
    if model_key != "jina":
        return query["text"]
    instruction = instruction_for_task(query["task"], instructions)
    return f"Instruct: {instruction}\nQuery: {query['text']}"


def encode_in_chunks(texts, chunk_size, encode_chunk, synchronize, after_chunk=None):
    """Encode bounded outer chunks and drain the device queue between them."""
    import numpy as np

    if not texts:
        raise ValueError("cannot encode an empty input sequence")
    chunks = []
    total = len(texts)
    for start in range(0, total, chunk_size):
        part = texts[start:start + chunk_size]
        values = encode_chunk(part)
        synchronize()
        chunks.append(np.asarray(values, dtype=np.float32))
        del values
        if after_chunk is not None:
            after_chunk(min(start + len(part), total), total)
    return np.concatenate(chunks, axis=0)


def prepare(root, out, task_instructions_path):
    root, out = Path(root), Path(out)
    if root.name != REVISION:
        raise ValueError("full LMEB input must be the pinned snapshot")
    instruction_path = Path(task_instructions_path)
    instruction_hash = hashlib.sha256(instruction_path.read_bytes()).hexdigest()
    if instruction_hash != INSTRUCTIONS_SHA256:
        raise ValueError("task instructions must match the pinned LMEB revision")
    instructions = json.loads(instruction_path.read_text())
    query_files = sorted(root.rglob("queries.jsonl"))
    qrels_files = sorted(root.rglob("qrels.tsv"))
    corpus_files = sorted(root.rglob("corpus.jsonl"))
    if (len(query_files), len(qrels_files), len(corpus_files)) != (193, 193, 103):
        raise ValueError("incomplete pinned LMEB snapshot")
    grouped = defaultdict(list)
    for query_file in query_files:
        parent = query_file.parent
        while parent != root and not (parent / "corpus.jsonl").is_file():
            parent = parent.parent
        if parent == root:
            raise ValueError("query task has no ancestor corpus")
        grouped[parent / "corpus.jsonl"].append(query_file)
    manifest = {"source": "KaLM-Embedding/LMEB", "revision": REVISION,
                "task_instruction_sha256": instruction_hash,
                "groups": [], "task_count": 0, "source_query_count": 0,
                "scored_query_count": 0, "excluded_no_positive_qrels": 0,
                "task_instruction_count": 0,
                "corpus_row_count": 0, "corpus_count_after_deduplication": 0,
                "duplicate_identical_corpus_rows": 0, "qrel_row_count": 0,
                "positive_qrel_row_count": 0, "duplicate_positive_qrel_pairs": 0,
                "missing_qrel_id_count": 0, "missing_qrel_links": 0,
                "missing_candidate_id_count": 0, "out_of_scene_qrel_links": 0,
                "duplicate_candidate_links": 0, "orphan_qrel_query_count": 0,
                "candidate_query_count": 0, "empty_candidate_query_count": 0}
    out.mkdir(parents=True, exist_ok=True)
    for corpus_file, tasks in sorted(grouped.items()):
        label = str(corpus_file.parent.relative_to(root))
        records, corpus_row_count, duplicate_corpus_rows = canonicalize_records(read_rows(corpus_file))
        candidate_file = corpus_file.parent / "candidates.jsonl"
        scenes = {}
        duplicate_candidate_links = 0
        if candidate_file.is_file():
            for row in read_rows(candidate_file):
                scene_id = str(row["scene_id"])
                refs = [str(ref) for ref in row["candidate_doc_ids"]]
                if scene_id in scenes and scenes[scene_id] != refs:
                    raise ValueError("conflicting source rows share a candidate scene identifier")
                if scene_id in scenes:
                    duplicate_candidate_links += len(refs)
                else:
                    scenes[scene_id] = refs
        id_set = {r["id"] for r in records}
        missing_candidates = set()
        for scene, refs in scenes.items():
            missing_candidates.update(ref for ref in refs if ref not in id_set)
            duplicate_candidate_links += len(refs) - len(set(refs))
            scenes[scene] = sorted(set(refs) & id_set)
        queries, task_meta = [], []
        missing_qrels = set()
        out_of_scene_links = 0
        missing_qrel_links = 0
        qrel_row_count = 0
        positive_qrel_row_count = 0
        duplicate_qrel_pairs = 0
        orphan_qrel_query_ids = set()
        for query_file in tasks:
            task = str(query_file.parent.relative_to(root))
            instruction_for_task(task, instructions)
            manifest["task_instruction_count"] += 1
            qrels = defaultdict(dict)
            with query_file.with_name("qrels.tsv").open() as stream:
                for row in csv.reader(stream, delimiter="\t"):
                    if not row:
                        continue
                    if row == ["query-id", "corpus-id", "score"]:
                        continue
                    if len(row) != 3:
                        raise ValueError("invalid qrel columns")
                    qrel_row_count += 1
                    grade = int(row[2])
                    if grade > 0:
                        positive_qrel_row_count += 1
                        query_id, corpus_id = str(row[0]), str(row[1])
                        previous = qrels[query_id].get(corpus_id)
                        if previous is not None:
                            if previous != grade:
                                raise ValueError("conflicting source qrels share a query-document pair")
                            duplicate_qrel_pairs += 1
                        qrels[query_id][corpus_id] = grade
            source_rows = list(read_rows(query_file))
            source_query_ids = {str(row["id"]) for row in source_rows}
            orphan_qrel_query_ids.update(set(qrels) - source_query_ids)
            excluded = 0
            for row in source_rows:
                qid = str(row["id"])
                if not qrels.get(qid):
                    excluded += 1
                    continue  # retrieval metrics undefined; counted explicitly
                q = {"id": task + "#" + qid, "source_query_id": qid,
                     "task": task, "text": row["text"], "relevant": qrels[qid]}
                missing_qrels.update(set(qrels[qid]) - id_set)
                missing_qrel_links += len(set(qrels[qid]) - id_set)
                if scenes:
                    q["candidates"] = candidate_scene(qid, scenes)
                    out_of_scene_links += len(set(qrels[qid]) - set(q["candidates"]))
                queries.append(q)
            task_meta.append({"task": task, "source_queries": len(source_rows),
                              "scored_queries": len(source_rows) - excluded,
                              "excluded_no_positive_qrels": excluded})
            manifest["source_query_count"] += len(source_rows)
            manifest["scored_query_count"] += len(source_rows) - excluded
            manifest["excluded_no_positive_qrels"] += excluded
            manifest["task_count"] += 1
        queries.sort(key=lambda q: q["id"])
        manifest["corpus_row_count"] += corpus_row_count
        manifest["corpus_count_after_deduplication"] += len(records)
        manifest["duplicate_identical_corpus_rows"] += duplicate_corpus_rows
        manifest["qrel_row_count"] += qrel_row_count
        manifest["positive_qrel_row_count"] += positive_qrel_row_count
        manifest["duplicate_positive_qrel_pairs"] += duplicate_qrel_pairs
        manifest["missing_qrel_id_count"] += len(missing_qrels)
        manifest["missing_qrel_links"] += missing_qrel_links
        manifest["missing_candidate_id_count"] += len(missing_candidates)
        manifest["out_of_scene_qrel_links"] += out_of_scene_links
        manifest["duplicate_candidate_links"] += duplicate_candidate_links
        manifest["orphan_qrel_query_count"] += len(orphan_qrel_query_ids)
        manifest["candidate_query_count"] += sum("candidates" in q for q in queries)
        manifest["empty_candidate_query_count"] += sum(
            not q.get("candidates") for q in queries if "candidates" in q)
        data = {"source": {"id": "KaLM-Embedding/LMEB", "revision": REVISION,
                           "corpus": label, "tasks": task_meta,
                           "corpus_sha256": hashlib.sha256(corpus_file.read_bytes()).hexdigest(),
                           "corpus_row_count": corpus_row_count,
                           "duplicate_identical_corpus_rows": duplicate_corpus_rows,
                           "qrel_row_count": qrel_row_count,
                           "positive_qrel_row_count": positive_qrel_row_count,
                           "duplicate_positive_qrel_pairs": duplicate_qrel_pairs,
                           "orphan_qrel_query_ids": sorted(orphan_qrel_query_ids),
                           "duplicate_candidate_links": duplicate_candidate_links,
                           "qrel_policy": "retain_unretrievable_denominator",
                           "missing_qrel_ids": sorted(missing_qrels),
                           "missing_qrel_links": missing_qrel_links,
                           "missing_candidate_ids": sorted(missing_candidates),
                           "out_of_scene_qrel_links": out_of_scene_links},
                "records": records, "queries": queries}
        validate_dataset(data, allow_unretrievable_qrels=True)
        digest = fingerprint(data)
        filename = hashlib.sha256(label.encode()).hexdigest()[:16] + ".json"
        (out / filename).write_text(json.dumps(data, ensure_ascii=False))
        manifest["groups"].append({"input": filename, "corpus": label,
                                    "dataset_sha256": digest, "corpus_count": len(records),
                                    "source_corpus_rows": corpus_row_count,
                                    "duplicate_identical_corpus_rows": duplicate_corpus_rows,
                                    "query_count": len(queries), "tasks": task_meta,
                                    "missing_qrel_count": len(missing_qrels),
                                    "missing_qrel_links": missing_qrel_links,
                                    "missing_candidate_count": len(missing_candidates),
                                    "out_of_scene_qrel_links": out_of_scene_links,
                                    "duplicate_positive_qrel_pairs": duplicate_qrel_pairs,
                                    "duplicate_candidate_links": duplicate_candidate_links,
                                    "orphan_qrel_query_count": len(orphan_qrel_query_ids)})
        print(json.dumps({"prepared": label, "records": len(records), "queries": len(queries)}), flush=True)
    if len(manifest["groups"]) != 103 or manifest["task_count"] != 193:
        raise ValueError("full task coverage mismatch")
    if len(instructions) != 22 or manifest["task_instruction_count"] != 193:
        raise ValueError("pinned LMEB task instruction coverage mismatch")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _harness_sha256():
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("compare.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_gzip_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(payload)
    os.replace(temporary, path)


def _read_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def stable_top_ids(scores, corpus_ids, candidate_positions=None, limit=10):
    """Return exact top-k, breaking score ties by sorted corpus position."""
    import numpy as np
    scores = np.asarray(scores)
    positions = (np.arange(len(corpus_ids), dtype=np.int64)
                 if candidate_positions is None else np.asarray(candidate_positions, dtype=np.int64))
    if not len(positions):
        return []
    values = scores[positions]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite similarity scores")
    if len(positions) > limit:
        threshold = np.partition(values, len(values) - limit)[len(values) - limit]
        greater = np.flatnonzero(values > threshold)
        equal = np.flatnonzero(values == threshold)
        selected = np.concatenate((greater, equal[:limit - len(greater)]))
    else:
        selected = np.arange(len(positions))
    order = np.lexsort((positions[selected], -values[selected]))
    return [corpus_ids[int(positions[i])] for i in selected[order][:limit]]


def _condition(instruction_sha256):
    return {
        "device": "mps", "dtype": "bf16", "max_seq_length": 512,
        "corpus_batch": 8, "quality_query_batch": 8,
        "encode_outer_chunk_size": ENCODE_CHUNK_SIZE,
        "mps_synchronize_after_each_encode_chunk": True,
        "mps_empty_cache_after_large_encode_chunks": True,
        "latency_probe_query_batch": 1, "latency_probe_per_task": 20,
        "latency_probe_seed": 1729, "warmup_calls_per_role_per_group": 2,
        "normalized": True, "ranking": "cosine; stable corpus-order ties",
        "candidate_policy": "published candidate scenes when present; full corpus otherwise",
        "positive_qrel_policy": "retain every positive source qrel in recall and NDCG denominators",
        "task_instruction_sha256": instruction_sha256,
        "task_instruction_policy": {
            "jina": "LMEB task instruction prepended as Instruct: ... newline Query: ...",
            "bge": "no LMEB task instruction, matching the published model setting",
            "gemma": "no LMEB task instruction, matching the published model setting",
        },
        "roles": {}, "torch_threads": None,
    }


def _source_summary(source):
    return {
        "id": source["id"], "revision": source["revision"],
        "corpus": source["corpus"], "tasks": source["tasks"],
        "corpus_sha256": source["corpus_sha256"],
        "corpus_row_count": source["corpus_row_count"],
        "duplicate_identical_corpus_rows": source["duplicate_identical_corpus_rows"],
        "qrel_row_count": source["qrel_row_count"],
        "positive_qrel_row_count": source["positive_qrel_row_count"],
        "duplicate_positive_qrel_pairs": source["duplicate_positive_qrel_pairs"],
        "orphan_qrel_query_count": len(source["orphan_qrel_query_ids"]),
        "duplicate_candidate_links": source["duplicate_candidate_links"],
        "qrel_policy": source["qrel_policy"],
        "missing_qrel_id_count": len(source["missing_qrel_ids"]),
        "missing_qrel_link_count": source["missing_qrel_links"],
        "missing_candidate_id_count": len(source["missing_candidate_ids"]),
        "out_of_scene_qrel_links": source["out_of_scene_qrel_links"],
    }


def _nearest_percentile(values, percentile):
    ordered = sorted(values)
    return ordered[max(1, math.ceil(percentile * len(ordered))) - 1]


def run(args):
    """Run all normalized LMEB corpus groups for one model, with resumable checkpoints."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    import numpy as np
    import psutil
    import torch
    from sentence_transformers import SentenceTransformer
    from importlib.metadata import version
    from poc.embedding.compare import MODELS, ranking_metrics, validate_dataset

    if args.model not in MODELS:
        raise ValueError("model is not preregistered")
    if not torch.backends.mps.is_available():
        raise RuntimeError("preregistered MPS device is unavailable")
    model_root = Path(args.model_root)
    if model_root.resolve().name != MODELS[args.model][1]:
        raise ValueError("model root must be the preregistered immutable snapshot")
    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite a completed model result")

    manifest = json.loads((Path(args.input_dir) / "manifest.json").read_text())
    if (manifest.get("source"), manifest.get("revision"), len(manifest.get("groups", [])),
            manifest.get("task_count"), manifest.get("source_query_count")) != (
            "KaLM-Embedding/LMEB", REVISION, 103, 193, 159011):
        raise ValueError("normalized full LMEB manifest is incomplete or differs from the pin")
    instruction_path = Path(args.task_instructions)
    instruction_sha256 = hashlib.sha256(instruction_path.read_bytes()).hexdigest()
    if instruction_sha256 != INSTRUCTIONS_SHA256 \
            or manifest.get("task_instruction_sha256") != instruction_sha256:
        raise ValueError("task instructions differ from the pinned preparation input")
    instructions = json.loads(instruction_path.read_text())
    instruction_tasks = [task["task"] for group in manifest["groups"] for task in group["tasks"]]
    if len(instruction_tasks) != 193 or len(set(instruction_tasks)) != 193:
        raise ValueError("normalized manifest has incomplete task instruction coverage")
    for task in instruction_tasks:
        instruction_for_task(task, instructions)

    torch.manual_seed(1729)
    stop = threading.Event()
    peaks = {"sample_interval_s": .1, "peak_rss_bytes": 0, "peak_mps_driver_bytes": 0}
    process = psutil.Process()

    def sample_resources():
        while not stop.is_set():
            peaks["peak_rss_bytes"] = max(peaks["peak_rss_bytes"], process.memory_info().rss)
            peaks["peak_mps_driver_bytes"] = max(
                peaks["peak_mps_driver_bytes"], torch.mps.driver_allocated_memory())
            stop.wait(.1)

    monitor = threading.Thread(target=sample_resources, daemon=True)
    monitor.start()
    try:
        kwargs = {"device": "mps", "local_files_only": True,
                  "trust_remote_code": args.model == "jina",
                  "model_kwargs": {"torch_dtype": torch.bfloat16}}
        if args.model == "gemma":
            kwargs["config_kwargs"] = {"audio_config": None, "vision_config": None}
        load_started = time.perf_counter()
        model = SentenceTransformer(str(model_root), **kwargs)
        model.to(dtype=torch.bfloat16)
        model.max_seq_length = 512
        model.eval()
        torch.mps.synchronize()
        load_s = time.perf_counter() - load_started

        def role(kind):
            if args.model == "gemma":
                return {"prompt_name": "Document" if kind == "document" else "SearchQuery"}
            if args.model == "jina":
                return {"task": "retrieval", "prompt_name": kind}
            return {}

        condition = _condition(instruction_sha256)
        condition["roles"] = {kind: role(kind) for kind in ("document", "query")}
        condition["torch_threads"] = torch.get_num_threads()
        common = {"normalize_embeddings": True, "convert_to_numpy": True,
                  "show_progress_bar": False}
        encode_context = {"group": 0, "corpus": "", "stage": "warmup"}

        def encode(texts, kind, batch):
            def encode_chunk(part):
                return model.encode(part, batch_size=batch, **role(kind), **common)

            def after_chunk(completed, total):
                if total > ENCODE_CHUNK_SIZE:
                    print(json.dumps({"model": args.model, **encode_context,
                                      "progress": completed, "total": total}), flush=True)
                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()

            return encode_in_chunks(texts, ENCODE_CHUNK_SIZE, encode_chunk,
                                    torch.mps.synchronize, after_chunk)

        def truncation(texts, kind):
            prompt_name = role(kind).get("prompt_name")
            prompt = model.prompts.get(prompt_name, "") if prompt_name else ""
            over_limit, maximum = 0, 0
            for start in range(0, len(texts), 32):
                batch = [prompt + text for text in texts[start:start + 32]]
                encoded = model.tokenizer(batch, add_special_tokens=True, truncation=False,
                                          padding=False)
                lengths = [len(row) for row in encoded["input_ids"]]
                over_limit += sum(length > 512 for length in lengths)
                maximum = max(maximum, max(lengths, default=0))
            return {"over_512": over_limit, "max_tokens_with_prompt": maximum}

        packages = {p: version(p) for p in (
            "torch", "sentence-transformers", "transformers", "peft", "numpy", "psutil")}
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=True).stdout.strip()
        harness_sha = _harness_sha256()
        checkpoint_dir = Path(args.checkpoint_dir) / args.model
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        datasets = {}
        expected_queries = 0
        expected_tasks = 0
        recorded_load_s = load_s
        for group_number, group in enumerate(manifest["groups"], 1):
            encode_context.update(group=group_number, corpus=group["corpus"])
            data_path = Path(args.input_dir) / group["input"]
            data = json.loads(data_path.read_text())
            validate_dataset(data, allow_unretrievable_qrels=True)
            if fingerprint(data) != group["dataset_sha256"]:
                raise ValueError("normalized dataset fingerprint differs from manifest")
            expected_queries += len(data["queries"])
            expected_tasks += len(data["source"]["tasks"])
            checkpoint = checkpoint_dir / (Path(group["input"]).stem + ".json.gz")
            if checkpoint.exists():
                saved = _read_json(checkpoint)
                if (saved.get("model_key"), saved.get("model"), saved.get("revision"),
                        saved.get("harness_sha256"), saved.get("source_commit"),
                        saved.get("condition"), saved.get("packages"),
                        saved.get("dataset", {}).get("dataset_sha256"),
                        isinstance(saved.get("resources"), dict)) != (
                        args.model, MODELS[args.model][0], MODELS[args.model][1],
                        harness_sha, source_commit, condition, packages, group["dataset_sha256"], True):
                    raise ValueError("checkpoint provenance does not match current model run")
                for resource in ("peak_rss_bytes", "peak_mps_driver_bytes"):
                    peaks[resource] = max(peaks[resource], saved["resources"].get(resource, 0))
                if group_number == 1:
                    recorded_load_s = saved["load_s"]
                saved_queries = saved.get("dataset", {}).get("per_query", [])
                if [q.get("query_id") for q in saved_queries] != [q["id"] for q in data["queries"]]:
                    raise ValueError("checkpoint query identifiers differ from normalized input")
                datasets[group["corpus"]] = saved["dataset"]
                print(json.dumps({"model": args.model, "group": group_number,
                                  "corpus": group["corpus"], "status": "resumed",
                                  "queries": len(saved_queries)}), flush=True)
                del saved_queries, saved, data
                continue

            texts = [record["text"] for record in data["records"]]
            corpus_ids = [record["id"] for record in data["records"]]
            id_to_position = {identifier: i for i, identifier in enumerate(corpus_ids)}
            query_texts = [query_input(query, args.model, instructions)
                           for query in data["queries"]]
            for _ in range(2):
                encode(texts[:1], "document", 1)
                encode(query_texts[:1], "query", 1)
            torch.mps.synchronize()
            started = time.perf_counter()
            encode_context["stage"] = "corpus_encode"
            docs = encode(texts, "document", 8)
            corpus_encode_s = time.perf_counter() - started
            if (not np.isfinite(docs).all()
                    or not np.allclose(np.linalg.norm(docs, axis=1), 1, atol=.01)):
                raise ValueError("invalid document vectors")
            started = time.perf_counter()
            encode_context["stage"] = "quality_query_encode"
            query_vectors = encode(query_texts, "query", 8)
            query_encode_s = time.perf_counter() - started
            if not np.isfinite(query_vectors).all():
                raise ValueError("nonfinite query vectors")

            per_query = []
            rank_started = time.perf_counter()
            for start in range(0, len(data["queries"]), 8):
                score_batch = docs @ query_vectors[start:start + 8].T
                for offset, query in enumerate(data["queries"][start:start + 8]):
                    candidates = query.get("candidates")
                    positions = (None if candidates is None else
                                 sorted(id_to_position[identifier] for identifier in candidates))
                    top_ids = stable_top_ids(score_batch[:, offset], corpus_ids, positions)
                    relevant = query["relevant"]
                    available_qrels = set(relevant).intersection(corpus_ids)
                    if candidates is not None:
                        available_qrels.intersection_update(candidates)
                    per_query.append({
                        "query_id": query["id"], "task": query["task"], "top10": top_ids,
                        "qrel_count": len(relevant),
                        "unavailable_qrel_count": len(relevant) - len(available_qrels),
                        "metrics": ranking_metrics(top_ids, relevant),
                    })
            ranking_s = time.perf_counter() - rank_started

            task_queries = defaultdict(list)
            for query in data["queries"]:
                task_queries[query["task"]].append(query)
            latency_samples = []
            latency_by_task = {}
            encode_context["stage"] = "latency_probe"
            for task, queries in sorted(task_queries.items()):
                seed_suffix = int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)
                rng = random.Random(1729 ^ seed_suffix)
                selected = rng.sample(queries, min(20, len(queries)))
                task_samples = []
                for query in selected:
                    torch.mps.synchronize()
                    started = time.perf_counter()
                    vector = encode([query_input(query, args.model, instructions)],
                                    "query", 1)[0]
                    embedding_ms = (time.perf_counter() - started) * 1000
                    candidates = query.get("candidates")
                    positions = (None if candidates is None else
                                 sorted(id_to_position[identifier] for identifier in candidates))
                    rank_started = time.perf_counter()
                    stable_top_ids(docs @ vector, corpus_ids, positions)
                    ranking_ms = (time.perf_counter() - rank_started) * 1000
                    item = {"query_id": query["id"], "task": task,
                            "embedding_ms": embedding_ms, "ranking_ms": ranking_ms,
                            "latency_ms": embedding_ms + ranking_ms}
                    task_samples.append(item)
                    latency_samples.append(item)
                times = [item["latency_ms"] for item in task_samples]
                latency_by_task[task] = {
                    "sample_count": len(times), "p50_ms": _nearest_percentile(times, .5),
                    "p95_ms": _nearest_percentile(times, .95),
                }

            metrics = {name: statistics.mean(row["metrics"][name] for row in per_query)
                       for name in per_query[0]["metrics"]}
            unavailable_link_count = sum(row["unavailable_qrel_count"] for row in per_query)
            source = _source_summary(data["source"])
            source["corpus_count_after_identical_id_deduplication"] = len(corpus_ids)
            dataset_result = {
                "source": source, "dataset_sha256": group["dataset_sha256"],
                "corpus_count": len(corpus_ids), "query_count": len(per_query),
                "unavailable_positive_qrel_links": unavailable_link_count,
                "candidate_query_count": sum("candidates" in q for q in data["queries"]),
                "empty_candidate_query_count": sum(not q.get("candidates")
                                                    for q in data["queries"] if "candidates" in q),
                "errors": 0, "dimension": int(docs.shape[1]),
                "corpus_encode_s": corpus_encode_s, "quality_query_encode_s": query_encode_s,
                "quality_query_encode_per_second": len(per_query) / query_encode_s,
                "ranking_s": ranking_s,
                "truncation": {"document": truncation(texts, "document"),
                               "query": truncation(query_texts, "query")},
                "metrics": metrics,
                "latency_probe": {
                    "sample_count": len(latency_samples), "sampling": "up to 20 queries per source task",
                    "p50_ms": _nearest_percentile([q["latency_ms"] for q in latency_samples], .5),
                    "p95_ms": _nearest_percentile([q["latency_ms"] for q in latency_samples], .95),
                    "by_task": latency_by_task, "samples": latency_samples,
                },
                "per_query": per_query,
            }
            saved = {"model_key": args.model, "model": MODELS[args.model][0],
                     "revision": MODELS[args.model][1], "harness_sha256": harness_sha,
                     "source_commit": source_commit, "condition": condition,
                     "packages": packages, "load_s": load_s,
                     "resources": dict(peaks), "dataset": dataset_result}
            _write_gzip_json(checkpoint, saved)
            datasets[group["corpus"]] = dataset_result
            del docs, query_vectors, per_query, data
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            print(json.dumps({"model": args.model, "group": group_number,
                              "corpus": group["corpus"], "status": "complete",
                              "queries": dataset_result["query_count"],
                              "corpus_encode_s": corpus_encode_s,
                              "quality_query_encode_s": query_encode_s}), flush=True)

        if len(datasets) != 103 or expected_tasks != 193 \
                or expected_queries != manifest["scored_query_count"] \
                or sum(d["query_count"] for d in datasets.values()) != manifest["scored_query_count"]:
            raise ValueError("full LMEB inference did not cover every normalized query group")
        output = {
            "model_key": args.model, "model": MODELS[args.model][0],
            "revision": MODELS[args.model][1], "harness_sha256": harness_sha,
            "source_commit": source_commit, "executed_at": datetime.now(timezone.utc).isoformat(),
            "platform": {"system": platform.system(), "release": platform.release(),
                         "machine": platform.machine(),
                         "cpu": (subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                                capture_output=True, text=True).stdout.strip()
                                 or platform.processor()),
                         "ram_bytes": psutil.virtual_memory().total},
            "condition": condition, "packages": packages,
            "load_s": recorded_load_s, "task_count": expected_tasks,
            "source_query_count": manifest["source_query_count"],
            "scored_query_count": sum(d["query_count"] for d in datasets.values()),
            "resources": peaks, "datasets": datasets,
            "limitations": [
                "full LMEB dense retrieval evaluates embeddings and rankings, not CPersona answer quality or hybrid gates",
                "public source qrels absent from corpus or outside published candidate scenes remain in recall and NDCG denominators",
                "exact duplicate QASPER corpus IDs are deduplicated; conflicting content under one ID aborts preparation",
                "query latency is a deterministic sample of up to 20 queries per task at batch size one; quality retrieval uses query batches of eight",
                "sampled RSS and MPS driver maxima may miss brief peaks; timing applies only to this MPS bf16 hardware/software condition",
            ],
        }
        _write_gzip_json(output_path, output)
    finally:
        stop.set()
        monitor.join(timeout=2)


def _bootstrap_mean_ci(matrix, seed, samples=2000):
    import numpy as np
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires finite paired values")
    rng = np.random.default_rng(seed)
    boots = []
    for start in range(0, samples, 64):
        size = min(64, samples - start)
        indices = rng.integers(0, values.shape[0], size=(size, values.shape[0]))
        boots.append(values[indices].mean(axis=1))
    estimates = np.concatenate(boots, axis=0)
    return np.percentile(estimates, [2.5, 97.5], axis=0).T.tolist()


def compare_task_rows(rows, query_ids, model_keys, metrics, samples=2000, seed=1729):
    import numpy as np
    if not query_ids or len(query_ids) != len(set(query_ids)) \
            or any(not set(query_ids).issubset(rows[key]) for key in model_keys):
        raise ValueError("paired task rows require identical nonempty query IDs")
    model_metrics = {
        key: {metric: statistics.mean(rows[key][qid]["metrics"][metric] for qid in query_ids)
              for metric in metrics}
        for key in model_keys
    }
    pairwise = {}
    for i, left in enumerate(sorted(model_keys)):
        for right in sorted(model_keys)[i + 1:]:
            pair_key = f"{left}_minus_{right}"
            deltas = np.asarray([
                [rows[left][qid]["metrics"][metric] - rows[right][qid]["metrics"][metric]
                 for metric in metrics] for qid in query_ids], dtype=np.float64)
            ci = _bootstrap_mean_ci(deltas, seed, samples)
            pairwise[pair_key] = {
                metric: {
                    "mean_delta": float(deltas[:, index].mean()),
                    "wins": int((deltas[:, index] > 1e-12).sum()),
                    "ties": int((np.abs(deltas[:, index]) <= 1e-12).sum()),
                    "losses": int((deltas[:, index] < -1e-12).sum()),
                    "query_bootstrap_95ci": ci[index],
                } for index, metric in enumerate(metrics)
            }
    return model_metrics, pairwise


def compare_full_results(results, samples=2000, seed=1729):
    """Compare three complete runs using query, task, then equal-dataset units."""
    import numpy as np
    from poc.embedding.compare import MODELS

    if set(results) != set(MODELS):
        raise ValueError("three preregistered model results are required")
    reference = results["jina"]
    for key, result in results.items():
        if (result.get("model_key"), result.get("model"), result.get("revision")) != (
                key, MODELS[key][0], MODELS[key][1]):
            raise ValueError("model identity differs from preregistration")
        if (result.get("harness_sha256"), result.get("packages"), result.get("condition"),
                result.get("source_commit"), result.get("platform")) != (
                reference.get("harness_sha256"), reference.get("packages"),
                reference.get("condition"), reference.get("source_commit"),
                reference.get("platform")):
            baseline = reference.get("condition", {})
            current = result.get("condition", {})
            if ({k: v for k, v in current.items() if k != "roles"}
                    != {k: v for k, v in baseline.items() if k != "roles"}
                    or result.get("harness_sha256") != reference.get("harness_sha256")
                    or result.get("packages") != reference.get("packages")
                    or result.get("source_commit") != reference.get("source_commit")
                    or result.get("platform") != reference.get("platform")):
                raise ValueError("model runs differ in code, packages, or execution controls")
        if result.get("errors"):
            raise ValueError("failed model run cannot be compared")
    corpus_keys = set(reference["datasets"])
    if any(set(result["datasets"]) != corpus_keys for result in results.values()):
        raise ValueError("model runs do not cover the same corpus groups")

    metrics = ("ndcg_at_10", "recall_at_1", "recall_at_5", "recall_at_10",
               "success_at_10", "mrr_at_10")
    tasks = {}
    dataset_tasks = defaultdict(list)
    latency_by_model = {key: [] for key in results}
    latency_ids = {key: set() for key in results}
    for corpus_key in sorted(corpus_keys):
        corpus_results = {key: result["datasets"][corpus_key] for key, result in results.items()}
        if len({item["dataset_sha256"] for item in corpus_results.values()}) != 1:
            raise ValueError("model runs use different normalized inputs")
        rows = {key: {q["query_id"]: q for q in item["per_query"]}
                for key, item in corpus_results.items()}
        if any(set(qrows) != set(rows["jina"]) for qrows in rows.values()):
            raise ValueError("model runs have mismatched query IDs")
        if any(any(q.get("errors") for q in item["per_query"]) for item in corpus_results.values()):
            raise ValueError("per-query errors prevent a complete comparison")
        task_ids = {q["task"] for q in rows["jina"].values()}
        if any({q["task"] for q in qrows.values()} != task_ids for qrows in rows.values()):
            raise ValueError("model runs have mismatched task assignments")
        for key, item in corpus_results.items():
            for sample in item["latency_probe"]["samples"]:
                sample_id = sample["task"] + "#" + sample["query_id"]
                latency_by_model[key].append(sample["latency_ms"])
                latency_ids[key].add(sample_id)
        for task in sorted(task_ids):
            ids = sorted(qid for qid, row in rows["jina"].items() if row["task"] == task)
            model_metrics, pairwise = compare_task_rows(
                rows, ids, sorted(results), metrics, samples=samples, seed=seed)
            task_row = {"query_count": len(ids), "model_metrics": model_metrics,
                        "pairwise": pairwise}
            tasks[task] = task_row
            dataset_id = "/".join(task.split("/")[:2])
            dataset_tasks[dataset_id].append((task, task_row))

    if len(dataset_tasks) != 22 or sum(v["query_count"] for v in tasks.values()) != 159011:
        raise ValueError("paired result is missing the pinned 22 dataset / 159,011 query coverage")
    datasets = {}
    for dataset_id, entries in sorted(dataset_tasks.items()):
        model_metrics = {key: {metric: statistics.mean(row["model_metrics"][key][metric]
                                                       for _, row in entries)
                               for metric in metrics} for key in results}
        pairs = {}
        for pair_key in next(iter(entries))[1]["pairwise"]:
            pairs[pair_key] = {}
            for metric in metrics:
                deltas = [row["pairwise"][pair_key][metric]["mean_delta"] for _, row in entries]
                ci = _bootstrap_mean_ci(np.asarray(deltas), seed, samples)[0]
                pairs[pair_key][metric] = {
                    "mean_task_delta": statistics.mean(deltas),
                    "task_wins": sum(delta > 1e-12 for delta in deltas),
                    "task_ties": sum(abs(delta) <= 1e-12 for delta in deltas),
                    "task_losses": sum(delta < -1e-12 for delta in deltas),
                    "task_bootstrap_95ci": ci,
                }
        datasets[dataset_id] = {"task_count": len(entries), "model_metrics": model_metrics,
                                "pairs": pairs,
                                "tasks": [task for task, _ in entries]}

    overall_model_metrics = {key: {metric: statistics.mean(d["model_metrics"][key][metric]
                                                           for d in datasets.values())
                                   for metric in metrics} for key in results}
    overall_pairs = {}
    for pair_key in next(iter(datasets.values()))["pairs"]:
        overall_pairs[pair_key] = {}
        for metric in metrics:
            deltas = [dataset["pairs"][pair_key][metric]["mean_task_delta"]
                      for dataset in datasets.values()]
            dataset_deltas = np.asarray(deltas, dtype=np.float64)
            ci = _bootstrap_mean_ci(dataset_deltas, seed, samples)[0]
            overall_pairs[pair_key][metric] = {
                "mean_dataset_delta": statistics.mean(deltas),
                "dataset_wins": sum(delta > 1e-12 for delta in deltas),
                "dataset_ties": sum(abs(delta) <= 1e-12 for delta in deltas),
                "dataset_losses": sum(delta < -1e-12 for delta in deltas),
                "dataset_bootstrap_95ci": ci,
            }
    memory_types = {}
    for memory_type in ("Episodic", "Dialogue", "Semantic", "Procedural"):
        selected = {name: value for name, value in datasets.items()
                    if name.split("/", 1)[0] == memory_type}
        if not selected:
            raise ValueError("LMEB memory type is missing from dataset aggregation")
        type_metrics = {key: {metric: statistics.mean(
            d["model_metrics"][key][metric] for d in selected.values())
            for metric in metrics} for key in results}
        type_pairs = {}
        for pair_key in next(iter(selected.values()))["pairs"]:
            type_pairs[pair_key] = {}
            for metric in metrics:
                deltas = [d["pairs"][pair_key][metric]["mean_task_delta"]
                          for d in selected.values()]
                ci = _bootstrap_mean_ci(np.asarray(deltas), seed, samples)[0]
                type_pairs[pair_key][metric] = {
                    "mean_dataset_delta": statistics.mean(deltas),
                    "dataset_wins": sum(delta > 1e-12 for delta in deltas),
                    "dataset_ties": sum(abs(delta) <= 1e-12 for delta in deltas),
                    "dataset_losses": sum(delta < -1e-12 for delta in deltas),
                    "dataset_bootstrap_95ci": ci,
                }
        memory_types[memory_type] = {
            "dataset_count": len(selected), "model_metrics": type_metrics,
            "pairs": type_pairs,
        }
    if {key: value["dataset_count"] for key, value in memory_types.items()} != {
            "Episodic": 2, "Dialogue": 6, "Semantic": 8, "Procedural": 6}:
        raise ValueError("LMEB memory type dataset counts differ from preregistration")
    mean_type_metrics = {key: {metric: statistics.mean(
        memory_types[kind]["model_metrics"][key][metric] for kind in memory_types)
        for metric in metrics} for key in results}
    mean_type_pairs = {}
    for pair_key in next(iter(memory_types.values()))["pairs"]:
        mean_type_pairs[pair_key] = {}
        for metric in metrics:
            deltas = [memory_types[kind]["pairs"][pair_key][metric]["mean_dataset_delta"]
                      for kind in memory_types]
            ci = _bootstrap_mean_ci(np.asarray(deltas), seed, samples)[0]
            mean_type_pairs[pair_key][metric] = {
                "mean_type_delta": statistics.mean(deltas),
                "type_wins": sum(delta > 1e-12 for delta in deltas),
                "type_ties": sum(abs(delta) <= 1e-12 for delta in deltas),
                "type_losses": sum(delta < -1e-12 for delta in deltas),
                "type_bootstrap_95ci": ci,
            }
    for key in results:
        if (latency_ids[key] != latency_ids["jina"]
                or len(latency_by_model[key]) != len(latency_ids[key])):
            raise ValueError("model runs sampled different latency probe query IDs")
    latency = {key: {"sample_count": len(values),
                     "p50_ms": _nearest_percentile(values, .5),
                     "p95_ms": _nearest_percentile(values, .95)}
               for key, values in latency_by_model.items()}
    return {
        "protocol": {
            "dataset_macro": "equal weight across 22 LMEB datasets",
            "mean_type": "equal weight across episodic, dialogue, semantic, procedural type means",
            "within_dataset": "equal weight across source tasks; each task is a query mean",
            "task_ci": f"paired query bootstrap, {samples} resamples, seed {seed}",
            "dataset_ci": f"paired bootstrap across 22 datasets, {samples} resamples, seed {seed}",
            "latency": "deterministic up-to-20-query-per-task batch-one probe",
        },
        "dataset_count": len(datasets), "task_count": len(tasks),
        "query_count": sum(v["query_count"] for v in tasks.values()),
        "model_macro_metrics": overall_model_metrics,
        "pairs": overall_pairs, "memory_types": memory_types,
        "mean_type_metrics": mean_type_metrics, "mean_type_pairs": mean_type_pairs,
        "latency": latency,
        "datasets": datasets, "tasks": tasks,
    }


def render_full_report(results, comparison):
    metrics = (("ndcg_at_10", "nDCG@10"), ("recall_at_10", "Recall@10"),
               ("mrr_at_10", "MRR@10"), ("success_at_10", "Success@10"))
    names = {
        "jina": "Jina v5-text-nano", "bge": "BGE-M3 dense",
        "gemma": "EmbeddingGemma 2",
    }
    reference = results["jina"]
    lines = [
        "# CP-02 full LMEB dense retrieval comparison",
        "",
        "This report covers all 22 datasets, 193 source tasks and 159,011 positive-qrel queries.",
        "It reports standalone dense retrieval, not CPersona hybrid retrieval or generated-answer accuracy.",
        "The Japanese smoke and SciFact phase are separate in `2026-10-26-mps-bf16/REPORT.md`.",
        "",
        "## Source and run identity",
        "",
        f"- Dataset: `KaLM-Embedding/LMEB` @ `{REVISION}`; instruction file SHA-256 `{reference['condition']['task_instruction_sha256']}`.",
        f"- Harness Git commit: `{reference['source_commit']}`; harness SHA-256 `{reference['harness_sha256']}`.",
        "- Model revisions: " + "; ".join(
            f"{names[key]} `{results[key]['revision']}`" for key in ("jina", "bge", "gemma")) + ".",
        f"- Hardware: {reference['platform']['cpu']}, {reference['platform']['ram_bytes'] / 1024**3:.0f} GiB RAM; {reference['platform']['system']} {reference['platform']['release']}.",
        f"- Shared condition: {reference['condition']['device']}/{reference['condition']['dtype']}, max {reference['condition']['max_seq_length']} tokens, corpus/query quality batch {reference['condition']['corpus_batch']}/{reference['condition']['quality_query_batch']} inside synchronized outer chunks of up to {reference['condition']['encode_outer_chunk_size']} records.",
        "- Task instructions: Jina follows the published with-instruction setting; BGE-M3 and Gemma follow their published no-instruction setting. All models use their declared document/query prompts.",
        "",
        "## Mean (Dataset)",
        "",
        "Each task is its mean over queries; each dataset is its mean over tasks; this table weights the 22 dataset scores equally. Scores are percentages.",
        "",
        "| Model | nDCG@10 | Recall@10 | MRR@10 | Success@10 | p50 query ms | p95 query ms | Latency n |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("jina", "bge", "gemma"):
        values = comparison["model_macro_metrics"][key]
        latency = comparison["latency"][key]
        lines.append(
            f"| {names[key]} | {values['ndcg_at_10']*100:.2f} | {values['recall_at_10']*100:.2f} | "
            f"{values['mrr_at_10']*100:.2f} | {values['success_at_10']*100:.2f} | "
            f"{latency['p50_ms']:.2f} | {latency['p95_ms']:.2f} | {latency['sample_count']} |")
    lines += ["", "## Mean (Type)", "",
              "Equal-weight mean of the Episodic, Dialogue, Semantic and Procedural dataset means.",
              "",
              "| Memory type | Datasets | Jina nDCG@10 | BGE-M3 nDCG@10 | Gemma nDCG@10 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for memory_type, item in comparison["memory_types"].items():
        scores = item["model_metrics"]
        lines.append(f"| {memory_type} | {item['dataset_count']} | "
                     f"{scores['jina']['ndcg_at_10']*100:.2f} | "
                     f"{scores['bge']['ndcg_at_10']*100:.2f} | "
                     f"{scores['gemma']['ndcg_at_10']*100:.2f} |")
    lines += ["| **Mean (Type)** | 4 | "
              f"{comparison['mean_type_metrics']['jina']['ndcg_at_10']*100:.2f} | "
              f"{comparison['mean_type_metrics']['bge']['ndcg_at_10']*100:.2f} | "
              f"{comparison['mean_type_metrics']['gemma']['ndcg_at_10']*100:.2f} |",
              "", "## Runtime, resource use and truncation", "",
              "Times sum the 103 corpus groups for each model. Resource maxima are sampled every 100 ms.",
              "",
              "| Model | Dim | Load s | Corpus encode s | Batched query encode s | RSS peak GiB | MPS peak GiB | Document >512 | Query >512 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key in ("jina", "bge", "gemma"):
        run = results[key]
        groups = list(run["datasets"].values())
        dimensions = sorted({group["dimension"] for group in groups})
        dimension = ", ".join(str(value) for value in dimensions)
        corpus_time = sum(group["corpus_encode_s"] for group in groups)
        query_time = sum(group["quality_query_encode_s"] for group in groups)
        doc_truncated = sum(group["truncation"]["document"]["over_512"] for group in groups)
        query_truncated = sum(group["truncation"]["query"]["over_512"] for group in groups)
        lines.append(
            f"| {names[key]} | {dimension} | {run['load_s']:.2f} | {corpus_time:.2f} | "
            f"{query_time:.2f} | {run['resources']['peak_rss_bytes']/1024**3:.2f} | "
            f"{run['resources']['peak_mps_driver_bytes']/1024**3:.2f} | "
            f"{doc_truncated:,} | {query_truncated:,} |")
    lines += ["", "## Paired dataset-macro differences", "",
              "Deltas are percentage points, with 2,000 paired bootstrap resamples and seed 1729. The interval resamples the 22 dataset scores equally.",
              "", "| Pair | Metric | Delta pp | 95% CI pp | Dataset W/T/L | Query W/T/L |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for pair_key, pair in comparison["pairs"].items():
        for metric, label in (("ndcg_at_10", "nDCG@10"), ("recall_at_10", "Recall@10"),
                              ("mrr_at_10", "MRR@10")):
            item = pair[metric]
            ci = item["dataset_bootstrap_95ci"]
            task_wins = sum(d["pairs"][pair_key][metric]["task_wins"]
                            for d in comparison["datasets"].values())
            task_ties = sum(d["pairs"][pair_key][metric]["task_ties"]
                            for d in comparison["datasets"].values())
            task_losses = sum(d["pairs"][pair_key][metric]["task_losses"]
                              for d in comparison["datasets"].values())
            query_wins = sum(t["pairwise"][pair_key][metric]["wins"]
                             for t in comparison["tasks"].values())
            query_ties = sum(t["pairwise"][pair_key][metric]["ties"]
                             for t in comparison["tasks"].values())
            query_losses = sum(t["pairwise"][pair_key][metric]["losses"]
                               for t in comparison["tasks"].values())
            lines.append(
                f"| {pair_key.replace('_minus_', ' − ')} | {label} | {item['mean_dataset_delta']*100:.2f} | "
                f"[{ci[0]*100:.2f}, {ci[1]*100:.2f}] | "
                f"{item['dataset_wins']}/{item['dataset_ties']}/{item['dataset_losses']} | "
                f"{query_wins}/{query_ties}/{query_losses} |")
    lines += ["", "## Per-dataset nDCG@10", "",
              "| Dataset | Tasks | Queries | Jina | BGE-M3 | Gemma |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for dataset_id, item in comparison["datasets"].items():
        query_count = sum(comparison["tasks"][task]["query_count"] for task in item["tasks"])
        lines.append(f"| {dataset_id} | {item['task_count']} | {query_count} | "
                     f"{item['model_metrics']['jina']['ndcg_at_10']*100:.2f} | "
                     f"{item['model_metrics']['bge']['ndcg_at_10']*100:.2f} | "
                     f"{item['model_metrics']['gemma']['ndcg_at_10']*100:.2f} |")

    source = reference["datasets"]
    source_totals = {}
    for field in ("corpus_row_count", "duplicate_identical_corpus_rows", "qrel_row_count",
                  "positive_qrel_row_count", "duplicate_positive_qrel_pairs",
                  "missing_qrel_id_count", "missing_qrel_link_count",
                  "missing_candidate_id_count", "out_of_scene_qrel_links",
                  "duplicate_candidate_links", "orphan_qrel_query_count"):
        source_totals[field] = sum(item["source"][field] for item in source.values())
    source_totals["corpus_count_after_deduplication"] = sum(item["corpus_count"] for item in source.values())
    source_totals["unavailable_positive_qrel_links"] = sum(
        item["unavailable_positive_qrel_links"] for item in source.values())
    source_totals["candidate_query_count"] = sum(item["candidate_query_count"] for item in source.values())
    source_totals["empty_candidate_query_count"] = sum(item["empty_candidate_query_count"] for item in source.values())
    lines += ["", "## Source evidence and resources", "",
              f"- Corpus rows: {source_totals['corpus_row_count']:,} source rows; "
              f"{source_totals['corpus_count_after_deduplication']:,} per-corpus unique records; "
              f"{source_totals['duplicate_identical_corpus_rows']:,} exact repeated rows collapsed.",
              f"- Positive qrels: {source_totals['positive_qrel_row_count']:,} rows, "
              f"{source_totals['duplicate_positive_qrel_pairs']} repeated same-grade pairs; "
              f"{source_totals['missing_qrel_link_count']} absent-corpus links to "
              f"{source_totals['missing_qrel_id_count']} corpus-scoped IDs; "
              f"{source_totals['out_of_scene_qrel_links']} links outside published scenes.",
              f"- Candidate scenes: {source_totals['candidate_query_count']:,} queries; "
              f"{source_totals['missing_candidate_id_count']:,} missing candidate IDs; "
              f"{source_totals['empty_candidate_query_count']} empty available pools.",
              f"- Positive qrel links unavailable in each query's effective search pool: "
              f"{source_totals['unavailable_positive_qrel_links']:,} (union count; categories can overlap).",
              f"- Orphan qrel queries: {source_totals['orphan_qrel_query_count']}; model errors: 0.",
              "- Full per-query top-10 IDs and metrics are in the compressed model result files; normalized input text is not included.",
              "", "## Limits", "",
              "- Mac MPS/bf16 timings and sampled resource maxima apply only to this recorded hardware and software. Resource sampling can miss brief peaks.",
              "- Batch-one latency uses up to 20 deterministically selected query IDs per source task; quality retrieval uses query batch 8.",
              "- Public positive qrels unavailable in the source corpus or an admissible candidate pool remain in Recall/NDCG denominators and can lower scores.",
              "- This does not evaluate CPersona's hybrid search, answer generation, or Japanese quality beyond the separate 10-query synthetic smoke.",
              ""]
    return "\n".join(lines)


def compare_full(args):
    results = {Path(path).stem.split(".")[0]: _read_json(path) for path in args.result}
    comparison = compare_full_results(results)
    Path(args.out).write_text(json.dumps(comparison, indent=2))
    report_path = Path(args.report) if args.report else Path(args.out).with_suffix(".md")
    report_path.write_text(render_full_report(results, comparison))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--snapshot", required=True)
    prepare_parser.add_argument("--task-instructions", required=True)
    prepare_parser.add_argument("--out", required=True)
    prepare_parser.set_defaults(action=lambda a: prepare(a.snapshot, a.out, a.task_instructions))
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", choices=("jina", "bge", "gemma"), required=True)
    run_parser.add_argument("--model-root", required=True)
    run_parser.add_argument("--input-dir", required=True)
    run_parser.add_argument("--task-instructions", required=True)
    run_parser.add_argument("--checkpoint-dir", required=True)
    run_parser.add_argument("--out", required=True)
    run_parser.set_defaults(action=run)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--result", action="append", required=True)
    compare_parser.add_argument("--out", required=True)
    compare_parser.add_argument("--report")
    compare_parser.set_defaults(action=compare_full)
    args = parser.parse_args()
    result = args.action(args)
    if args.command == "prepare":
        print(json.dumps({k: v for k, v in result.items() if k != "groups"}), flush=True)
