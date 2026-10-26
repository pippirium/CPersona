"""Model-free checks of CP-02 pairing and graded metrics; no downloads."""
import pytest
from poc.embedding.compare import paired_comparison, ranking_metrics, validate_dataset


def test_multiple_relevant_grades_and_rank_cutoffs():
    result = ranking_metrics(["irrelevant", "a", "b"], {"a": 3, "b": 1, "missing": 2})
    assert result["recall_at_1"] == 0
    assert result["recall_at_5"] == pytest.approx(2 / 3)
    assert result["mrr_at_10"] == .5
    assert 0 < result["ndcg_at_10"] < 1
    assert ranking_metrics([str(i) for i in range(10)], {"absent": 1})["mrr_at_10"] == 0


def test_dataset_rejects_missing_evidence_and_duplicate_ids():
    data = {"records": [{"id": "a", "text": "synthetic"}],
            "queries": [{"id": "q", "text": "test", "relevant": {"a": 1}}]}
    validate_dataset(data)
    data["queries"][0]["relevant"] = {"unknown": 1}
    with pytest.raises(ValueError):
        validate_dataset(data)
    data["queries"][0]["relevant"] = {"a": 1}
    data["records"] *= 2
    with pytest.raises(ValueError):
        validate_dataset(data)


def _run(values, sha="same"):
    return {"dataset_sha256": sha, "errors": 0,
            "per_query": [{"query_id": str(i), "metrics": {k: v for k in
                          ("ndcg_at_10", "recall_at_10", "mrr_at_10")}}
                          for i, v in enumerate(values)]}


def test_paired_bootstrap_is_seeded_and_checks_alignment():
    left, right = _run([1, .5, 0]), _run([.5, .5, .5])
    result = paired_comparison(left, right)
    assert result == paired_comparison(left, right)
    for value in result.values():
        assert (value["wins"], value["ties"], value["losses"]) == (1, 1, 1)
        assert value["mean_delta_left_minus_right"] == 0
        assert value["bootstrap_95ci"][0] <= 0 <= value["bootstrap_95ci"][1]
    with pytest.raises(ValueError):
        paired_comparison(left, _run([1, .5, 0], "other"))
    with pytest.raises(ValueError):
        paired_comparison(left, _run([1]))
    right["errors"] = 1
    with pytest.raises(ValueError):
        paired_comparison(left, right)


def test_compare_rejects_duplicate_model_results(tmp_path):
    import json
    from types import SimpleNamespace
    from poc.embedding.compare import compare, MODELS

    model, revision = MODELS["jina"]
    paths = []
    for name in ("a", "b", "c"):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({"model": model, "revision": revision}))
        paths.append(str(path))
    with pytest.raises(ValueError, match="distinct"):
        compare(SimpleNamespace(result=paths, out=str(tmp_path / "result.json")))
    assert not (tmp_path / "result.json").exists()


def test_full_source_unretrievable_qrels_stay_in_recall_denominator():
    data = {"records": [{"id": "available", "text": "synthetic"}],
            "queries": [{"id": "q", "text": "test",
                         "relevant": {"available": 1, "missing": 1}}]}
    with pytest.raises(ValueError):
        validate_dataset(data)
    validate_dataset(data, allow_unretrievable_qrels=True)
    assert ranking_metrics(["available"], data["queries"][0]["relevant"])["recall_at_10"] == .5


def test_candidate_scene_can_be_empty_only_for_audited_source_policy():
    data = {"records": [{"id": "available", "text": "synthetic"}],
            "queries": [{"id": "q", "text": "test", "relevant": {"missing": 1},
                         "candidates": []}]}
    with pytest.raises(ValueError):
        validate_dataset(data)
    validate_dataset(data, allow_unretrievable_qrels=True)


def test_full_lmeb_canonicalizes_only_identical_duplicate_records():
    from poc.embedding.full_lmeb import canonicalize_records, candidate_scene, stable_top_ids

    records, raw_count, duplicate_count = canonicalize_records([
        {"id": "doc", "title": "Title", "text": "Body"},
        {"id": "doc", "title": "Title", "text": "Body"},
        {"id": 2, "text": "Another"},
    ])
    assert (raw_count, duplicate_count) == (3, 1)
    assert records == [{"id": "2", "text": "Another"},
                       {"id": "doc", "text": "Title\nBody"}]
    with pytest.raises(ValueError, match="conflicting"):
        canonicalize_records([{"id": "doc", "text": "one"},
                              {"id": "doc", "text": "two"}])
    assert candidate_scene("scene_1_turn_2", {"scene_1": ["a"]}) == ["a"]
    with pytest.raises(ValueError):
        candidate_scene("scene_10_turn_2", {"scene_1": ["a"]})
    with pytest.raises(ValueError):
        candidate_scene("scene_1_turn_2", {"scene_1": ["a"],
                                            "scene_1_turn_2": ["b"]})
    assert stable_top_ids([.8, .9, .9, .9, .1], ["a", "b", "c", "d", "e"],
                          limit=2) == ["b", "c"]
    assert stable_top_ids([.8, .9, .9, .9, .1], ["a", "b", "c", "d", "e"],
                          candidate_positions=[3, 0, 2], limit=2) == ["c", "d"]


def test_full_lmeb_encode_chunks_synchronize_and_preserve_order():
    import numpy as np
    from poc.embedding.full_lmeb import encode_in_chunks

    batches, syncs, progress = [], [], []

    def encode(chunk):
        batches.append(list(chunk))
        return np.asarray([[value, value + 1] for value in chunk])

    vectors = encode_in_chunks(list(range(9)), 4, encode,
                               lambda: syncs.append(True),
                               lambda completed, total: progress.append((completed, total)))
    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    assert len(syncs) == 3
    assert progress == [(4, 9), (8, 9), (9, 9)]
    assert vectors.tolist() == [[value, value + 1] for value in range(9)]
    from poc.embedding.full_lmeb import ENCODE_CHUNK_SIZE, _condition
    policy = _condition("sha")
    assert ENCODE_CHUNK_SIZE == policy["encode_outer_chunk_size"] == 1024
    assert policy["mps_synchronize_after_each_encode_chunk"] is True
    assert policy["mps_empty_cache_after_large_encode_chunks"] is True
    with pytest.raises(ValueError, match="empty"):
        encode_in_chunks([], 4, encode, lambda: None)


def test_full_lmeb_task_pairing_uses_query_means_and_seeded_ci():
    from poc.embedding.full_lmeb import compare_task_rows

    metrics = ("ndcg_at_10", "recall_at_10")
    values = {
        "jina": [1.0, .5], "bge": [.5, .5], "gemma": [.5, .5],
    }
    rows = {model: {f"task#q{i}": {"metrics": {
        "ndcg_at_10": score, "recall_at_10": score}}
        for i, score in enumerate(scores)} for model, scores in values.items()}
    rows["jina"]["other#q"] = {"metrics": {"ndcg_at_10": 0, "recall_at_10": 0}}
    model_metrics, pairs = compare_task_rows(
        rows, ["task#q0", "task#q1"], ["jina", "bge", "gemma"], metrics,
        samples=200, seed=1729)
    assert model_metrics["jina"]["ndcg_at_10"] == .75
    result = pairs["bge_minus_jina"]["ndcg_at_10"]
    assert (result["wins"], result["ties"], result["losses"]) == (0, 1, 1)
    assert result["mean_delta"] == -.25
    assert result["query_bootstrap_95ci"] == compare_task_rows(
        rows, ["task#q0", "task#q1"], ["jina", "bge", "gemma"], metrics,
        samples=200, seed=1729)[1]["bge_minus_jina"]["ndcg_at_10"]["query_bootstrap_95ci"]
    with pytest.raises(ValueError):
        compare_task_rows(rows, ["not-present"], ["jina", "bge", "gemma"], metrics)


def test_full_lmeb_instruction_mapping_preserves_model_specific_settings():
    from poc.embedding.full_lmeb import instruction_for_task, query_input

    instructions = {
        "EPBench": {"Entities": "retrieve event participants"},
        "MemGovern": {"MemGovern": "retrieve software fixes"},
        "CovidQA": {"Covid-QA": "retrieve COVID evidence"},
    }
    query = {"task": "Episodic/EPBench/default_claude_long/Entities", "text": "who went?"}
    assert instruction_for_task(query["task"], instructions) == "retrieve event participants"
    assert query_input(query, "jina", instructions) == (
        "Instruct: retrieve event participants\nQuery: who went?")
    assert query_input(query, "bge", instructions) == "who went?"
    assert query_input(query, "gemma", instructions) == "who went?"
    assert instruction_for_task("Procedural/MemGovern/example-project", instructions) == (
        "retrieve software fixes")
    assert instruction_for_task("Semantic/Covid-QA", instructions) == "retrieve COVID evidence"
    with pytest.raises(ValueError, match="missing"):
        instruction_for_task("Semantic/Unknown", instructions)


def test_full_lmeb_report_contains_only_aggregated_evidence():
    from poc.embedding.full_lmeb import render_full_report

    metric_names = ("ndcg_at_10", "recall_at_10", "mrr_at_10", "success_at_10")
    metric_values = {key: {metric: .5 for metric in metric_names}
                     for key in ("jina", "bge", "gemma")}
    source = {
        "corpus_row_count": 2, "duplicate_identical_corpus_rows": 1,
        "qrel_row_count": 1, "positive_qrel_row_count": 1,
        "duplicate_positive_qrel_pairs": 0, "missing_qrel_id_count": 0,
        "missing_qrel_link_count": 0, "missing_candidate_id_count": 0,
        "out_of_scene_qrel_links": 0, "duplicate_candidate_links": 0,
        "orphan_qrel_query_count": 0,
    }
    condition = {"task_instruction_sha256": "sha", "device": "mps", "dtype": "bf16",
                 "max_seq_length": 512, "corpus_batch": 8, "quality_query_batch": 8,
                 "encode_outer_chunk_size": 1024,
                 "mps_synchronize_after_each_encode_chunk": True,
                 "mps_empty_cache_after_large_encode_chunks": True}
    results = {key: {"condition": condition, "source_commit": "commit",
                     "revision": "revision", "load_s": 1.0,
                     "resources": {"peak_rss_bytes": 1024**3,
                                   "peak_mps_driver_bytes": 1024**3},
                     "harness_sha256": "harness", "platform": {
                         "cpu": "CPU", "ram_bytes": 32 * 1024**3,
                         "system": "Darwin", "release": "test"},
                     "datasets": {"Semantic/SciFact": {"source": source,
                         "dimension": 768, "corpus_encode_s": 2.0,
                         "quality_query_encode_s": 1.0,
                         "truncation": {"document": {"over_512": 0},
                                        "query": {"over_512": 0}},
                         "corpus_count": 1, "candidate_query_count": 1,
                         "empty_candidate_query_count": 0,
                         "unavailable_positive_qrel_links": 0}}}
               for key in ("jina", "bge", "gemma")}
    pair_key = "bge_minus_jina"
    paired_metric = {"mean_dataset_delta": 0.0, "dataset_bootstrap_95ci": [0.0, 0.0],
                     "dataset_wins": 0, "dataset_ties": 1, "dataset_losses": 0,
                     "task_wins": 0, "task_ties": 1, "task_losses": 0}
    comparison = {
        "model_macro_metrics": metric_values,
        "latency": {key: {"p50_ms": 1.0, "p95_ms": 2.0, "sample_count": 1}
                    for key in metric_values},
        "memory_types": {kind: {"dataset_count": 1, "model_metrics": metric_values}
                         for kind in ("Episodic", "Dialogue", "Semantic", "Procedural")},
        "mean_type_metrics": metric_values,
        "pairs": {pair_key: {metric: paired_metric
                             for metric in ("ndcg_at_10", "recall_at_10", "mrr_at_10")}},
        "datasets": {"Semantic/SciFact": {"task_count": 1, "model_metrics": metric_values,
            "tasks": ["Semantic/SciFact"], "pairs": {pair_key: {
                metric: {"task_wins": 0, "task_ties": 1, "task_losses": 0}
                for metric in ("ndcg_at_10", "recall_at_10", "mrr_at_10")}}}},
        "tasks": {"Semantic/SciFact": {"query_count": 1, "pairwise": {pair_key: {
            metric: {"wins": 0, "ties": 1, "losses": 0}
            for metric in ("ndcg_at_10", "recall_at_10", "mrr_at_10")}}}},
    }
    report = render_full_report(results, comparison)
    assert "Mean (Dataset)" in report and "Mean (Type)" in report
    assert "exact repeated rows" in report
    assert "Instruct:" not in report and "CPU" in report
