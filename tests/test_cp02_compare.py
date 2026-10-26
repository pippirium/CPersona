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
