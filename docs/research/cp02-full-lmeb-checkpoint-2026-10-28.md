# CP-02 full LMEB checkpoint — 2026-10-28

Status: measurement, partial. The requested pause was taken after BGE-M3 saved its LongMemEval group 8 checkpoint. Jina is complete; BGE-M3 and Gemma do not yet have comparable full-run results.

This is a partial checkpoint report, not a three-model ranking. Conditions were preregistered in [`poc/embedding/COMPARISON-PROTOCOL.md`](../../poc/embedding/COMPARISON-PROTOCOL.md). The current branch handoff pushes code and this report only; it does not create a PR.

## 1. Conditions and provenance

- Dataset: `KaLM-Embedding/LMEB` at `f9d4294be4a24b8c16d6bfb59d56a6fec4bd95c0`; 22 datasets, 193 retrieval tasks, 103 corpus groups and 159,011 positive-qrel queries.
- Task instructions: SHA-256 `f3f83069f3a3818a89c28de320051521bc37c6b8314d110b8911356eb9adfaa8`.
- Models: Jina v5-text-nano `8a7f00aac812071b69403df470f1038ec85f8925`; BGE-M3 `5617a9f61b028005a4858fdac845db406aefb181`; EmbeddingGemma 2 `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- Hardware and execution: Apple M1 Pro, 32 GB unified memory, MPS/bf16, max sequence length 512, batch 8 for corpus and quality queries, outer encoder chunks up to 1,024. Models ran in separate workers.
- Harness source commit: `bd1c29d737ecdc8b3e02f84364d088c0554b376d`. The full-LMEB checkpoint validation confirmed the saved input and harness identities match the current pinned run.
- Jina result provenance: composite harness fingerprint `f2a1dead9ecfb86612b840646c4f0f539bcbfa8a0fdbb7ba8add9e5cc1ef38cb`.

## 2. Jina — complete

Jina completed all 103 corpus groups and 193 tasks: 22 datasets, 159,011 scored queries, zero errors. Values below are newly aggregated from this run using the preregistered equal-task, equal-dataset and equal-memory-type rules. They are descriptive model aggregates; no paired three-model test is available yet.

| Metric | Mean (Dataset), 22 datasets | Mean (Type), 4 types |
| --- | ---: | ---: |
| Recall@1 | 0.326083 | 0.336396 |
| Recall@5 | 0.563917 | 0.567626 |
| Recall@10 | 0.664546 | 0.661801 |
| Success@10 | 0.810033 | 0.811694 |
| MRR@10 | 0.580861 | 0.592297 |
| nDCG@10 | 0.558271 | 0.563594 |

Latency used 3,554 batch-one samples: p50 40.22975 ms, p95 62.038916 ms. Total corpus embedding time was 41,947.9953 s and quality-query embedding time was 1,777.39494 s. The tokenizer diagnostic counted 19,009 corpus documents over 512 tokens and no quality queries over 512.

## 3. BGE-M3 — paused after group 8

Eight of 103 corpus-group checkpoints are present. LongMemEval group 8 completed with 237,655 corpus documents, 500 queries and zero errors. Its group-only metrics are:

| Recall@1 | Recall@5 | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.442900 | 0.818867 | 0.889133 | 0.938000 | 0.792825 | 0.786695 |

These values describe that one corpus group only. They are not a BGE full-LMEB aggregate. Documents over 512 tokens: 22,456 of 237,655; longest observed input was 17,246 tokens before the encoder's 512-token limit. Queries did not exceed 512 tokens. The long-sequence warning came from a separate post-inference length-count diagnostic run without truncation; the model inference itself used the registered 512-token limit.

The worker was stopped after this checkpoint, following the user's instruction to pause when the remaining roughly 41,000-document segment finished. No BGE full-result file was produced. This run had no recorded inference errors, but the available resource record is not enough to infer spare unified-memory capacity.

## 4. Gemma and comparison status

EmbeddingGemma 2 has not been run for full LMEB. Since BGE is partial and Gemma is unrun, do not report wins/ties/losses, bootstrap intervals or a final model recommendation. The Jina values above and the BGE group 8 values use different coverage.

## 5. Resume and artifact handling

The branch contains the harness and this checkpoint summary, not the full-LMEB normalized query/corpus text, model weights, checkpoint blobs or compressed per-query results. Those runtime files are ignored and were not pushed; the separately committed Phase 1 smoke/SciFact files are not substitutes. The Jina full result currently exists only as a local ignored artifact (`cp02-full-results/jina.json.gz`); a second PC needs that file copied through a private channel or must rerun Jina before it can produce paired results.

When evaluation resumes, first verify that the ignored normalized inputs and BGE checkpoints match the dataset, task-instruction, model, harness and execution fingerprints above. If they match, continue BGE from the first missing group (group 9); otherwise regenerate the isolated inputs and rerun it. Then run Gemma under the same registered conditions and call `full_lmeb compare` with all three complete per-query results. Do not add raw benchmark text or local paths to Git.

Example BGE resume command (replace placeholders with the local isolated directories):

```sh
python -m poc.embedding.full_lmeb run --model bge \
  --model-root <model-snapshot>/5617a9f61b028005a4858fdac845db406aefb181 \
  --input-dir <normalized-input-dir> \
  --task-instructions <pinned-task_instructions.json> \
  --checkpoint-dir <checkpoint-dir> \
  --out <result-dir>/bge.json.gz
```

## 6. Interpretation

The measured outcomes so far establish that Jina completed the registered dense-retrieval workload and that BGE-M3 completed eight saved corpus-group checkpoints before the requested pause. They do not establish relative model quality, CPersona hybrid retrieval quality, generated-answer accuracy or Japanese general quality. The next decision point is a complete, paired run after all three models have compatible full-LMEB results.
