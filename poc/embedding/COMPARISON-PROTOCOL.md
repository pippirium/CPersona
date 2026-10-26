# CP-02 — Preregistered dense comparison protocol

Status: Phase 1 (Japanese smoke plus SciFact) is measured and reported separately. The full-LMEB Phase 2 conditions below were fixed before its model inference. The user authorized evaluation work and PR publication in this experimental fork. Neither phase changes the MCP runtime or provider contracts.

## Fixed inputs

| Component | Revision / condition |
| --- | --- |
| CPersona reference PoC | fork `7821840f114866673022edfc5fe4b21a9b08a5e5`, `poc/embedding/sample.json` (10 Japanese synthetic smoke queries) |
| LMEB task | full published `mteb/LMEB_SciFact` test at `432cf48c879c2de792720d5c0af6900312ebe7b6`: 1,748 corpus passages, 188 queries, 366 qrel rows |
| Jina | `jinaai/jina-embeddings-v5-text-nano` @ `8a7f00aac812071b69403df470f1038ec85f8925` |
| BGE-M3 | `BAAI/bge-m3` @ `5617a9f61b028005a4858fdac845db406aefb181` |
| Gemma | `google/embeddinggemma-2` @ `914f7f89142e33e77833254d9c9b90c3cef7303b` |

The LMEB [paper](https://arxiv.org/abs/2603.12572) and [processed dataset](https://huggingface.co/datasets/mteb/LMEB_SciFact) define this as one semantic-memory task. Running it is not a claim to reproduce the entire 22-dataset LMEB aggregate. The Japanese data remains the original PoC smoke set, not a representative Japanese benchmark. Every model receives all identical records, query IDs and qrels of each dataset. No query-selected candidate reduction, reranker or generated answer is used.

## Conditions fixed before measurement

- Isolated standalone evaluation worker per model; no CPersona live database, MCP configuration or existing provider index. Model source is not patched. No generation LLM or external embedding service.
- Primary run: one Mac / MPS / bf16 for all three models. CPU/fp32 is a separate unmeasured condition, never mixed into these results. Fixed batch size 8 for corpus, 1 for timed queries; common max sequence length 512. Report tokenizer-specific truncation and native dimensions rather than assuming identical tokenization.
- Before full inference, bounded encoder dispatch after a same-machine CP-02 run reported MPS queue stalls on larger corpora: each model call is split into outer chunks of at most 1,024 records, with MPS synchronization and cache release after each chunk. The inner batch sizes remain 8 for corpus and quality queries and 1 for latency probes; 1,024 is divisible by 8, preserving batch boundaries. This changes dispatch and memory scheduling, not inputs, model roles, ranking, or metrics. The harness records the chunk size in run provenance and emits per-chunk progress.
- Gemma: text-only configuration (vision/audio disabled); Document/SearchQuery prompts. Jina: pinned publisher custom code, retrieval task and document/query prompts. BGE-M3: dense SentenceTransformer encode without query instruction, per [publisher usage](https://huggingface.co/BAAI/bge-m3). No sparse or multi-vector score is added.
- Build fresh normalized document vectors per model and dataset, use cosine/dot-product ranking with stable corpus-order tie breaks. No shared vector cache or mixed-model index. Corpus text is title plus body when title is nonempty, using the same assembly for every model.
- Two untimed document/query priming calls per dataset before corpus encoding and timed query execution. Explicit device synchronization bounds timings. Report model load, corpus encoding, query embedding and ranking separately. No cross-run warm/cold inference from these timings.
- Recall@1/5/10, Success@10, MRR@10 and graded NDCG@10 (`2^grade-1` gain), query counts and missing/error counts. Preserve per-query IDs, ranks and metrics; do not commit corpus/query text or private paths.
- Sample process RSS and MPS driver allocation during inference; record sampling interval and limits. Model download time is excluded. Record package versions, exact model/data revisions, input fingerprints, native dimensions and truncation counts.
- Paired comparison uses identical query IDs, win/tie/loss and seeded bootstrap 95% intervals (seed 1729; 2,000 resamples) for NDCG@10/Recall@10/MRR@10. Report Japanese smoke and LMEB task separately. Do not combine them into a general model-selection verdict.

## Phase 2 — full LMEB (fixed before full-model inference)

### Inputs and source audit

- Source is `KaLM-Embedding/LMEB` at immutable revision `f9d4294be4a24b8c16d6bfb59d56a6fec4bd95c0`, downloaded before inference. It contains 22 dataset groups, 193 query tasks and 103 corpus groups. The normalized inputs contain 2,032,230 per-corpus unique records (summed across groups) and 159,011 queries with positive qrels; no source query is excluded for missing positive qrels.
- Corpus rows use `title + newline + text` when title is present, otherwise `text`; IDs are converted to strings and sorted lexically. Repeated IDs are accepted only when the retained ID and normalized text are identical. QASPER has 44,792 such exact duplicate rows (65,300 source rows, 20,508 unique); any conflicting row for one ID aborts preparation. Four repeated positive qrel pairs have the same grade and collapse to one query/document relationship. Conflicting grades abort preparation.
- 28,587 queries have published candidate scenes; those queries rank only the available documents in their published scene. Other queries rank the full corpus group. Candidate IDs absent from the source corpus are omitted from the available pool and counted (4,588 corpus-group-scoped IDs); no query has an empty available pool. Scene assignment must match exactly one scene ID or `scene_id + "_"` prefix; unmatched or ambiguous IDs abort preparation.
- All positive qrels stay in Recall and NDCG denominators, including 109 qrel links to 96 corpus-group-scoped IDs absent from the corresponding corpus and 72 links outside their published candidate scene. These limits are counted and never silently removed to improve scores. There are no orphan qrel query IDs. The two unavailable-qrel categories can overlap and are not added together.

### Full-run conditions

- Run all three fixed model snapshots listed above in separate processes on the same Apple M1 Pro / 32 GB Mac using MPS and bf16. Stay offline during inference. Each run uses max sequence length 512 and corpus/query quality batch 8 inside synchronized outer encoding chunks of at most 1,024 records. Gemma uses text-only Document/SearchQuery prompts; Jina uses pinned retrieval custom code with document/query prompts; BGE-M3 uses dense encoding without a query instruction. Store native output dimensions.
- Follow the LMEB model-specific instruction settings in [the paper's Table 13](https://arxiv.org/abs/2603.12572): Jina gets the pinned task instruction while BGE-M3 and EmbeddingGemma do not. Use `Instruct: {task instruction}\nQuery: {query}` before Jina's retrieval query prompt. The instruction map is `task_instructions.json` at the same dataset revision, SHA-256 `f3f83069f3a3818a89c28de320051521bc37c6b8314d110b8911356eb9adfaa8`; resolve and validate an instruction for every one of the 193 source tasks before inference.
- Build fresh normalized vectors for each corpus group and model. Use cosine scores and deterministic top-10 ranking; ties follow ascending normalized corpus ID. Never share vectors between models. Warm each role twice for each corpus group before timed stages. Report corpus and batched query embedding time separately; batched quality time is not presented as single-query latency.
- Measure single-query latency separately after quality scoring. For each of the 193 source tasks, select up to 20 query IDs deterministically (seed 1729 plus a stable task hash), then time batch-one embedding and CPU ranking with device synchronization. Report sample count and nearest-rank p50/p95. This is a stratified task sample, not a latency claim for all queries.
- Compute Recall@1/5/10, Success@10, MRR@10 and graded NDCG@10 (`2^grade-1`) for every query. Preserve query IDs, task IDs, top-10 IDs, metrics and unavailable-qrel counts in compressed results; never commit public query/corpus text or local paths. Report tokenizer-specific document/query truncation counts at the shared 512-token limit.
- A task score is the mean of its query scores. A dataset score averages its source task scores equally; the 22-dataset macro averages dataset scores equally, matching LMEB's Mean (Dataset). Also report the paper's Mean (Type), the equal-weight mean of the four memory-type dataset averages. Per-task paired 95% intervals resample query IDs (2,000 resamples, seed 1729); per-dataset intervals resample tasks; the overall interval resamples the 22 datasets equally. Report wins/ties/losses at query and task levels. Keep this aggregate distinct from the Phase 1 SciFact regime and the downstream Track B answer benchmark.
- Sample process RSS and MPS driver allocation every 100 ms. Save immutable model/data revisions, normalized input fingerprints, harness hash, Git source commit, package versions, hardware, truncation counts and stage timings. Resume only checkpoints matching all of those identities and controls.
- This measures standalone dense retrieval only. It does not measure CPersona hybrid retrieval, answer generation, memory quality, or a general Japanese benchmark. Sampling can miss brief resource peaks; latency and resource results apply only to this Mac/MPS/bf16 condition.

## Delivery boundary

Deliver the standalone evaluator, testable metrics, measured reports and compressed per-query evidence in an independent PR to the user's `pippirium/CPersona` fork's `master`. Bridge #19 tracks cross-repository completion. Never push or open a PR against `Cloto-dev/CPersona`. Existing published PoC/LMEB scores are reference history and are not copied into a new run as if remeasured.
