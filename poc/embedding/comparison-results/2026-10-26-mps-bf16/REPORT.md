# CP-02 phase 1: fixed-condition dense comparison

Actual inference at harness commit `9544110`: MPS/bf16, max sequence 512, corpus batch 8/query batch 1, two untimed document/query priming calls per dataset. Each model ran in a separate interpreter against the identical normalized inputs recorded in dataset-manifest.json. No CPersona live database, MCP client configuration, provider index or generation LLM was used.

This is **one complete published LMEB_SciFact task plus the original Japanese synthetic smoke**, not the full 22-dataset LMEB aggregate or CPersona hybrid/Track B evaluation. Existing published CPU/Track B scores are not mixed with these new measurements. No model-selection verdict follows from this phase.

| dataset | model | queries | NDCG@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 / p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| japanese_smoke | jina | 10 | 0.8762 | 0.7000 | 1.0000 | 1.0000 | 0.8333 | 35.8 / 193.1 |
| japanese_smoke | bge | 10 | 0.8987 | 0.8000 | 0.9000 | 1.0000 | 0.8667 | 23.7 / 109.4 |
| japanese_smoke | gemma | 10 | 0.9000 | 0.8000 | 1.0000 | 1.0000 | 0.8667 | 37.6 / 42.1 |
| lmeb_scifact | jina | 188 | 0.8228 | 0.4763 | 0.8494 | 0.9657 | 0.8038 | 35.4 / 42.1 |
| lmeb_scifact | bge | 188 | 0.7644 | 0.4340 | 0.8013 | 0.9201 | 0.7540 | 26.0 / 34.6 |
| lmeb_scifact | gemma | 188 | 0.7606 | 0.4313 | 0.8177 | 0.9404 | 0.7389 | 34.7 / 43.1 |

| LMEB_SciFact paired NDCG@10 | mean left-right | bootstrap 95% CI | win / tie / loss |
| --- | ---: | --- | --- |
| bge_minus_gemma | 0.0038 | [-0.0322, 0.0416] | 68 / 62 / 58 |
| bge_minus_jina | -0.0583 | [-0.0889, -0.0288] | 36 / 79 / 73 |
| gemma_minus_jina | -0.0622 | [-0.0944, -0.0321] | 31 / 77 / 80 |

| model | native dimension | load seconds | sampled peak RSS MiB | sampled peak MPS driver MiB |
| --- | ---: | ---: | ---: | ---: |
| jina | 768 | 3.68 | 1754.3 | 1400.1 |
| bge | 1024 | 3.76 | 3541.0 | 2194.6 |
| gemma | 768 | 1.70 | 1128.5 | 1280.4 |

Per-query ranks/metrics and embedding/ranking times are in jina.json, bge.json and gemma.json. paired.json contains all paired metrics, win/tie/loss and 2,000-resample seed-1729 confidence intervals. Dataset/model revisions, input fingerprints, exact dependency versions, source commit and harness hash are retained. All 198 queries per model completed without errors or missing qrels. Tokenizer checks found **zero truncations** for all three models in this phase.

Limits: Japanese has only ten synthetic queries; LMEB_SciFact is one semantic-memory task. Native dimensions differ (cosine-normalized vectors are rebuilt per model); this is a deployment-native dense comparison, not a forced equal-dimension ablation. Sampling every 100ms may miss transient peaks; MPS driver allocation is not total system VRAM. Cold-start/steady-state paired baselines, concurrent system load and energy were not controlled/measured. Query timings are descriptive observations after the declared priming, not a universal speed claim. CPU/fp32, full LMEB, downstream hybrid retrieval and final-answer accuracy remain unmeasured by this phase. Dataset/model downloads are separate from inference; no packet-capture offline audit was performed.

Reproduce with requirements-compare.txt, compare.py prepare/run/compare and COMPARISON-PROTOCOL.md. Input text and local path maps intentionally remain outside Git; identifiers, hashes and outcomes suffice to audit the run without publishing corpus/query text.
