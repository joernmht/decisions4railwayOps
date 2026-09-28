# H2 routing notes (P2-H2a, P2-H2b routing half)

Dated entries, newest first. The orchestrator merges these into `docs/lab-notebook.md`.

Hypothesis labels (`docs/hypotheses.md` v0.1): **P2-H2a** = closed-taxonomy routing (this file);
**P2-H2b** = requirement -> constraint-family routing (this file) plus relevance filtering
(`NOTES_filtering.md`); **P2-H2c** (semantic pre-solve gate) is not yet tested.

## 2026-09-28: post-verification fixes (privacy, held-out gate)

- **Private class names removed from the public files.** The 30 H2a class names and the
  descriptions written for them now live outside every repository, in
  `~/.local/share/d4r/h2a_class_descriptions.json` (override: `D4R_CLASS_DESCRIPTIONS`), and
  `routing.py` loads them lazily. Without that file every class is described by its name alone.
  The request payloads are unchanged, so the offline replay still hits every cached request
  (0 live calls, 0 errors), and `routing_summary.json` is identical apart from the renamed keys.
- **Opaque ids.** Public results name the H2a classes C01.., V01.., P01.., O01.., numbered by
  sorted class name per kind. The id -> name map is written only to
  `~/.local/share/d4r/h2a_class_ids.json`. Tests use dummy classes.
- **Gitignored caches.** `runs/cache/h2-routing-{jev,deepseek,emb}.jsonl` carry class labels or
  catalogue-derived requests, so they are gitignored and kept locally for replay. H2b names come
  from lp2graph's public `ConstraintDomainClass` enum and stay.
- **Held-out gate.** θ = 0.9 was the in-sample best of 6 thresholds. The headline is now the
  5-fold cross-validated gate (below): 0.844, not 0.850. The verifier's independent
  implementation also gave 0.844 (range 0.837–0.850).
- **Offline strictness.** In `--offline` mode a cache miss now raises instead of failing open.

## 2026-09-28: routing results for P2-H2a (catalogue) and P2-H2b (synthetic requirements)

**What ran.** `python experiments/h2/run_routing.py` (all arms, 6 workers), then `--offline`
replays. Replays reproduce the output byte for byte. Models: Jev `jev-1.13.0` (served
`jev-1.13.0` on every call), DeepSeek `deepseek-flash` (non-thinking, JSON mode, T=0),
Qwen3-Embedding-4B on ScaDS.AI, plus a pure-Python TF-IDF and a keyword baseline.
Caches: `runs/cache/h2-routing-{jev,deepseek,emb}.jsonl` (gitignored, local only), 849 Jev and
849 DeepSeek records.
**Spend: Jev $0.049 (1.16 M input tokens), DeepSeek $0.032.** The caps were $0.20 and $1.00.
Code: `src/d4r/h2/{routing,requirements}.py`. Tests: `tests/test_h2_routing.py` (27 offline
tests). Results: `results/h2/routing_summary.{md,json}`, plus the P2-H2b dataset
`results/h2/routing_h2b_statements.json`.

**P2-H2a: closed-taxonomy routing on the private literature catalogue.** 319 rows: 123
constraints, 48 variables, 123 parameters and 25 objectives, over 30 classes (`ml_class`). The
data is read at runtime from `D4R_CATALOG_DIR`, and only aggregates are written. Class names and
descriptions are read from the private `D4R_CLASS_DESCRIPTIONS` file; results use opaque class
ids. Pooled results:

| arm | accuracy | macro-F1 | p50 / p95 ms | USD / 1k |
|---|---|---|---|---|
| Jev Choice | 0.818 | 0.791 | 250 / 506 | 0.028 |
| DeepSeek JSON | 0.840 | 0.813 | 999 / 1299 | 0.037 |
| embedding nearest-description | 0.345 | 0.335 | 11 / 54 | 0 |
| embedding kNN-1, leave-one-out (sees gold labels) | 0.765 | 0.736 | 12 / 24 | 0 |
| TF-IDF nearest-description | 0.530 | 0.515 | – | 0 |

- **Jev minus DeepSeek:** −0.022 (paired bootstrap 95% CI −0.056 to +0.013), 13 vs 20 discordant
  pairs.
  - Per set: constraints −0.057 (CI −0.114 to 0.000), variables −0.021, parameters −0.008,
    objectives +0.080.
  - The point estimate is inside the pre-registered 5-point margin. The CI's lower bound
    (−5.6 pts) does not formally establish non-inferiority, and the constraints set alone misses
    the margin.
  - The pilot is reproduced: constraints were 0.756 vs 0.805 with `deepseek-chat`, and are now
    0.764 vs 0.821.
- **Against embeddings:** zero-shot Jev beats nearest-description embeddings by 47 points. It
  also beats the supervised embedding kNN that sees the other rows' labels.
- **Cost:** Jev is 1.3x cheaper, not dramatically. DeepSeek's shared-prefix cache bills the long
  class list at the cache-hit price; without caching it would be about $0.059 per 1k. Jev is 4x
  faster at p50.
- **Gate, headline (held-out θ).** Jev acts if p_top >= θ, otherwise DeepSeek. θ is chosen by
  5-fold cross-validation over 200 seeded random partitions: each fold uses the grid threshold
  with the best cascade accuracy on the other four folds (ties go to the lower θ).
  - Pooled cascade accuracy is **0.844** (2.5–97.5 % over partitions: 0.834–0.850). That is
    **+0.004** vs DeepSeek alone (0.840) and +0.026 vs Jev alone (0.818).
  - Mean Jev coverage is 0.71, at $0.039 per 1k. θ = 0.9 was chosen in 850 of 1000 folds and
    0.8 in 144.
  - Per set: constraints 0.804 (−0.017 vs DeepSeek), variables 0.977 (+0.019), parameters 0.796
    (−0.017), objectives 0.920 (+0.080; θ = 0.5 in every fold, which means Jev alone).
- **Full threshold curve** (pooled, in-sample, so each row is scored on the items its θ is read
  from):

  | θ | Jev coverage | Jev acc (kept) | cascade | random esc. (95 %) | oracle esc. | cascade − DeepSeek (95 % CI) | USD / 1k |
  |---|---|---|---|---|---|---|---|
  | 0.50 | 0.962 | 0.834 | 0.824 | 0.819 (0.812–0.824) | 0.856 | −0.016 (−0.047 to +0.016) | 0.030 |
  | 0.60 | 0.909 | 0.852 | 0.824 | 0.820 (0.812–0.831) | 0.881 | −0.016 (−0.044 to +0.013) | 0.032 |
  | 0.70 | 0.868 | 0.866 | 0.828 | 0.821 (0.809–0.834) | 0.881 | −0.013 (−0.038 to +0.013) | 0.033 |
  | 0.80 | 0.799 | 0.882 | 0.843 | 0.823 (0.809–0.837) | 0.881 | +0.003 (−0.019 to +0.025) | 0.036 |
  | 0.90 | 0.690 | 0.918 | 0.850 | 0.825 (0.809–0.840) | 0.881 | +0.009 (−0.003 to +0.025) | 0.040 |
  | 0.95 | 0.605 | 0.922 | 0.843 | 0.827 (0.809–0.843) | 0.881 | +0.003 (+0.000 to +0.009) | 0.043 |

  - θ = 0.9 (cascade 0.850, +0.009 vs DeepSeek) is only the **in-sample best**, not a held-out
    estimate.
  - The gate beats random escalation at the same rate, unlike the reranking study. Held out,
    though, it beats DeepSeek alone by only +0.4 pts, which is well inside the noise.
  - At about $0.039 per 1k it is not cheaper than DeepSeek alone ($0.037), because every item
    pays for Jev.
- **Jev p_top calibration** (pooled): 0.90–1.00 → 0.92 accuracy (n = 220); 0.80–0.90 → 0.66;
  0.60–0.80 → 0.63; 0.40–0.60 → 0.52.
- **Verdict:**
  - "Within 5 points" holds on the point estimate. The CI does not establish it, and it fails on
    constraints.
  - "Far better than embeddings" holds. "Cheaper" holds, weakly.
  - "The gate beats both": with a held-out θ it beats Jev by +2.6 pts. It beats DeepSeek only
    nominally (+0.4 pts) and is not cheaper. On constraints and parameters it is below DeepSeek.

**P2-H2b, routing half: requirement to constraint families (synthetic, publishable).** 200 held-out test
statements and 60 dev statements (seed 20260928). Labels are the 11 families of lp2graph
`ConstraintDomainClass` without `unclassified`, read from the enum at runtime.

- **Composition:** 89 / 75 / 36 statements carry 1 / 2 / 3 families. 117 contain negation:
  - 81 prohibitions that still need their family, e.g. "No overtaking at X" → precedence_ordering.
  - 72 waivers that must not trigger theirs, e.g. "Headways do not need to be modelled".
  - 120 statements contain distractor sentences with irrelevant numbers.
- **Held-out split:** test statements use only templates that the descriptions never saw. A test
  checks that no 5-gram of a test template appears in any description.
- **Dev-only selection:** for each router, the dev split chooses the description style (modelling
  vs requirement cue), the threshold and top-k.

| arm (dev-selected) | micro-F1 | P / R | exact set | prohibition recall | waiver FP | p50 ms | USD / 1k |
|---|---|---|---|---|---|---|---|
| DeepSeek JSON (modelling) | 0.983 | 0.975 / 0.991 | 0.955 | 1.000 | 0.042 | 1016 | 0.040 |
| Jev Nouls, per-family thresholds (modelling) | 0.718 | 0.644 / 0.813 | 0.265 | 0.914 | 0.028 | 246 | 0.086 |
| Jev Nouls @0.5 (pre-registered threshold) | 0.546 | 0.381 / 0.965 | 0.060 | 0.988 | 0.208 | 246 | 0.086 |
| keyword | 0.695 | 0.617 / 0.795 | 0.290 | 0.642 | 0.653 | – | 0 |
| TF-IDF @0.12, >=1 | 0.490 | 0.401 / 0.631 | 0.080 | 0.630 | 0.639 | – | 0 |
| Qwen3 embedding @0.55, >=1 (requirement) | 0.432 | 0.379 / 0.501 | 0.060 | 0.420 | 0.264 | 11 | 0 |

- **Jev minus DeepSeek, micro-F1:** −0.264 (CI −0.294 to −0.237) for Jev with per-family
  thresholds, and −0.437 for Jev@0.5.
- **Negation is not Jev's problem.** Mean Noul p is 0.83 for plain gold families, 0.90 for
  prohibitions, 0.29 for waived families and 0.35 for absent ones.
- **Precision on absent families is the problem.** Jev over-fires on abstract modelling families.
  Per-family F1: variable_bound_fix 0.48, coupling_linking_definition 0.53,
  precedence_ordering 0.59, capacity_resource 0.62. Concrete families score higher: subtour 0.91,
  headway 0.89, periodic 0.88, objective 0.88.
- **Styles:** the literal "Does `statement` ask for ...?" style (Jev docs advice) was worse on dev
  and on test (0.634).
- **Cost:** in this mode Jev costs 2x DeepSeek per statement. 11 Nouls use about 2k input tokens,
  while DeepSeek's system prompt is billed at the cache-hit price.
- **Verdict on the routing step:** the LLM routes, and Jev does not. Jev's usable strength is
  negation-robust Nouls: its waiver FP rate is 0.03, against 0.65 for keywords and 0.26 for
  embeddings. That suggests a per-family *verifier or filter* after an LLM proposal, not a router.
  The P2-H2b claim about fewer generation attempts and tokens was **not** tested for family
  routing here, only the retrieval-routing step. The relevance-filtering half of P2-H2b tests it
  for sentence filtering (`NOTES_filtering.md`). P2-H2c is not yet tested.

**Caveats.**

- **(a) H2a circularity.**
  - The `ml_class` taxonomy was iterated by a TF-IDF/SVM loop on these very rows until it was
    learnable from their descriptions, so every arm gets a favourable label space.
  - The rows and labels come from LLM-assisted extraction.
  - The class descriptions (now in the private `D4R_CLASS_DESCRIPTIONS` file) were written from
    class names and textbook OR knowledge. They were written by the same assistant family that wrote the pilot, which had
    seen rows.
  - Absolute accuracies are therefore optimistic. The arm *comparison* is fair because every arm
    sees identical descriptions.
- **(b) Privacy of the H2a label space.**
  - The class names, the descriptions and the id -> name map live only in
    `~/.local/share/d4r/` (see the post-verification entry above). Public results use opaque
    ids, and the H2a caches, which hold class labels, are gitignored.
  - No row text, paper key or canonical name is written anywhere. This was checked by grepping
    691 catalogue snippets against the caches, results and code, with 0 hits.
  - The H2a cache records hold only request hashes plus labels, probabilities, usage and
    similarity rows.
- **(c) H2b authorship.** Templates, gold sets and keyword lists were written by an LLM (Claude),
  so an LLM reader may be favoured. The keyword lists were written with the templates in view and
  are an optimistic lexical baseline. Statements are short and clean.
- **(d) When the style variant was added.** The requirement-cue style and per-family thresholds were
  added after a smoke run on 3 dev and 3 test statements showed Jev Nouls over-firing. The
  selection itself uses dev only, and all variants are reported.
- **(e) Latency** is client wall-clock time with 6 concurrent workers, identical for all arms.
  Embedding latencies are batch-averaged: the wall time of a batch of up to 16 texts is divided by
  its size, and embedding the descriptions is excluded. So they are not per-call like Jev's and
  DeepSeek's.
- **(f) Duplicates.** Two parameter rows have identical text, so the cache served them once.
