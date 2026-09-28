# H2 notes: Jev as a relevance filter in front of an MILP generator (P2-H2b, filtering half)

Hypothesis labels (`docs/hypotheses.md` v0.1): **P2-H2a** = closed-taxonomy routing
(`NOTES_routing.md`). **P2-H2b** = requirement -> constraint-family routing (`NOTES_routing.md`)
plus relevance filtering (this file); v0.1 describes it as "selecting constraint-family templates
... and filtering irrelevant sentences". **P2-H2c** (semantic pre-solve gate) is not yet tested.
The orchestrator first ran this experiment as "P2-H2c"; that label is retired.

## 2026-09-28: post-verification fixes (replay, threshold labels)

- **MILP cache compacted.** `runs/cache/h2-filtering-milp.jsonl` held 30 keys with two or three
  different answers each, left over from the cache race. It now keeps one record per key, the
  last one, because the cache replays last-write-wins: 164 lines became 128 records.
- **Offline replay reproduces the reported results.**
  - `run_filtering.py --offline` now replays every arm and the downstream step from the caches.
    A missing request raises `ReplayMiss` before any client is created.
  - It writes to `results/h2/offline/` (gitignored), so it no longer overwrites the reported
    files.
  - Checked: every metric equals `results/h2/filtering_summary.json`, and
    `filtering_per_statement.json` is byte-identical. Only the timestamp and the latencies
    measured live on replay differ: the regex arm, and the HiGHS solve time inside the downstream
    `latency_ms_mean`. 0 live calls.
  - `--render-only` re-renders the `.md` from the reported JSON without any calls.
- **θ = 0.3 is labelled post-hoc** in the results (`downstream_conditions`, table rows
  "jev@0.3 (post-hoc margin)") and below.

## 2026-09-28 — filter arm ("driver of RAG"): statements, six filter arms, downstream MILP

**Built.** `src/d4r/h2/statements.py`: seeded generator (seed 2026) of synthetic, publishable,
noisy statements for one problem family (single-track corridor, 2-4 trains, 2-3 sections, release,
running and due times, headway, priority weights, optional no-overtaking rule, train fault or
section blockage; objective: total weighted tardiness at the exit). Every sentence has a gold
needed/not-needed label and a kind; noise = numbered distractors (unit numbers, passengers,
tickets, years, platforms, staff numbers), chatter, near misses (other line, next week, yesterday,
a cancelled train, tomorrow's speed restriction, old headway, onward due time) and, in 31 of 60
statements, a value that a later "Correction:" sentence replaces (`superseded`, not needed). Each
statement carries its exact instance; `reference_optimum` brute-forces the train order on every
section and agrees with HiGHS on the code-built big-M MILP for all 60 instances.
`src/d4r/h2/filtering.py`: arms, metrics, generate-and-solve. `experiments/h2/run_filtering.py`
writes `results/h2/filtering_summary.{md,json}` and `filtering_per_statement.json`; every paid call
is cached in `runs/cache/h2-filtering-*.jsonl`, and `--offline` replays every arm from the caches
into `results/h2/offline/` (see the post-verification entry above).
Offline tests: `tests/test_h2_filtering.py` (11 tests).

**Corpus.** 60 statements (40 dev for the threshold, 20 test), 1321 sentences: 940 needed (700
data-bearing + 240 frame), 381 not needed, of which 351 contain a digit (the numeracy trap) and
117 are near misses.

**Filter arms, test split** (threshold = largest with recall 1 on dev):

| arm | thr. | recall needed | precision | trap kept | near-miss kept | superseded kept | chars cut | p50 ms | USD/stmt |
|---|---|---|---|---|---|---|---|---|---|
| Jev, Noul per sentence + criteria, 1 request | 0.5 | 0.997 (1 dropped) | 0.971 | 0.070 | 0.023 | 0.73 | 0.211 | 283 | 0.00021 |
| Jev, no criteria (ablation) | 0.3 | 1 | 0.956 | 0.108 | 0.116 | 0.82 | 0.199 | 250 | 0.00008 |
| DeepSeek-flash, non-thinking, JSON list | – | 1 | 0.956 | 0.108 | 0.047 | 1.00 | 0.199 | 820 | 0.00015 |
| Qwen3-Embedding-4B cosine (ScaDS) | 0.279 | 1 | 0.780 | 0.620 | 0.977 | 1.00 | 0.074 | 185 | 0 |
| regex (keep sentences with a digit) | – | 0.737 | 0.635 | 1 | 1 | 1 | 0.325 | 0 | 0 |
| keep-all | – | 1 | 0.685 | 1 | 1 | 1 | 0 | 0 | 0 |

- Jev sweep over all 60: recall of needed sentences stays 1.000 up to θ = 0.4 (precision 0.964,
  trap kept 0.100) and first drops at 0.5 (one sentence). The dev rule picked 0.5, and on test
  that one drop is a plain running-time sentence (noul 0.49): **the dev floor does not carry over
  exactly; use a margin** (θ = 0.3-0.4 kept all 940 needed sentences).
- **θ = 0.3 is a post-hoc safety margin, not a dev-selected threshold.**
  - On dev, recall is 1.0 for every θ up to 0.5, so dev alone cannot pick 0.3.
  - The "0.3–0.4 kept all 940" observation uses all 60 statements, test split included.
  - The `jev@0.3` downstream condition is therefore post-hoc. Checking a pre-registered rule
    ("dev floor − 0.1") needs a fresh test split.
- The criteria matter: without them the perfect-recall threshold falls to 0.3, and at 0.5 test
  recall is 0.977 (0.70 of statements complete). They also make Jev's request 2.6x larger (5.1k vs 1.9k input
  tokens per statement, because the criteria repeat for every sentence), so Jev+criteria costs
  about the same per statement as DeepSeek's list ($0.00021 vs $0.00015) while being ~3x faster.
- Jev is the only arm that drops superseded values by itself.
  - At θ = 0.5 it keeps 55 % of them over all 60 statements, against 90 % for the LLM list.
  - **On the test split**, Jev keeps 73 % (8 of 11) and DeepSeek 100 % (11 of 11).
- Embeddings measure topic, not relevance. At a safe threshold they cut only 7 % of the text and
  keep 62 % of numbered distractors and 98 % of near misses.
- The regex keeps every trap and loses all 240 frame sentences (objective, occupancy and
  overtaking have no digits).
- **Embedding latency is a batch figure.**
  - The p50 of 185 ms is one request that embeds all of a statement's sentences (about 22), so
    it is about 8 ms per sentence.
  - The once-per-run query embedding and the rate-limit pauses are excluded.
  - It is per statement like Jev's one request, but it comes from a different, rate-limited
    endpoint.
  - In the routing arms (`NOTES_routing.md`), embedding latency is batch-averaged as batch time
    divided by batch size. In neither case is it a per-call latency like Jev's or DeepSeek's.

**Downstream** (20 test statements; DeepSeek-flash writes a `MilpSpec` from a fixed template prompt,
HiGHS solves, up to 3 attempts with error feedback; correct = optimum equals the brute force):

| input text | thinking: correct | non-thinking: correct | prompt tokens (1st) |
|---|---|---|---|
| full statement | 19/20 | 7/20 | 1287 |
| Jev θ = 0.5 (dev floor) | 18/20 | 7/20 | 1167 (-9.3 %) |
| Jev θ = 0.3 (post-hoc margin, not dev-selected) | 19/20 | 7/20 | 1176 (-8.6 %) |
| gold labels | 19/20 | 5/20 | 1155 (-10.3 %) |
| DeepSeek list | 19/20 | 8/20 | 1176 (-8.6 %) |

- Thinking mode is optimal on 20/20 in every condition, with 1.0 attempts. Its three failure cases
  are all informative: (a) h2f-050, full text only: the near miss "Tomorrow RE 4967 will need 16
  minutes for Steinbrück–Wiesental" leaked into the model (every filter removed it); (b) h2f-043,
  Jev θ = 0.5 only: the dropped running-time sentence; (c) h2f-049, every filtered input incl. gold:
  removing the superseded "minute 7" sentence left the correction "(not 7)" without its antecedent,
  and both generators then treated the correction as replacing the train's fault sentence. So
  **sentence-level filtering can break cross-sentence references**, a failure mode the gold labels
  do not capture.
- Non-thinking generation fails on big-M bookkeeping (headway rows, sign errors) in ~2/3 of cases
  regardless of input, so it cannot show a filtering effect; McNemar p = 1.0 everywhere (n = 20).
- Cost view: filtering cuts the statement by ~20 % of characters, the prompt by only ~9 % (the
  template system prompt is fixed), and the thinking generator's cost is dominated by ~11k
  completion tokens per model, so the prompt saving is worth well under 1 % of generation cost
  (the condition-to-condition cost differences come from reasoning length, not from the input). For P2-H2b this means:
  solve success unchanged, attempts already at 1.0, token saving small. The filter's value here is
  robustness against poisoning near misses, not cost.

**Development and spend.** The first generator prompt (modelling approach in prose) gave 0/8
correct in a 2-statement pilot; the template prompt (`MILP_SYSTEM`, variable names given) gave 7/16
non-thinking and 10/10 thinking on dev statements 0-7 of an earlier generator version without
corrections (probe cached outside the repo). Spend for
this part: DeepSeek ≈ $0.75 (incl. pilot, prompt probes and a first run whose duplicate texts
raced on the cache), Jev ≈ $0.037, ScaDS embeddings about 130 calls, no charge. The calls behind
the reported numbers cost $0.50 (downstream thinking $0.37, non-thinking $0.10, filters $0.03).

**Bugs found on the way.** (1) `ResponseCache` defines `__len__`, so an *empty* cache is falsy and
`cache or ResponseCache(None)` silently replaces a new cache file by an in-memory cache (nothing is
ever written). Fixed in the filter classes (`is not None`); the engines under `src/d4r/engines/` no
longer use the pattern, but any new code should avoid it. (2) Two conditions that produce the same
text raced on one cache key under the thread pool, so a rerun replayed a different chain than the
first run reported; identical texts are now generated once per generator.

**Next.** Threshold rule with a margin (dev floor minus 0.1) and a check on a larger test split;
harder noise with anaphora (corrections, "the same as", "the other train") and a Noul asking
whether a sentence is superseded, keeping the correction's antecedent when it is; shorten Jev's
request by moving the criteria into structured instructions once and measuring the accuracy cost.
