# H2 notes: independent verification of the routing and filtering runs

## 2026-09-28: verifier pass over P2-H2a/b (routing) and the filter arm

**What ran.** `ruff check src tests experiments` is clean, and `ruff format --check` passes on all
9 H2 files. The full offline suite passes (46 tests, 33 of them in `test_h2_*`). Every number
below was recomputed from `runs/cache/h2-*.jsonl` with my own code: request keys rebuilt,
metrics recounted and cached MILPs re-solved with HiGHS. The catalogue was read at runtime and
only counts were printed. No paid calls were made and no H2 file was edited.

**Headline numbers re-derived (all match the results files):**

- **P2-H2a, pooled, n = 319.**
  - Jev accuracy is 0.818 and DeepSeek 0.840. Cost is $0.0284 vs $0.0365 per 1k.
  - The gate at θ = 0.9 covers 0.690 of items with a cascade accuracy of 0.850.
- **P2-H2b, test, n = 200.**
  - DeepSeek micro-F1 is 0.983. Jev with per-family thresholds tuned on dev scores 0.718, and
    0.546 at 0.5.
  - Jev's waiver FP rate is 0.028 (n = 72). DeepSeek's prohibition recall is 1.000 (n = 81).
  - The dev and test statement texts are disjoint, and the dataset file equals the generator
    output.
- **Filter, test, n = 20.**
  - The dev-floor threshold is 0.5. There, Jev keeps needed sentences at 0.997 (1 dropped), with
    precision 0.971 and 0.070 of numbered distractors kept.
  - With thinking, the generator is correct on 19/20 statements from the full text and 18/20
    from the Jev-filtered text.
  - Brute force and HiGHS agree on all 60 instances.

**Confirmed issues (none invalidate the headline comparisons):**

1. **The H2a gate's θ = 0.9 was chosen on the same 319 items.** It is the in-sample best of 6
   thresholds, and none was pre-registered.
   - With 5-fold cross-validated θ selection (200 splits), the cascade scores 0.844 (range
     0.837–0.850), not 0.850.
   - Against DeepSeek that is about +0.004, down from +0.009.
   - Report the whole curve, or the cross-validated number.
2. **The `jev@0.3` downstream condition is not dev-selected.** On dev, recall is 1.0 up to 0.5.
   The "0.3–0.4 kept all 940" claim uses all 60 statements, test included.
   - Label it as a post-hoc margin, or pre-register "dev floor − 0.1" and check it on a fresh
     test split.
   - Its test filter row (0.956 precision) happens to equal the jev-bare and DeepSeek rows. That
     is genuine: the kept sets are identical on 12 of 20 statements.
3. **`h2-filtering-milp.jsonl` holds 30 keys with two different answers each,** left over from
   the race.
   - A replay uses the last answer. Using the first answer would move the non-thinking "gold"
     count from 5/20 to 6/20; no other count changes.
   - Compact the cache to one record per key before committing.
4. **`run_filtering.py --offline` overwrites the results.** It rewrites
   `results/h2/filtering_*.json/md` with only the regex and keep-all arms.
   - Point `--offline` at another output directory, or replay all arms from the cache.
5. **The private H2a class names would become public.** The 30 `ml_class` names appear in
   `routing.py`, `routing_summary.{md,json}`, `test_h2_routing.py` and both routing caches
   (`runs/cache/h2-routing-{jev,deepseek}.jsonl`).
   - They occur nowhere in any public repo, and none of these files is gitignored.
   - No row text leaked: 1089 snippets and all word 6-grams of the catalogue were checked, with
     0 hits.
   - Joern must approve this before commit. Otherwise, load the names and descriptions at runtime
     and write opaque class ids.
6. **Minor reporting slips.**
   - "Jev keeps 55 % of replaced values, DeepSeek 90 %" is over all 60 statements. On the test
     split it is 73 % vs 100 %.
   - Embedding latencies are batch-amortised. They are not per-call like Jev's and DeepSeek's.
   - Some H2b family descriptions paraphrase held-out test templates, e.g. the connection-wait,
     diversion-if-opened and count-of-late-trains examples. The 5-gram test does not catch
     paraphrases. Still, overall word overlap is lower for test templates than for dev (0.35 vs
     0.45), so this is not systematic leakage.

**No problems found in these checks:**

- No key patterns and no key values anywhere in `src/`, `tests/`, `experiments/`, `results/` or
  the H2 caches.
- No cache-key collisions across arms: engine ids differ, and the style is part of the payload.
- Every arm returned zero errors, so the fail-open path never fired.
- The cost formulas agree with list prices.
- Gold labels of the negated H2b templates and of the superseded and correction sentences are
  correct: every correction restates the full value.
