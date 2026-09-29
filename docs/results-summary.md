# Results summary: pre-registered hypotheses and verdicts

Status: 2026-09-29, after runs `pilot1`, `pilot2`, `m1`, `lat-tau{10,3,1}`, `retest`, `h2`, `big`, `fresh`.
**Revised after an adversarial claim verification (148 claims re-derived from `runs/`; see the lab
notebook).** Earlier verdicts that did not survive are corrected below.
Numbers are regenerated from `runs/` (see `results/<tag>/` and `d4r paper-tables`). "Exploratory"
marks analyses that were not pre-registered.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **P2-H1** fit (registered: closed-loop arrival share, non-inferiority margin 3 pp vs generative LLM) | **partly supported** | Non-inferior to DeepSeek fast on B (+0.8 pp, 95 % CI −1.4 … +3.0); inconclusive vs reasoning on B (CI −3.8 … +3.6) and vs fast on M (−4.6 … 0.0); **worse** than reasoning on M (−2.8 pp, CI −6.2 … −0.2). Decision level: lowest pooled mean regret (15.0; replicated 15.0 on 828 cards of seeds 51–200) but not significantly different from the default; 0.25 s, $0.035 per 1k decisions. |
| **P2-H1b** speed in railway time | **cost of latency shown, crossover not** | Reasoning LLM already loses at τ = 10 s (0.84 → 0.82 arrival, p = 0.004) and falls to 0.745 / 0.665 at τ = 3 / 1 s; τ > 10 s not tested; Jev never delayed (latency < τ). Slow arms were not better on the paused clock, so no crossover. |
| **P2-H3** calibration | **average only** | Jev ECE 0.12 (LLMs: one-hot, ECE = 1 − agreement), but p_top does not separate right from wrong (AUROC 0.51 on A–C, 0.67 on M); top bins over-confident; random has lower ECE. |
| **P2-H4** confidence-gated escalation | **not supported** | B gain in the first (uncached) run not significant and not reproduced in the cached rerun (12.4 vs Jev 11.6 vs random 12.5); worse than random on A and M; closed loop B significantly worse than the default. |
| **P2-H5** test–retest | **partly supported** | Pairwise flip rate Jev 6 %, reasoning 27 %, DeepSeek fast (T = 0) 1 %. |
| **P2-H6** numbers vs buckets | **not significant** | Jev raw +1.8 (CI −2.6 … +6.2); DeepSeek fast raw −1.3 (CI −5.0 … +2.3). |
| **P2-H7** joint coordination | **not testable in this design** | No same-step meeting pairs occur. |
| **P2-H9a** rule adherence rises with guidance | **supported** | Jev 0.85 → 0.92, DeepSeek fast 0.88 → 0.94, reasoning 0.79 → 0.91 (86 decisive M cards). |
| **P2-H9b** price of the rules | **supported** | Closed loop M: rule-ril420 arrival 0.788 vs 0.81, 0 wins / 19 losses; loss from fewer arrivals, significant for 3 of 5 classes; also under the weighted objective. |
| **P2-H9c** judgement within rules | **supported (bench)** | jev-ril 13.2 ≤ rule-ril420 13.6. Rule guidance raises the reasoning LLM's regret 8.7 → 18.2, not significantly. |
| M "computing arms win" (exploratory) | **not robust** | Means LLM→MILP 6.1 vs Jev 13.7, but paired Δ −7.6 (CI −19.6 … +3.2), driven by one deadlock seed (M13); without it Jev 3.4 vs LLM→MILP 6.6. |
| Learned reference (exploratory) | **learnable signal, not significant** | Seed-grouped CV on seeds 51–200 selects no regularization; test regret 13.3 (lowest), vs Jev −1.7 (CI −6.9 … +3.0), vs default −4.0 (CI −9.7 … +1.0). |
| **P2-H2a** closed-taxonomy routing | **near, not shown non-inferior** | Jev 0.818 vs DeepSeek 0.840 (CI −0.056 … +0.013 breaches the 5-point margin); embeddings 0.345; CV gate 0.844. |
| **P2-H2b** routing + filtering | **not supported for routing; filtering neutral** | Requirement → family: DeepSeek F1 0.983 vs Jev 0.718; filtering: generator success 19/20 unfiltered vs 18/20 with the dev-selected Jev filter (p = 1). |
| P2-H2c pre-solve semantic gate | not run | — |

## Cross-cutting findings

- **Heavy-tailed stakes dominate every comparison.** Most consequential decisions are worth a few
  reward units, a few are worth hundreds (deadlock episodes), and single seeds reverse rankings.
  With 273 benchmark cards no zero-shot engine is significantly better than the default.
- **Decision-level regret does not transfer to closed loop.** Only the rollout oracle improves on
  the interlocking reliably (B: +11.6, 12/38/0; M: +16.1, 10/40/0); other engines lose in the
  typical changed episode even where rare large wins keep their mean positive.
- **The interlocking (DLA) is not deadlock-free:** 10 game-set episodes and 9 of 1 700 closed-loop
  episodes ended with 2–3 deadlocked trains.
- **Reproducibility caveats:** pilot1 answers were lost to a caching bug; engines are not
  deterministic (pilot1 vs pilot2 on identical cards: Jev 94 %, reasoning 77 %, LLM→MILP 62 % same
  choices); DeepSeek is reachable only through the moving alias `deepseek-flash`.
- **Cost.** All experiments ≈ US$ 10 (DeepSeek ≈ 9.8, Jev ≈ 0.2).
