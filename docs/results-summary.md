# Results summary: pre-registered hypotheses and verdicts

Status: 2026-09-29, after runs `pilot1`, `pilot2`, `m1`, `lat-tau{10,3,1}`, `retest`, `h2`.
Numbers are regenerated from `runs/` (see `results/<tag>/` and `d4r paper-tables`). "Exploratory"
marks analyses that were not pre-registered.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **P2-H1** fit: typed model not worse than a generative LLM on the same card, faster and cheaper | **supported** | A–C bench (273 cards): Jev regret 15.0 vs DeepSeek fast 17.4, reasoning 18.6; 0.25 s vs 1.0 s / 10 s; $0.035 vs $0.05 / $1.83 per 1k decisions. Closed loop B: Jev +1.7 (n.s.), DeepSeek fast −1.9 (worse, p = 0.011). Differences at decision level are not significant (paired bootstrap). |
| P2-H1 (limit) | **qualified** | Mixed speeds (M): computing arms far better (LLM→MILP 6.1, reasoning 8.7 vs Jev 13.7). |
| **P2-H1b** speed in railway time | **supported, narrow** | Reasoning LLM arrival share 0.84 (paused) → 0.82 / 0.745 / 0.665 at τ = 10 / 3 / 1 s per step; Jev 0.84 at every τ. At main-line τ (tens of seconds) latency is irrelevant. |
| **P2-H3** calibration | **supported** | ECE Jev 0.12 vs 0.44–0.53 for DeepSeek (near one-hot log-probabilities; reasoning has none). Jev `confidence` = linear rescaling of p_top. |
| **P2-H4** confidence-gated escalation | **mixed** | Family B (θ fitted on A): 10.8 vs random escalation 11.9 at 38 % escalation. Family M (exploratory): no better than random. Cost-sensitive threshold (θ from A's cost ratio): good on B (11.5), fails on C (24.5). |
| **P2-H5** test–retest | **partly supported** | Pairwise flip rate: Jev 6 %, reasoning 27 %, but DeepSeek fast at temperature 0 only 1 %. |
| **P2-H6** serialisation (numbers) | **supported for Jev** | Raw integers: Jev 15.0 → 16.8 (worse); DeepSeek fast 17.4 → 16.1 (better). Order/ID permutations not yet run. |
| **P2-H7** joint coordination | **not testable in this design** | No same-step meeting pairs occurred: the interlocking plus the hold-safety rule prevent jointly inconsistent per-train questions. |
| **P2-H9a** rule adherence rises with guidance | **supported** | Adherence on 86 decisive M cards: Jev 0.85 → 0.92, DeepSeek fast 0.88 → 0.94, reasoning 0.79 → 0.91. |
| **P2-H9b** price of the rules | **supported (stronger than expected)** | Closed loop M: rule-ril420 arrival 0.788 vs 0.81 default, 0 wins / 19 losses; worse also under the priority-weighted objective and for Express passenger trains. |
| **P2-H9c** judgement within rules | **supported** | jev-ril regret 13.2 ≤ rule-ril420 13.6. But rule guidance hurts the reasoning LLM (8.7 → 18.2). |
| **P2-H2a** closed-taxonomy routing | **supported (near-LLM)** | Jev 0.818 vs DeepSeek 0.840 (CI −0.056 … +0.013); embeddings 0.345; CV gate 0.844. |
| **P2-H2b** routing + filtering as retrieval driver | **not supported for routing; filtering neutral** | Requirement → family: DeepSeek F1 0.983 vs Jev 0.718 (Jev good on negation). Filtering: Jev best filter (recall 0.997) but no change in generator success (19/20). |
| P2-H2c pre-solve semantic gate | not run | — |

## Cross-cutting findings

- **The card is the bottleneck.** A spread-weighted logistic regression trained on 828 labelled
  cards reaches test regret 17.5–18.5, no better than the interlocking alone (17.4).
- **Decision-level regret does not transfer to closed loop.** Only the rollout oracle improves on
  the interlocking reliably (B: +11.6, 12/38/0; M: +16.1, 10/40/0). Every engine that intervenes
  more often loses in the typical changed episode (negative median over changed seeds), even where
  rare large wins make its mean positive (LLM→MILP on M: mean +6.7, median −5.0, 7/16/27).
- **Scenario families differ.** Holding pays off in small networks (A) and hurts in large ones (C);
  computing arms win on A and on mixed speeds (M) and lose on C.
- **Cost.** All experiments ≈ US$ 8 (DeepSeek 7.9, Jev ≈ 0.2 for > 3 400 calls).
