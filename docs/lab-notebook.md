# Lab notebook

Dated observations, newest first. One entry per session of work: what was run, what was seen, what
it means, what is next. Numbers come from committed run records under `runs/`.

## 2026-09-28 (evening) — pilot1: game-set bench (A+B) and closed loop (B, 20 seeds)

**Ran.** Game sets, 50 seeds each: A 211 cards (53 consequential), B 372 (117), C 360 (103).
Bench `pilot1` on the 170 consequential A+B cards; closed loop on B seeds 1–20. Tables:
`results/pilot1/`.

**Decision level (170 consequential cards; regret in reward units ≈ delay steps).**

| engine | agree | mean regret | non-default | Brier | ECE | p50 latency | $/decision |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-think | 0.49 | 11.9 | 0.69 | 1.03 | 0.52 | 8.6 s (p95 46 s) | 0.0017 |
| fixed-milp | 0.63 | 12.8 | 0.52 | – | – | 2 ms | 0 |
| jev | 0.65 | 13.5 | 0.16 | **0.47** | **0.07** | 256 ms | 0.000035 |
| rule-slack | 0.59 | 14.1 | 0.25 | – | – | 0 | 0 |
| dla-default | 0.71 | 15.6 | 0.00 | – | – | 0 | 0 |
| random | 0.48 | 16.3 | 0.54 | – | – | 0 | 0 |
| deepseek-fast | 0.49 | 18.1 | 0.46 | 1.02 | 0.51 | 988 ms | 0.000076 |

(deepseek-think: 162 of 170 cards at the time of the table.)

**Closed loop (B, seeds 1–20).** Arrival share: oracle 0.785 (+21.1 reward vs DLA, 8/12/0,
Wilcoxon p = 0.008), jev 0.760 (+10.1, p = 0.56), fixed-milp 0.755 (+14.4), deepseek-fast 0.735
(+6.6), rule-slack 0.730 (+3.5), dla-default 0.715, random 0.720 (−3.5). Zero deadlocks for all.
20 seeds are not enough to separate the engines; only the oracle is significant.

**What it means (provisional).**
1. **Stakes are asymmetric.** When HOLD/WAIT is best it is worth ~50–90 reward units; a needless
   hold costs ~15. Agreement with the oracle therefore rewards the conservative default, while
   regret rewards engines that hold more (deepseek-think holds 69 % of the time and has the lowest
   regret; the default has the highest agreement and a high regret).
2. **Jev is the only well-calibrated engine** (ECE 0.07 vs ~0.5 for DeepSeek, whose log-probabilities
   are near one-hot and whose thinking mode gives no probabilities). Its distributions are close to
   uniform (p_top ≈ 0.55–0.6), i.e. it is honestly uncertain.
3. **Calibration makes a cost-sensitive policy possible.** Choosing the non-default option whenever
   Jev's probability for it exceeds θ ≈ 0.46 cuts Jev's regret from 13.5 to ~11.1–11.3 (in-sample!),
   matching deepseek-think at ~1/35 of the latency and ~1/50 of the cost. θ differs between A (~0.40)
   and B (~0.46), so this must be tested with θ fixed on a development split (pre-registered below).
4. DeepSeek-fast is worse than random on regret; its fast answers do not use the card well.

**Bug found and fixed.** `ResponseCache` is falsy when empty (`__len__`), so `cache or
ResponseCache(None)` silently dropped the file-backed cache: pilot1 API answers were not written to
`runs/cache/` (bench/contest records still hold every decision). Fixed with a regression test.

**Pre-registered for the next run (pilot2).** (a) θ for "Jev + cost-sensitive threshold" is fixed on
scenario A game-set cards and tested on B and C (and on fresh seeds 51–100); (b) closed loop on B with
50 seeds for all engines incl. deepseek-think and the LP arm; (c) gating curves Jev → deepseek-think /
LP arm with random-escalation control; (d) test–retest k = 5 on 60 stratified cards.

## 2026-09-28 — first working harness, first game sets

**Built.** DLA interlocking (vendored), decision cards, dispatch controller with `depart` and
`meet` decision points, rollout oracle, engines (Jev, DeepSeek fast/thinking, DeepSeek → MILP →
HiGHS, fixed MILP, slack-priority rule, random, DLA default), response cache, game-set builder.
Offline test suite (10 tests) green.

**Observed.**
- DLA-default over 20 seeds: mean arrival share A 0.67, B 0.72, C 0.73, zero deadlocks.
- Game sets (20 seeds each): A 99 cards (21 consequential), B 177 (52), C 144 (38). About a quarter
  of the decisions change the outcome at all; among those, the non-default option (HOLD/WAIT) is
  best in about a quarter of cases. So **regret**, not raw accuracy, is the headline metric (an
  always-default engine already agrees with the oracle on ~75 % of consequential cards).
- Stakes are very uneven: in B/seed 4 two oracle HOLD decisions lift the arrival share from 0.2 to
  1.0 (reward −401 → −26). Most decisions are worth 0–5 reward units, a few are worth hundreds.
- Live smoke test on three B cards: Jev 236–367 ms and ~$0.00004 per decision with near-uniform
  distributions (p_top 0.56–0.61); DeepSeek-flash non-thinking 0.7–1.7 s, log-probabilities put
  ~1.0 on its answer; thinking mode 4.7–18.8 s and once ran out of its 4000-token budget with an
  empty answer (budget raised to 16000).
- Meet decisions were first asked as soon as an oncoming train appeared anywhere on the route;
  now only once the nearest oncoming train is within 15 cells (holding for a far-away train is
  never sensible).

**Next.** Bench runner over the game sets for all engines; closed-loop contest on B; analysis
(regret, calibration, latency/cost frontier).

## 2026-09-28 — lab set up, Paper 2 reframed

- Paper 2 of the LP2Graph trilogy is reframed around **typed decision models for railway operations**.
  This repository is its public laboratory.
- Toolchain: Python 3.12, flatland-rl 4.2.6, typesafe-sdk 0.7.2 (Jev), OpenAI-compatible client for
  DeepSeek, PuLP + HiGHS/CBC for LP execution, lp2graph 0.3.0 (PyPI) for structural checks.
- Smoke test (before this repository existed): a single-track conflict at Riesa gave
  Noul `conflict` = 0.95 and Choice `hold_passenger` (p = 0.79, confidence 0.69).
