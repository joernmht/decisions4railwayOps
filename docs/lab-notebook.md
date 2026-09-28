# Lab notebook

Dated observations, newest first. One entry per session of work: what was run, what was seen, what
it means, what is next. Numbers come from committed run records under `runs/`.

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
