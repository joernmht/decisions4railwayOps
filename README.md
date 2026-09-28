# decisions4railwayOps

**Typed decision models for railway operations — an open laboratory.**

Railway dispatching decisions are fast (solutions are expected within minutes) and formal (a small,
closed set of actions: hold, release, reorder, reroute). This lab tests whether that makes them a
good fit for *typed-output decision models* such as TypeSafe AI's **Jev**, which answer a closed
question with a probability distribution instead of free text, and compares them head-to-head with
generative LLMs (DeepSeek), LLMs that write and solve an optimisation model, classical OR and rule
baselines, in the [Flatland](https://flatland.aicrowd.com) railway simulator.

It is the experimental repository behind Paper 2 of the LP2Graph trilogy
(Maurischat & Bešinović, TU Dresden, Chair of Railway Operations).

## What is in here

```
src/d4r/
  sim/          Flatland scenarios, deadlock detection, vendored DLA interlocking (MIT)
  dispatch/     decision cards (the shared state every engine sees) and the dispatch controller
  engines/      contestants: Jev, DeepSeek (fast/thinking), DeepSeek -> MILP -> HiGHS,
                fixed MILP, slack-priority rule, random, DLA default, rollout oracle
  contest/      game-set builder (labelled decision cards) and runners
docs/
  hypotheses.md       pre-registered hypotheses (P2-H1 ... P2-H7, P2-H2a-c)
  design/contest.md   contest design: interlocking vs dispatcher, cards, engines, metrics
  adr/                architecture decision records
  lab-notebook.md     dated observations, newest first
runs/           committed run records (cards, answers, latency, tokens, cost) and API response caches
results/        tables and figures regenerated from runs/ by script
```

The key design choice: **code computes, the model selects.** A deadlock-avoidance interlocking
guarantees safety; engines only choose among safe options that code has prepared, from a decision
card with code-computed facts. Every engine sees the same card.

## Quick start

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                      # offline suite, no API keys needed

cp .env.example .env        # add TYPESAFE_API_KEY / DEEPSEEK_API_KEY for live engines
```

```python
from d4r.sim.scenario import PRESETS
from d4r.dispatch.controller import run_episode
from d4r.engines.rules import SlackPriorityEngine

result, decisions = run_episode(PRESETS["B"], seed=4, engine=SlackPriorityEngine())
print(result.to_dict())
```

## Reproducibility

- A run is fully determined by `(scenario preset, seed, engine)`; Flatland breakdowns depend only on
  the seeded environment, so all engines face the same disruptions.
- Paid API answers are cached with the served model id, usage, latency and time; re-running an
  experiment replays the cache. Jev is pinned to `jev-1.13.0`.
- Results in `results/` are regenerated from `runs/` by script, never edited by hand.

## Status

Early and moving fast; see [docs/lab-notebook.md](docs/lab-notebook.md). Nothing here is peer
reviewed yet.

## License and citation

Code: MIT (see [LICENSE](LICENSE)). The vendored DLA heuristic is MIT, © 2025 Flatland Association.
Please cite via [CITATION.cff](CITATION.cff).

## AI disclaimer

This repository was developed with substantial assistance from AI coding tools
(primarily Anthropic's Claude). Code, documentation and results have been
reviewed by the author, who takes full responsibility for the content.
