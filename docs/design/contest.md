# Contest design: typed decisions in Flatland dispatching

Status: v0.1, 2026-09-28. Companion to [`../hypotheses.md`](../hypotheses.md).

## 1. Division of labour: interlocking vs dispatcher

A real control centre separates *safety* (the interlocking: no two trains may be granted
conflicting routes) from *dispatching* (which train goes first, when a train departs, which route
it takes). The contest mirrors this split so that every engine is compared on the same thing:

| Layer | Who | What |
|---|---|---|
| Interlocking | code: the deadlock-avoidance heuristic (DLA) from flatland-baselines, vendored under `src/d4r/sim/_vendor/dla` (MIT) | moves every train along its route whenever that is safe; 0 deadlocks in 120 census episodes |
| Dispatcher | the engine under test | chooses among **safe options** at **decision points** |

The default option at every decision point is what DLA alone would do, so the `dla-default` engine
*is* the plain DLA baseline, and every other engine is measured as a deviation from it.

## 2. Decision points and options

| Kind | Trigger (all computed in code) | Options (default first) | Railway meaning |
|---|---|---|---|
| `depart` | a train is ready to depart, DLA would release it, and oncoming trains are on its route | `DEPART_NOW`, `WAIT` (5 steps, then asked again, at most 3 asks) | retiming / release into the network |
| `meet` | a moving train that DLA lets proceed has oncoming trains on its route, the nearest within 15 cells, and it stands where they can pass it (hold-safety check) | `PROCEED`, `HOLD` (until the oncoming trains have passed, at most 12 steps) | reordering: choosing where trains cross on single track |

A situation is asked once per (train, set of oncoming trains). Holds are released early when the
awaited trains have passed or when holding would start to obstruct another train.
Route choice at facing switches (`route`) is designed but not yet enabled.

## 3. Decision cards

Every engine receives the same **decision card** (`d4r.dispatch.cards`): code-computed facts about
the train (status, schedule reserve = latest arrival − now − remaining running time, remaining
distance, trains queued behind it) and about each oncoming train (distance along the route, status,
reserve, remaining distance, breakdown time left, shared track ahead, trains queued behind it),
plus the options with plain-language descriptions.

- **Bucketed rendering (default).** Numbers become named ranges ("close (3-6 cells)",
  "tight (0-4 steps of reserve)"). Jev's documentation says it is unreliable with raw numbers,
  counting and time comparison, and the literature says the same of LLMs; all direct-choice engines
  get the same bucketed card.
- **Raw rendering.** Exact integers. The LP arms need coefficients, so they receive it; the direct
  engines are also run on it as an ablation (P2-H6).

## 4. Engines

| Engine id | What it is |
|---|---|
| `dla-default` | never intervenes (first come, first served, as the interlocking resolves it) |
| `random` | uniform over options, seeded by the card id |
| `rule-slack-priority` | the train with more schedule reserve yields (delay-based priority rule) |
| `fixed-milp` | code builds a small lateness MILP of the decision, HiGHS picks the option (OR baseline, no LLM) |
| `jev[jev-1.13.0]` | TypeSafe Jev, one Choice question per card; full distribution recorded |
| `deepseek-flash[fast]` | DeepSeek V4.1 Flash, non-thinking, JSON answer; option probabilities from token log-probabilities |
| `deepseek-flash[think]` | the same model in thinking mode (reasoning), one-hot probabilities |
| `deepseek-flash+lp[fast]` | DeepSeek writes a MILP as a JSON model spec (validated, never code), HiGHS solves it, the selector variable decides; up to 3 attempts with error feedback |
| `rollout-oracle` | simulates every option to the end of the episode and picks the best (compute-heavy upper reference) |

Every paid call is cached (`runs/cache/*.jsonl`) with the served model id, token usage, latency and
UTC time (DeepSeek prices depend on peak hours). Jev is pinned to `jev-1.13.0`.

## 5. Two levels of evaluation

1. **Game set (offline, decision level).** DLA episodes over fixed seeds; at every decision point
   the rollout oracle values each option (total Flatland reward from that step to the end, with DLA
   continuing). Each engine answers the same labelled cards. Metrics: agreement with the best option
   on *consequential* cards (options differ in value), **regret** (best value − chosen value),
   Brier score and expected calibration error of the option probabilities, test–retest flips,
   latency, tokens and cost.
2. **Closed loop (episode level).** Each engine dispatches whole episodes over the same seeds.
   Metrics: share of trains arrived, Flatland normalised reward (= delay-based), total arrival
   delay, deadlocked trains (own wait-for-cycle detector; Flatland 4.2.6's counter is broken),
   decisions and holds per episode, latency and cost. Paired per-seed comparison against
   `dla-default` (Wilcoxon signed-rank).

**Clock.** Default is a paused clock (decision quality only). P2-H1b adds a latency-charged clock:
a queried train waits `ceil(latency / τ)` steps before its decision takes effect, swept over τ.

## 6. Scenarios

| Preset | Map | Trains | Cities | Breakdowns |
|---|---|---|---|---|
| A (development) | 30×30 | 7 | 3 | interval 540 steps, 20–50 steps long |
| B (main) | 30×30 | 10 | 3 | same |
| C (stretch) | 40×40 | 10 | 4 | same |

First game-set census (20 seeds each, 2026-09-28): A 99 cards (21 consequential), B 177 (52),
C 144 (38); mean DLA arrival share 0.67 / 0.72 / 0.73. In one B episode two oracle HOLDs raise the
arrival share from 0.2 to 1.0, so single dispatching decisions can matter a great deal.

## 7. Known limitations

- Flatland is a grid abstraction (one cell ≈ one block section, one step ≈ the time to traverse it).
- The rollout label is one step of policy improvement over DLA, not a global optimum.
- Per-train questions are answered independently (joint coordination is P2-H7).
- DLA is vendored from a package that officially targets flatland-rl 4.3.0; we run it on 4.2.6.
