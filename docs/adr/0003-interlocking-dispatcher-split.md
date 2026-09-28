# 3. DLA as interlocking, engines as dispatcher

- Status: accepted
- Date: 2026-09-28
- Deciders: Jörn Maurischat

## Context and Problem Statement

If an engine controls raw Flatland actions (left/forward/right/stop per cell), a contest measures
path finding and collision avoidance, which LLMs are known to be bad at and which is not what a
railway dispatcher does. It also makes the arms incomparable: whoever handles safety best wins.

## Decision Drivers

- Compare engines on dispatching judgement, not on safety or path finding.
- Keep safety in code (railway practice: the interlocking is never delegated).
- A strong, simple, reproducible default so that "no intervention" is a meaningful baseline.

## Considered Options

1. Engines emit raw actions every step.
2. A greedy shortest-path follower plus engine overrides at triggers (the research census design).
3. The deadlock-avoidance heuristic (DLA) from flatland-baselines as interlocking plus engine
   choices among safe options at decision points.

## Decision Outcome

Option 3. DLA (vendored, MIT) is the strongest simple baseline on Flatland 4.2.x (0 deadlocks in
120 census episodes; about 0.6–0.7 arrival share). Engines decide at `depart` and `meet` points
among options DLA has already judged safe; the default option equals DLA's own action.

### Consequences

- Good: every arm starts from the same safe substrate; `dla-default` is the plain baseline.
- Good: decisions are few (about 5–20 per episode), so paid engines stay affordable.
- Bad: the decision space is narrower than full dispatching (no reroute yet); improvements are
  measured relative to DLA.
- Bad: DLA upstream targets flatland-rl 4.3.0; we run it on 4.2.6 (the ECML 2026 version).
