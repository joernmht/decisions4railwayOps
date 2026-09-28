# 4. One shared decision card per decision, bucketed by default

- Status: accepted
- Date: 2026-09-28
- Deciders: Jörn Maurischat

## Context and Problem Statement

A comparison of Jev with an LLM is confounded if they receive different inputs. Jev's own
documentation says it is unreliable with raw numbers, counting, time comparison and irrelevant
state, and suggests computing numbers in code.

## Decision Outcome

Code computes every fact (distances, schedule reserve, queued followers, breakdown time) and renders
one `DecisionCard` per decision. All direct-choice engines receive the same **bucketed** rendering
(named ranges instead of integers); the **raw** rendering exists for the ablation (P2-H6) and for
the LP arms, which need coefficients. Options are a closed list with plain-language descriptions;
engines can only pick one of them.

### Consequences

- Good: differences between arms are differences in judgement, not in information.
- Bad: bucketing throws information away; the raw ablation measures how much that costs each arm.
