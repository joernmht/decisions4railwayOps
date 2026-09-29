# Hypotheses (pre-registration draft)

Status: **draft v0.1, 2026-09-28.** Frozen before the first full run; later changes are logged at
the bottom with a date and a reason. Hypothesis ids carry a `P2-` prefix because they belong to
Paper 2 of the LP2Graph trilogy (Paper 3 uses H1–H3 for different claims).

## Background claim

Railway dispatching decisions are **fast** (solutions "within minutes", Cacchiani et al. 2014) and
**formal** (a small, closed action set: retime, reorder, reroute, hold a connection, cancel), and in
practice they are still largely made by experienced people choosing among a few options. That is the
profile of a *typed-output decision model*: TypeSafe's Jev answers a closed question (yes/no, one of
N options, an ordinal level) with a probability distribution instead of free text. The literature
also gives the counter-argument: underneath, dispatching is job-shop scheduling with blocking, rule
dispatching loses 25–40 % delay against branch and bound (D'Ariano et al. 2008), and Jev's own
documentation says it is weak at numbers, counting, time comparison and multi-hop reasoning.

We therefore test a sharpened version: **a typed decision model is a good fit for the *selection*
layer of dispatching (choose among precomputed, safe options), not for the *computation* layer
(conflict detection, running times, paths), which stays in code.**

## H1 cluster — typed decisions in dispatching (Flatland contest)

**P2-H1 (fit).** At the decision points of a Flatland dispatching simulation, Jev choosing among
code-computed safe options reaches episode outcomes (share of trains arrived, total arrival delay,
deadlocks) that are **not worse than a generative LLM** (DeepSeek, non-thinking and thinking) given
the identical decision card, at **lower latency and lower cost per decision**.
Non-inferiority margin: 3 percentage points of arrival share (paired over seeds).

**P2-H1b ("fast" in railway time).** With a latency-charged clock (a queried train waits
`ceil(latency / τ)` steps before its decision takes effect) and a sweep of τ, there is a crossover
τ* below which the fast typed model beats the slow arms (thinking LLM, LLM + LP execution) on
outcomes even where the slow arms win on a paused clock.

**P2-H3 (calibration).** Jev's option probabilities are better calibrated against rollout-labelled
best options (Brier score, ECE, reliability diagram) than DeepSeek's token log-probabilities
(non-thinking), its sample frequencies (thinking) and its verbalised confidence.
Note: Jev's `confidence` is a linear rescaling of the top probability for a given number of options
(our fit: `(p_max − 1/n)/(1 − 1/n)`); we therefore analyse the distribution and `p_top`, not
`confidence` as a separate signal.

**P2-H4 (confidence-gated escalation).** "Jev acts when `p_top ≥ θ`, otherwise escalate to the
slow arm" lies on or above the quality–cost–latency Pareto front of all single arms.
Controls: **random escalation at the same rate** (an independent reranking study found
uncertainty-based escalation no better than random) and oracle escalation.

**P2-H5 (test–retest consistency).** Repeating the identical decision card k = 5 times, the top
option flips less often for Jev than for DeepSeek (non-thinking at temperature 0 and thinking).
Relevant because a dispatcher expects the same answer to the same situation.

**P2-H6 (robustness to state serialisation).** Meaning-preserving changes to the decision card
(option order, train-id renaming, bucketed vs raw numbers, JSON vs prose) change Jev's decision less
than DeepSeek's. The raw-numbers variant doubles as the numeracy ablation for Jev's documented
weakness.

**P2-H7 (joint coordination).** When several trains are asked in the same step, independently
answered per-train questions produce jointly inconsistent decisions (e.g. both trains of a meeting
pair told to proceed) at a measurable rate for Jev; one joint question per conflict group removes
it. DeepSeek answering all trains in one call coordinates natively.

## H2 cluster — typed decisions in an MILP-modelling assistant

**P2-H2a (query routing onto a closed label space).** A Jev Choice over closed optimisation-model
facets (constraint family, variable role, model type) is within 5 points of an LLM router, far
better than embedding similarity to class descriptions, and cheaper; gating on its probability and
escalating the rest to the LLM beats both. (Pilot on the literature catalogue: Jev 0.756 vs DeepSeek
0.805 accuracy on 123 constraints, gate 0.821.)

**P2-H2b (driver of retrieval-augmented generation).** Using Jev as a reranker/filter/router in
front of the MILP generator — selecting constraint-family templates for a raw disruption description
and filtering irrelevant sentences — reduces generation attempts and tokens without reducing solve
success. We test the vendor's "replace or supplement embeddings in RAG pipelines" claim only in this
bounded sense: Jev drives retrieval; it does not replace generation, and its "alternative" mode is
limited to closed candidate sets (≤ 255 options per Choice, ≤ 32k tokens of state).

**P2-H2c (semantic pre-solve gate).** Per-concept Nouls over a generated formulation predict solve
failure better than the keyword-overlap gate. Jev stays a *filter*: deterministic structural
validation (LP2Graph) remains the validator.

## H9 — codified dispatching rules (mixed traffic, scenario M)

Registered 2026-09-29, before any scenario-M engine run. Rules: the paraphrased, cited
DB InfraGO Ril 420.0201 priority order (`src/d4r/rules/ril420.json`, P1–P6), applied to a
mixed-traffic Flatland family in which trains carry service classes consistent with their speed.

**P2-H9a (adherence).** On *decisive* cards (the rules order the train against an oncoming train),
rule-guided engines (`jev-ril`, `deepseek-fast-ril`, `deepseek-think-ril`) choose the
Ril-conformant option more often than their unguided counterparts, which see the service classes
but not the rules.

**P2-H9b (price of the rules).** Under the directive's overarching objective O0 (unweighted delay of
all trains), the code rule engine `rule-ril420` has positive regret against the rollout oracle;
under a priority-weighted delay (weights `SERVICE_WEIGHT`, our assumption) its regret is smaller
relative to the other engines. I.e. the rules trade total delay for the delay of high-priority
trains.

**P2-H9c (judgement within rules).** The guided typed model (`jev-ril`) does not have higher O0
regret than `rule-ril420`: rules that apply "in principle" leave room for justified exceptions,
which a judgement model can use.

Protocol: game set M seeds 1–50 (bench, consequential under either objective); closed loop on
fresh seeds 151–200; any fitted component (the learned baseline) uses seeds 51–150 only.

## What is *not* claimed

- No human baseline (that is Paper 3). All comparisons are machine vs machine, in simulation.
- No claim that Jev can plan or compute; code computes, Jev selects.
- Flatland is a grid abstraction of railway operations; external validity is discussed, and DISPLIB
  is the planned real-data follow-up.

## Change log

- 2026-09-29 — added P2-H9 (Ril 420 rules, scenario M) before any scenario-M engine run.

- 2026-09-28 — v0.1 drafted from the research sweep (docs, literature, endpoints, Flatland census).
