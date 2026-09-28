# 5. The LLM + LP arm emits a JSON model spec, solved by HiGHS

- Status: accepted
- Date: 2026-09-28
- Deciders: Jörn Maurischat

## Context and Problem Statement

The "LLM with LP execution" contestant needs the LLM to formulate an optimisation model for each
decision. Executing model-authored code is a security and reproducibility problem (the predecessor
raiLPminer did this; raiLParchitect replaced it by structured formulations).

## Considered Options

1. LLM writes PuLP/Pyomo code that we execute.
2. LLM writes a template-level lp2graph `Formulation` + instance, grounded by lp2graph.
3. LLM writes a small explicit MILP as JSON (variables, linear rows, objective), validated by a
   pydantic schema and solved by HiGHS.

## Decision Outcome

Option 3 for v1: decisions are local (one train and a few oncoming trains), so explicit models are
small, and the schema is easy to validate. Selector binaries `choose_<OPTION_ID>` with a sum-to-one
row map the solution to an option. Invalid or infeasible models are fed back (up to 3 attempts);
after that the default option applies and the decision counts as a fallback.

### Consequences

- Good: nothing executable crosses the trust boundary; every model is a data record we can analyse.
- Good: the same solver path serves the code-built `fixed-milp` OR baseline.
- Open: option 2 would connect the arm to Paper 1's LP2Graph representation and its structural
  validation; revisit once local decisions work.
