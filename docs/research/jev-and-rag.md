# Is Jev a driver of, or an alternative to, retrieval-augmented generation?

Status: evidence review, 2026-09-28. Our own experiments are P2-H2a–c (`src/d4r/h2`,
`results/h2`). Sources are listed at the end; vendor numbers are marked as such.

## Short answer

- **Driver of RAG: yes, supported.** As a *stage inside* a retrieval pipeline (reranker, passage
  filter, router, verifier), Jev performs at the level of commercial rerankers at lower price and
  latency, and it is clearly better than a generative LLM at negated queries. Independent evidence
  exists, although it is young and small.
- **Alternative to RAG: only in a narrow sense.** Jev can replace *embedding retrieval over a
  small, closed, listable candidate set* (a taxonomy, a skill roster, ≤ 255 options per Choice
  question, state ≤ ~32k tokens). It is not a replacement for RAG as a whole: it cannot generate
  the answer, and it is not a retrieval index over an open corpus. TypeSafe's own wording is
  "replace or supplement embeddings in RAG pipelines", i.e. the scoring stage, and its
  documentation says accuracy drops on large states full of irrelevant detail.
- **For an MILP assistant this narrow sense fits unusually well**, because the vocabulary of
  optimisation models is closed: constraint families, variable roles and model types are finite
  enums (LP2Graph's `ConstraintDomainClass`, 12 values; the mined taxonomy of Paper 1). Open
  collections (paper PDFs) need the driver mode.

## Evidence

### Vendor (TypeSafe cookbooks, mostly on jev-1.12, small n)

| Cookbook | Method | Result (vendor) |
|---|---|---|
| Re-ranking (CLERC legal, 40 queries, BM25 top-30) | one yes/no question per query–candidate pair | top-1 5 % → 18 %, top-10 38 % → 62 %; 1.5 M input tokens for $0.065; no competitor reranker |
| Classifying RAG passages (81 passages, 6 queries) | four yes/no questions per passage (relevant, evidence, contradicts premise, injection) | qualitative: a planted injection ranked first by cosine similarity is dropped |
| Skill suggestion (182 skills, 488 synthetic requests) | Choice over the roster + "is any skill needed?" | wrong loads 16.8 % → 7.3 %, needless loads 9.8 % → 4.0 % |
| Hierarchical classification (4 examples) | beam search over a taxonomy | beam 4/4 vs greedy 2/4 |
| Classification using confidence (60 filings, 75 classes) | flat Choice, fall back to parent when unsure | 39/60 exact; confident half 27/30 |

### Independent (published September 2026; READMEs, not re-run by us)

| Study | Setup | Finding |
|---|---|---|
| jev-rerank-bench | 8 English datasets, 1 617 queries, BM25 top-30 | nDCG@10: Jev 0.692, Cohere Rerank 4 Pro 0.691 (tie), DeepSeek V4.1 Flash as JSON reranker 0.682; negation (NevIR) Jev 71 % vs DeepSeek 22 %; Jev 422 ms, $0.45 per 1k queries vs Cohere 844 ms, $2.51; reversing passage order flips Jev's top pick in 24.7 % of queries |
| S1Rank | BM25 top-100, all documents in one request | beats bge-reranker-v2-m3 and monoT5-3B on DL19/DL20/TREC-COVID/NFCorpus/SciFact; probabilities change across byte-identical requests; escalation on Jev's uncertainty did **not** beat random escalation |
| Samelogic blog | 240 synthetic records | top-1: Jev with all documents 20/20, BM25+Jev 19/20, embeddings 17/20; tokens grow with the corpus; "improved evidence selection, but did not replace RAG" |

### Ours (pilot, 2026-09-28)

Routing 123 published railway-model constraints and 48 variables onto the classes of a curated
literature catalogue (the catalogue itself is private; only aggregates are reported): Jev accuracy
0.756 (constraints) / 0.917 (variables), DeepSeek-Flash 0.805 / 0.917 (difference not
significant), embedding similarity to class descriptions 0.309 / 0.625; Jev with escalation to
DeepSeek below confidence 0.9 reached 0.821 / 0.958. Jev was about 2.5× faster than DeepSeek-Flash,
not the advertised 100×; against cached DeepSeek prompts its cost advantage was small.
The full H2 study (requirement → constraint-family routing with negation, relevance filtering in
front of an MILP generator) is in `results/h2/`.

## Caveats

- All evidence is from the first weeks after Jev's release (2026-09-15); the model is closed and
  versions change (we pin `jev-1.13.0` and cache every answer).
- Jev is not bit-deterministic; test–retest flips were 6.5 % on our dispatching cards.
- Numbers, counting and dates are documented weak spots; our dispatching cards therefore bucket
  numbers in code, which helped Jev (regret 13.5 bucketed vs 15.5 raw) and hurt DeepSeek.
- Uncertainty-gated escalation helped in our routing pilot and in our dispatching bench (Jev → LLM
  + LP, fixed on scenario A, tested on B: regret 10.8 vs 11.9 for random escalation at the same
  rate), but did not help in S1Rank's reranking study: it is task-dependent and must always be
  compared with random escalation.

## Sources

- TypeSafe documentation: https://docs.typesafe.ai/llms.txt (cookbooks `rerank_typesafe`,
  `classifying_rag_passages`, `skill_suggestion`, `hierarchical_classification`,
  `classification_using_confidence`; `concepts/use-case-map`; `model-jaggedness/jev-1.13`;
  `models`), accessed 2026-09-28.
- https://github.com/anessbelbati/jev-rerank-bench
- https://github.com/zaesho/S1Rank
- https://samelogic.com/blog/can-jev-replace-rag-or-just-improve-retrieval
- Background: Lewis et al. 2020 (RAG); Jeong et al. 2024 (Adaptive-RAG); Asai et al. 2023
  (Self-RAG); Weller et al. 2024 (NevIR).
