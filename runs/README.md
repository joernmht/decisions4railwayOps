# Run records

Everything under `runs/` is raw experimental record, committed for reproducibility:

- `gameset/` — labelled decision cards (one JSON per line: card, rollout value of every option,
  best option(s), spread). Built by `d4r gameset`.
- `bench/<tag>/<engine>.jsonl` — every engine's answer to every consequential card (option,
  probabilities, regret, latency, tokens, cost, served model).
- `contest/<tag>/<scenario>-<engine>.jsonl` — closed-loop episodes (result + all decisions).
- `cache/*.jsonl` — the paid API responses keyed by request hash; rerunning an experiment replays
  them. They contain requests and answers, never credentials.

**Terms of use of third-party outputs.** Jev answers are TypeSafe "Output", owned by the customer
under TypeSafe's Master Customer Agreement (§4.2); that agreement prohibits using Output to distil
or train a model that imitates the service (§2.3(b)). Please respect that restriction when reusing
these records. DeepSeek answers are published under DeepSeek's terms of use for API output.
