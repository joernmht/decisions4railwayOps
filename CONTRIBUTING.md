# Contributing

Thanks for your interest. This is a research lab repository, so the bar is **reproducibility**
more than feature breadth.

## Ground rules

1. **Determinism where we control it.** Every scenario is fully defined by a seed and a config.
   Seed every RNG, sort before emitting order-sensitive output, never rely on wall-clock values in
   results.
2. **Code decides, models judge.** Engines return a typed decision; the contest harness applies it.
   No model output is ever executed as code.
3. **No keys in the repo.** See [SECURITY.md](SECURITY.md).
4. **Record, then interpret.** Every live run writes a JSONL record (state, question, answer,
   latency, tokens, cost) under `runs/`. Summaries that go into `results/` are regenerated from
   those records by a script, never edited by hand.
5. **Write down decisions.** Architecture decisions go into `docs/adr/` (MADR format); dated
   observations go into `docs/lab-notebook.md`.

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy src && pytest
```

The offline test suite never touches the network. Live tests are marked `@pytest.mark.live` and
only run with `pytest -m live` when keys are present.

## Style

ruff (lint + format, line length 100) and mypy. `from __future__ import annotations` in every
module; frozen dataclasses for value objects; public functions carry docstrings.
