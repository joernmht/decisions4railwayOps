# 2. API keys live outside the (public) repository

- Status: accepted
- Date: 2026-09-28
- Deciders: Jörn Maurischat

## Context and Problem Statement

The lab calls paid APIs (TypeSafe Jev, DeepSeek) from a public repository. A single leaked key costs
money and forces a rotation.

## Decision Drivers

- The repository is public and mirrored.
- Run records must be publishable as-is (prompts, answers, latency, tokens).

## Decision Outcome

Keys are read from the process environment only (`d4r.apikeys.load_secrets`), sourced from real
environment variables, a file named by `D4R_SECRETS_FILE`, or a git-ignored `.env`. The code never
prints or logs a key or a request header. Three guards back this up: `.gitignore` (`.env`,
`secrets*`, `*.key`), `scripts/check_secrets.sh` as a pre-commit hook, and the same scan as a CI job.

### Consequences

- Good: run records can be committed verbatim for reproducibility.
- Bad: a contributor has to provision their own keys to rerun live experiments; the offline test
  suite and the recorded runs remain usable without them.
