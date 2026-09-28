# Security and secrets

This repository is **public**. It calls paid third-party APIs (TypeSafe, DeepSeek), so the
one security rule that matters most is: **never commit an API key.**

- Keys are read from the environment only (`TYPESAFE_API_KEY`, `DEEPSEEK_API_KEY`). Put them in a
  local `.env` (git-ignored, see `.env.example`) or in a secrets file outside the repository and
  point `D4R_SECRETS_FILE` at it.
- The code never logs request headers or keys. Run records (`runs/`) store prompts, answers,
  token usage and latency, but no credentials.
- CI runs a secret scan (`scripts/check_secrets.sh`) on every push; please install the same check as
  a local pre-commit hook (`pre-commit install`).

If you find a leaked credential or a vulnerability, please do **not** open a public issue. Contact
the maintainer via the e-mail address in the Git history or on the TU Dresden staff page instead.
