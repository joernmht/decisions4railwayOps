#!/usr/bin/env bash
# Secret scan for this PUBLIC repository. Used by CI and as a pre-commit hook.
#
#   scripts/check_secrets.sh            scan all tracked files (CI)
#   scripts/check_secrets.sh --staged   scan only staged additions (pre-commit)
#
# Patterns: TypeSafe keys (apikey_…), OpenAI-style keys (sk-…, which DeepSeek also uses),
# and any committed .env file other than .env.example.
set -euo pipefail

PATTERN='(apikey_[0-9a-fA-F]{20,}|sk-[A-Za-z0-9]{20,}|(TYPESAFE|DEEPSEEK|SCADS)_API_KEY=[^[:space:]]+)'

if [[ "${1:-}" == "--staged" ]]; then
  if git diff --cached --name-only | grep -qE '(^|/)\.env(\..+)?$' \
     && ! git diff --cached --name-only | grep -qE '(^|/)\.env\.example$'; then
    echo "check_secrets: refusing to commit a .env file" >&2; exit 1
  fi
  if git diff --cached -U0 | grep -E '^\+' | grep -qE "$PATTERN"; then
    echo "check_secrets: staged content looks like an API key — remove it" >&2; exit 1
  fi
  exit 0
fi

if git ls-files | grep -E '(^|/)\.env(\..+)?$' | grep -vqE '\.env\.example$'; then
  echo "check_secrets: a .env file is tracked" >&2; exit 1
fi
if git ls-files -z | xargs -0 grep -nIE "$PATTERN" -- 2>/dev/null; then
  echo "check_secrets: tracked content looks like an API key (see lines above)" >&2; exit 1
fi
echo "check_secrets: clean"
