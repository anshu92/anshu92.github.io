#!/usr/bin/env bash
# One-time helper: paste secrets locally; they are sent to GitHub via `gh` only (not printed).
# Prereq: `gh` installed and `gh auth login` done with access to the repo.
#
# Usage:
#   ./scripts/setup-github-secrets.sh
#   REPO=owner/repo ./scripts/setup-github-secrets.sh
#   ./scripts/setup-github-secrets.sh owner/repo
#
set -euo pipefail

REPO_ARG="${1:-${REPO:-}}"

if ! command -v gh >/dev/null 2>&1; then
  echo "Install GitHub CLI: https://cli.github.com/  (e.g. brew install gh)" >&2
  exit 1
fi

# Resolve repo: arg > env > origin remote
if [[ -n "$REPO_ARG" ]]; then
  REPO="$REPO_ARG"
else
  REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)
  if [[ -z "$REPO" ]]; then
    echo "Could not detect repo. Set REPO=owner/name or: $0 owner/name" >&2
    exit 1
  fi
fi

echo "Repository: $REPO"
echo "Auth check..."
gh auth status -h github.com >/dev/null

set_secret() {
  local name="$1"
  local help="$2"
  local v=""
  echo ""
  echo "[$name] $help"
  read -r -s -p "Paste value (empty = skip): " v
  echo "" >&2
  if [[ -n "$v" ]]; then
    printf '%s' "$v" | gh secret set "$name" --repo "$REPO"
    echo "  -> set $name"
  else
    echo "  (skipped $name)"
  fi
  v=""
}

echo ""
echo "Paste each secret when prompted. Input is hidden. Leave empty to skip a key."
echo ""

# --- Secrets used by .github/workflows/daily-blog-draft.yml + blogpipe ---
set_secret OPENROUTER_API_KEY "LLM (OpenRouter) — required for draft/rank/edit"
set_secret OPENAI_API_KEY "OpenAI (optional: embeddings if you use OpenAI base URL)"
set_secret GEMINI_API_KEY "Google Gemini (optional: hero image generation)"
set_secret GROQ_API_KEY "Groq (optional; workflow passes it for parity with evr)"
set_secret SEMANTIC_SCHOLAR_API_KEY "Semantic Scholar (optional: higher rate limits for research)"
set_secret FAL_API_KEY "Fal (optional: Flux image fallback)"
set_secret HF_TOKEN "Hugging Face (optional: Inference API FLUX fallback for body figures; leave unset to use Imagen+Pollinations only)"
set_secret LANGCHAIN_API_KEY "LangSmith (optional: set LANGCHAIN_TRACING_V2 var to true to trace blogpipe graph runs)"
set_secret TAVILY_API_KEY "Tavily (optional: free-tier web search for BLOGPIPE_MCP_ENRICHMENT; use instead of paid Brave Search)"
set_secret EMAIL_USERNAME "Gmail address used for SMTP (must match the account for the app password below)"
set_secret EMAIL_PASSWORD "Gmail App Password ONLY: myaccount.google.com → Security → 2-Step Verification → App passwords (16 chars). Do NOT use your regular Gmail password — SMTP will return 534 if wrong."
set_secret EMAIL_TO "Notification recipient (optional; default is EMAIL_USERNAME)"

echo ""
echo "Done. Verify: gh secret list --repo $REPO"
echo "To set a variable (non-secret), e.g.: gh variable set BLOGPIPE_DRY_RUN -b 0 --repo $REPO"
