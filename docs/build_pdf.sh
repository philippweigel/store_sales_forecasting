#!/usr/bin/env bash
# Render BUSINESS_SUMMARY.md to a print-ready PDF for sharing as an attachment.
#
#   bash docs/build_pdf.sh
#
# Requires pandoc and Chrome (or Edge). Run from the repository root.
set -euo pipefail

REPO_URL="https://github.com/philippweigel/store_sales_forecasting"
OUT="docs/retail-demand-forecasting-case-study.pdf"
TMP_MD=".pdf_build.md"
TMP_HTML=".pdf_build.html"

CHROME="/c/Program Files/Google/Chrome/Application/chrome.exe"
[ -x "$CHROME" ] || CHROME="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"

trap 'rm -f "$TMP_MD" "$TMP_HTML"' EXIT

# The README link is meaningless in a standalone PDF; point it at the repository.
sed "s|\[README.md\](README.md)|[$REPO_URL]($REPO_URL)|" BUSINESS_SUMMARY.md > "$TMP_MD"

pandoc "$TMP_MD" \
  --standalone \
  --self-contained \
  --css docs/case-study.css \
  --metadata title="Retail demand forecasting case study" \
  --template docs/pdf-template.html \
  -o "$TMP_HTML"

"$CHROME" \
  --headless \
  --disable-gpu \
  --no-pdf-header-footer \
  --print-to-pdf="$(pwd -W 2>/dev/null || pwd)/$OUT" \
  "file:///$(pwd -W 2>/dev/null || pwd)/$TMP_HTML" 2>/dev/null

echo "Wrote $OUT"
