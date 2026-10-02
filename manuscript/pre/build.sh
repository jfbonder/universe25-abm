#!/usr/bin/env bash
# Build the reduced PRE version (manuscript/pre/main.pdf, and supplement.pdf if supplement.tex exists)
# with pdflatex + bibtex (REVTeX 4.2, apsrev4-2), and fail on undefined references or citations.
#   usage: ./build.sh            (main.tex and, if present, supplement.tex)
#          ./build.sh main       (only main.tex)
set -euo pipefail
cd "$(dirname "$0")"

LATEX=(pdflatex -interaction=nonstopmode -halt-on-error -file-line-error)
status=0

check_bib_keys() {
  # Every \cite key of the given .tex files must be in refs.bib; report keys available in ../extended/refs.bib.
  local cited have
  cited=$(cat "$@" | sed 's/\([^\\]\)%.*$/\1/; s/^%.*$//' \
          | grep -o '\\cite[a-zA-Z*]*\(\[[^]]*\]\)*{[^}]*}' | sed 's/.*{//; s/}//' | tr ',' '\n' \
          | sed 's/[[:space:]]//g' | grep -v '^$' | sort -u || true)
  have=$(grep -o '^@[a-zA-Z]*{[^,]*' refs.bib | sed 's/.*{//' | sort -u)
  local missing
  missing=$(comm -23 <(echo "$cited") <(echo "$have") || true)
  if [ -n "$missing" ]; then
    echo "ERROR: cited keys missing from manuscript/pre/refs.bib:" >&2
    for k in $missing; do
      if grep -q "^@[a-zA-Z]*{$k," ../extended/refs.bib 2>/dev/null; then
        echo "  $k  (copy the verified entry from manuscript/extended/refs.bib)" >&2
      else
        echo "  $k  (NOT in manuscript/extended/refs.bib: needs verification)" >&2
      fi
    done
    status=1
  fi
}

build_one() {
  local doc=$1
  echo "== building $doc.tex"
  rm -f "$doc.aux" "$doc.bbl" "$doc.blg"
  "${LATEX[@]}" "$doc.tex" >/dev/null || { echo "ERROR: pdflatex failed on $doc.tex (see $doc.log)" >&2; grep -m5 -A3 '^!\|:[0-9]*:' "$doc.log" >&2 || true; return 1; }
  if grep -q '\\bibdata' "$doc.aux"; then
    bibtex "$doc" >/dev/null || true
  fi
  "${LATEX[@]}" "$doc.tex" >/dev/null
  "${LATEX[@]}" "$doc.tex" >/dev/null
  # Undefined references and citations (LaTeX log) and missing database entries (BibTeX log).
  local bad=0
  if grep -E "LaTeX Warning: (Reference|Citation) .* undefined|There were undefined (references|citations)" "$doc.log" \
       | sort -u >&2; then bad=1; fi
  if [ -f "$doc.blg" ] && grep -E "I didn't find a database entry|^Warning--I didn't find" "$doc.blg" >&2; then bad=1; fi
  if [ -f "$doc.blg" ] && grep -E "error message" "$doc.blg" >&2; then bad=1; fi
  if grep -q "Label(s) may have changed" "$doc.log"; then
    echo "WARNING: labels may have changed in $doc.log; rerun ./build.sh" >&2
  fi
  local pages
  pages=$(grep -o 'Output written on [^ ]* ([0-9]* page' "$doc.log" | grep -o '[0-9]* page' | cut -d' ' -f1 || echo '?')
  if [ "$bad" -eq 0 ]; then
    echo "   OK: $doc.pdf, $pages pages, no undefined references or citations"
  else
    echo "ERROR: undefined references or citations in $doc (listed above)" >&2
    status=1
  fi
}

check_bib_keys main.tex sections/*.tex
build_one main || status=1
if [ "${1:-all}" != "main" ] && [ -f supplement.tex ]; then
  check_bib_keys supplement.tex
  build_one supplement || status=1
fi
# Leftover placeholders (stubs or \todo) are reported, not fatal.
if grep -l '\\todo{' main.tex sections/*.tex 2>/dev/null | grep -q .; then
  echo "NOTE: \\todo placeholders remain in: $(grep -l '\\todo{' main.tex sections/*.tex | tr '\n' ' ')"
fi
exit $status
