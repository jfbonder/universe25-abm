#!/usr/bin/env bash
# Build the Ecological Modelling manuscript (main.pdf) and its Supplementary Material (supplement.pdf) with
# pdflatex + bibtex (elsarticle, elsarticle-harv). The supplement is built first, because main.tex reads its labels
# through xr-hyper (\externaldocument[SM-]{supplement}). Fails on undefined references or citations, on missing
# BibTeX entries and on "??" in the output.
#   usage: ./build.sh
# Requirements: pdflatex and bibtex (TeX Live), the Elsevier class elsarticle.cls and elsarticle-harv.bst, which
# come with the package texlive-publishers (Debian/Ubuntu: apt install texlive-publishers), plus xr-hyper, booktabs,
# longtable, array and adjustbox (texlive-latex-extra). Optional: detex (word counts) and pdftotext (check for "??").
set -euo pipefail
cd "$(dirname "$0")"
kpsewhich elsarticle.cls >/dev/null || { echo "ERROR: elsarticle.cls not found; install texlive-publishers" >&2; exit 1; }

LATEX=(pdflatex -interaction=nonstopmode -halt-on-error -file-line-error)
status=0

build_one() {
  local doc=$1
  echo "== building $doc.tex"
  rm -f "$doc.aux" "$doc.bbl" "$doc.blg"
  "${LATEX[@]}" "$doc.tex" >/dev/null || { echo "ERROR: pdflatex failed on $doc.tex (see $doc.log)" >&2;
    grep -m5 -A3 '^!\|:[0-9]*:' "$doc.log" >&2 || true; return 1; }
  bibtex "$doc" >/dev/null || true
  "${LATEX[@]}" "$doc.tex" >/dev/null
  "${LATEX[@]}" "$doc.tex" >/dev/null
  local bad=0
  if grep -E "LaTeX Warning: (Reference|Citation) .* undefined|There were undefined (references|citations)" "$doc.log" \
       | sort -u >&2; then bad=1; fi
  if grep -E "I didn't find a database entry|^Warning--|error message" "$doc.blg" >&2; then bad=1; fi
  if command -v pdftotext >/dev/null && pdftotext "$doc.pdf" - | grep -n '??' >&2; then
    echo "ERROR: '??' in $doc.pdf" >&2; bad=1; fi
  if grep -q "Label(s) may have changed" "$doc.log"; then "${LATEX[@]}" "$doc.tex" >/dev/null; fi
  local pages
  pages=$(grep -o 'Output written on [^ ]* ([0-9]* page' "$doc.log" | grep -o '[0-9]* page' | cut -d' ' -f1 || echo '?')
  if [ "$bad" -eq 0 ]; then echo "   OK: $doc.pdf, $pages pages"; else echo "ERROR in $doc (listed above)" >&2; status=1; fi
}

build_one supplement || status=1
build_one main || status=1

# Word counts (detex; approximate): abstract, body (Introduction to Conclusions, without captions), highlights.
if command -v detex >/dev/null; then
  strip() { sed '/\\caption{/,/\\label{fig:/d' "$@" | detex -n | wc -w; }
  echo "   words: abstract $(detex -n sections/abstract.tex | wc -w), body without captions" \
       "$(strip sections/intro.tex sections/methods.tex sections/results.tex sections/discussion.tex sections/conclusions.tex)"
fi
echo "   references cited: $(grep -c '^\\bibitem' main.bbl)"
awk 'length > 85 {print "ERROR: highlight longer than 85 characters: " $0; e=1} END {exit e}' highlights.txt || status=1
grep -l 'authorpending{' main.tex sections/*.tex supplement.tex 2>/dev/null | sed 's/^/NOTE: author placeholder in /' || true
exit $status
