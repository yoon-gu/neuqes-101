#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

JOBNAME="neuqes-101-ch01-34-preview"
rm -rf build/epub "${JOBNAME}-epub3"
rm -f "${JOBNAME}".{4ct,4tc,aux,css,dvi,html,idv,lg,log,ncx,opf,tmp,xdv,xref}
rm -f "${JOBNAME}"*.xhtml content.opf

mkdir -p build/epub
python3 epub/generate_covers.py
pandoc ebook-main.tex \
  --from=latex \
  --to=epub3 \
  --standalone \
  --toc \
  --toc-depth=1 \
  --mathml \
  --metadata=language:ko-KR \
  --css=epub/epub-style.css \
  --epub-cover-image=epub/covers/cover-illustration-token-core.png \
  --resource-path=.:.. \
  --output="build/epub/${JOBNAME}.epub"

echo "EPUB written to book/build/epub/${JOBNAME}.epub"
