#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

JOBNAME="neuqes-101-ch01-34-preview"
DRAFT_JOBNAME="neuqes-101-ch01-34-publisher-draft-watermarked"
COMPACT_DRAFT_JOBNAME="neuqes-101-ch01-34-publisher-draft-watermarked-compact"
DRAFT_SOURCE=".ebook-publisher-draft.tmp.tex"
COMPACT_DRAFT_SOURCE=".ebook-publisher-draft-compact.tmp.tex"
rm -rf build/epub "${JOBNAME}-epub3"
rm -f "${JOBNAME}".{4ct,4tc,aux,css,dvi,html,idv,lg,log,ncx,opf,tmp,xdv,xref}
rm -f "${DRAFT_JOBNAME}".{4ct,4tc,aux,css,dvi,html,idv,lg,log,ncx,opf,tmp,xdv,xref}
rm -f "${COMPACT_DRAFT_JOBNAME}".{4ct,4tc,aux,css,dvi,html,idv,lg,log,ncx,opf,tmp,xdv,xref}
rm -f "${JOBNAME}"*.xhtml "${DRAFT_JOBNAME}"*.xhtml "${COMPACT_DRAFT_JOBNAME}"*.xhtml content.opf "${DRAFT_SOURCE}" "${COMPACT_DRAFT_SOURCE}"

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

python3 - <<'PY'
from pathlib import Path

def draft_source(compact: bool = False) -> str:
    source = Path("ebook-main.tex").read_text(encoding="utf-8")
    source = source.replace(
    "{\\Large EPUB 미리보기 샘플\\par}",
    "{\\Large EPUB 출판 검토용 초안\\par}",
    )
    source = source.replace(
    "{\\small Ch 1--34 변환 검증용\\par}",
    "{\\small Ch 1--34 출판사 검토용 · 무단 복제 및 배포 금지\\par}",
    )
    source = source.replace(
    "이 EPUB 샘플은 리디북스와 같은 리플로우 전자책 뷰어에서 코드, 출력, 표, 그림, FAQ 박스가 어떻게 보이는지 확인하기 위한 변환 검증본입니다.",
    "이 EPUB은 출판사 검토를 위해 제공되는 초안입니다. 무단 복제, 재배포, 외부 공유를 금하며, 본문과 그림은 검토 과정에서 수정될 수 있습니다.",
    )
    if compact:
        source = source.replace(
            "이 EPUB은 출판사 검토를 위해 제공되는 초안입니다.",
            "이 EPUB은 출판사 검토를 위해 제공되는 compact 초안입니다. 책 본문에는 핵심 코드만 남기고 전체 코드는 각 장의 Colab 링크로 연결합니다.",
        )
    return source

Path(".ebook-publisher-draft.tmp.tex").write_text(draft_source(compact=False), encoding="utf-8")
Path(".ebook-publisher-draft-compact.tmp.tex").write_text(draft_source(compact=True), encoding="utf-8")
PY

pandoc "${DRAFT_SOURCE}" \
  --from=latex \
  --to=epub3 \
  --standalone \
  --toc \
  --toc-depth=1 \
  --mathml \
  --metadata=language:ko-KR \
  --metadata=title:"Hugging Face로 시작하는 텍스트 분석 입문 - 출판 검토용 초안" \
  --css=epub/epub-style.css \
  --css=epub/epub-draft.css \
  --epub-cover-image=epub/covers/cover-illustration-token-core-draft.png \
  --resource-path=.:.. \
  --output="build/epub/${DRAFT_JOBNAME}.epub"

rm -f "${DRAFT_SOURCE}"
echo "Draft EPUB written to book/build/epub/${DRAFT_JOBNAME}.epub"

pandoc "${COMPACT_DRAFT_SOURCE}" \
  --from=latex \
  --to=epub3 \
  --standalone \
  --toc \
  --toc-depth=1 \
  --mathml \
  --metadata=language:ko-KR \
  --metadata=title:"Hugging Face로 시작하는 텍스트 분석 입문 - 출판 검토용 compact 초안" \
  --css=epub/epub-style.css \
  --css=epub/epub-draft.css \
  --epub-cover-image=epub/covers/cover-illustration-token-core-draft.png \
  --resource-path=.:.. \
  --output="build/epub/${COMPACT_DRAFT_JOBNAME}.epub"

rm -f "${COMPACT_DRAFT_SOURCE}"
echo "Compact draft EPUB written to book/build/epub/${COMPACT_DRAFT_JOBNAME}.epub"
