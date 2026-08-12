#!/usr/bin/env python3
"""챕터 검수의 기계 검사 — 사람이 산문을 읽기 전에 먼저 돌린다.

검수 가이드(REVIEW_GUIDE_2026-09.pdf) 2·3쪽이 "노트북과 나란히 놓고 출력 개수를
세는 것이 가장 확실" 이라고 한 대조를 자동화한다. 사람은 린터도 이 스크립트도
잡을 수 없는 것(사실정확성·설명의 흐름)에 시간을 쓴다.

검사 항목:
  C1  실행본 존재 · 에러 셀 0건            executed/<폴더>.ipynb
  C2  출력 개수 대조                        실행본 출력 있는 셀 수  ==  pages 의 `▶ 실행 결과` 수
  C3  변환 누락 신호                        `<IPython.core.display.HTML object>`, `<Figure size ...>` 등
  C4  T4 제약                               bf16 / flash_attention_2 문자열 0건, fp16=True 존재
  C5  산문 수치 vs 실측                     페이지 산문의 0.xxxx 가 실행본 출력에 실재하는지
  C6  그림 참조                              pages 의 ![](../assets/...) 가 실파일로 존재하는지
  C7  교차참조                               "부록 §N" 같은 참조가 대상 산출물에 실재하는지

C5 는 *후보* 만 뽑는다. 실행마다 값이 달라지는 metric·소요 시간은 결함이 아니므로
(가이드의 "실측과 다른 수치" 는 재현 가능한 값이 어긋난 경우를 말한다) 사람이
판단해야 한다. 자동으로 결함 처리하지 말 것.

사용:
    python3 check_chapter_review.py 12
    python3 check_chapter_review.py 12 --json
종료코드: C1~C4·C6 위반이 하나라도 있으면 1, 없으면 0 (C5·C7 은 후보 보고라 0).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]      # <repo>/.claude/skills/chapter-review/scripts
PAGES = ROOT / "pages"
EXECUTED = ROOT / "executed"
ASSETS = ROOT / "assets"

# 변환기가 흘렸을 때 페이지에 남는 흔적 (가이드 3쪽 "변환이 흘린 것")
LEAK_PATTERNS = [
    r"<IPython\.core\.display\.[A-Za-z]+ object>",
    r"<Figure size [^>]*>",
    r"<matplotlib\.[A-Za-z.]+ object at 0x[0-9a-f]+>",
    r"object at 0x[0-9a-f]+",
]
RESULT_MARKER = re.compile(r"\*\*▶ 실행 결과\*\*|▶ \*\*실행 결과\*\*")
NUM = re.compile(r"(?<![\w.])(\d\.\d{3,4})(?![\w%])")    # 0.5580 같은 metric 후보 (0.003% 는 제외)
MATH = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$", re.DOTALL)  # LaTeX 수식은 교육용 예시라 대조 대상 아님
SECTION_REF = re.compile(r"부록\s*§\s*(\d+)")


def chapter_dir(num: int) -> Path:
    hits = sorted(ROOT.glob(f"{num:02d}_*"))
    hits = [p for p in hits if p.is_dir()]
    if not hits:
        raise SystemExit(f"챕터 {num} 폴더를 찾을 수 없습니다 (<repo>/{num:02d}_*)")
    return hits[0]


def notebook_outputs(nb_path: Path) -> tuple[int, int, list[str]]:
    """(출력 있는 코드셀 수, 에러 셀 수, 출력 텍스트 목록)"""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    with_out, errors, texts = 0, 0, []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        outs = cell.get("outputs", [])
        if not outs:
            continue
        with_out += 1
        for o in outs:
            if o.get("output_type") == "error":
                errors += 1
                texts.append(f"[ERROR] {o.get('ename')}: {o.get('evalue')}")
            elif o.get("output_type") == "stream":
                texts.append("".join(o.get("text", [])))
            else:
                data = o.get("data", {})
                for key in ("text/plain", "text/html"):
                    if key in data:
                        texts.append("".join(data[key]))
    return with_out, errors, texts


def page_files(num: int) -> list[Path]:
    return sorted(PAGES.glob(f"{num:02d}-*.md"))


def strip_code_fences(md: str) -> str:
    """코드펜스 안(코드·실행 결과)을 제거해 *산문만* 남긴다."""
    out, inside = [], False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            inside = not inside
            continue
        if not inside:
            out.append(line)
    return "\n".join(out)


def prose_only(md: str) -> str:
    """코드펜스와 LaTeX 수식을 제거한 순수 산문.

    수식 안의 숫자는 loss 표·softmax 예시처럼 *가르치려고 고른 값* 이라
    실측과 대조할 대상이 아니다. 대조해야 할 것은 산문이 주장하는 측정값이다.
    """
    return MATH.sub(" ", strip_code_fences(md))


def notebook_prose(nb_path: Path) -> list[tuple[int, str]]:
    """노트북 마크다운 셀의 산문을 (셀 인덱스, 줄) 로. 실측 주장이 여기에도 산다."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    rows = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        for line in prose_only("".join(cell.get("source", []))).splitlines():
            rows.append((i, line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chapter", type=int, help="챕터 번호 (예: 12)")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 으로")
    args = ap.parse_args()

    n = args.chapter
    cdir = chapter_dir(n)
    report: dict = {"chapter": n, "chapter_dir": cdir.name, "violations": [], "candidates": []}
    V = report["violations"].append
    C = report["candidates"].append

    # ── C1 실행본 · 에러 셀 ────────────────────────────────────────────────
    # 챕터 산출물은 <NN>_*.ipynb 뿐이다. 같은 폴더의 appendix_*.ipynb 는 TOC·pages·
    # executed 어디에도 없는 보조 자료라 검수 대상이 아니다(실측 대조 풀에도 안 넣는다).
    all_nbs = sorted(cdir.glob("*.ipynb"))
    nbs = [p for p in all_nbs if p.stem.startswith(f"{n:02d}_")]
    side_nbs = [p for p in all_nbs if p not in nbs]
    report["side_notebooks"] = [p.name for p in side_nbs]
    main_nb = next((p for p in nbs if p.stem == cdir.name), nbs[0] if nbs else None)
    appendix_nbs = [p for p in nbs if p is not main_nb]
    exec_texts: list[str] = []
    main_with_out = 0
    for nb in nbs:
        ex = EXECUTED / nb.name
        if not ex.exists():
            V(f"C1 실행본 없음: executed/{nb.name} — 검수 전에 Colab 실행본을 먼저 만드세요")
            continue
        with_out, errors, texts = notebook_outputs(ex)
        exec_texts += texts                      # 수치 대조는 본편+부록 전부를 실측 풀로 씀
        if nb is main_nb:
            main_with_out = with_out
        if errors:
            V(f"C1 실행본에 에러 셀 {errors}건: executed/{nb.name}")
        report.setdefault("executed", {})[nb.name] = {"cells_with_output": with_out, "error_cells": errors}

    # ── C2 출력 개수 대조 ──────────────────────────────────────────────────
    # 본편 노트북만 대조한다. 부록 페이지는 손으로 쓰는 것이 이 저장소의 관례라
    # (12·14·18·20 부록 페이지 모두 코드펜스 0개) 합산 비교하면 항상 오탐이 난다.
    pages = page_files(n)
    if not pages:
        V(f"C2 pages/{n:02d}-*.md 가 없습니다")
    markers = 0
    for p in pages:
        markers += len(RESULT_MARKER.findall(p.read_text(encoding="utf-8")))
    report["page_result_blocks"] = markers
    report["main_notebook_cells_with_output"] = main_with_out
    if pages and markers < main_with_out:
        V(f"C2 출력 누락 의심: 본편 실행본 {main_with_out}개 vs 페이지 `▶ 실행 결과` {markers}개 "
          f"— 변환이 흘린 출력이 있는지 노트북과 나란히 확인하세요")
    elif pages and markers > main_with_out:
        C(f"C2 페이지 `▶ 실행 결과` {markers}개 > 본편 실행본 {main_with_out}개 "
          f"— 부록 페이지가 출력을 싣고 있는지 확인 (관례상 부록 페이지는 산문 요약)")

    # ── C3 변환 누락 신호 ──────────────────────────────────────────────────
    for p in pages:
        text = p.read_text(encoding="utf-8")
        for pat in LEAK_PATTERNS:
            for m in re.finditer(pat, text):
                line = text[: m.start()].count("\n") + 1
                V(f"C3 변환 누락 흔적 {p.name}:{line} — {m.group(0)[:60]}")

    # ── C4 T4 제약 ─────────────────────────────────────────────────────────
    # 문자열 존재만으로 잡으면 안 된다. "bf16은 T4 미지원" 같은 *설명* 이 주석·산문에
    # 흔히 나오고(Ch 9 가 그렇다), 그건 오히려 옳은 내용이다. 코드 셀에서 주석을 떼고
    # 실제 *사용* 만 본다.
    USE_BF16 = re.compile(r"\bbf16\s*=\s*True|\bbf16_full_eval\s*=\s*True")
    USE_FA2 = re.compile(r"flash_attention_2|attn_implementation\s*=\s*[\"']flash")
    for nb in nbs:
        cells = json.loads(nb.read_text(encoding="utf-8")).get("cells", [])
        code = []
        for c in cells:
            if c.get("cell_type") != "code":
                continue
            for line in "".join(c.get("source", [])).splitlines():
                code.append(line.split("#", 1)[0])          # 주석 제거
        code_src = "\n".join(code)
        for pat, label in ((USE_BF16, "bf16"), (USE_FA2, "flash_attention_2")):
            if pat.search(code_src):
                V(f"C4 T4 미지원 옵션 *사용*: {nb.name} 의 {label} — T4(CC 7.5)는 미지원, 실행 전 결함")
        # fp16 부재는 *후보* 다. Ch 24-34 처럼 커스텀 학습 루프·AMP(GradScaler)·accelerate 를
        # 쓰는 챕터가 많아 위반으로 올리면 3분의 1 챕터에서 오탐이 난다. 사람이 확인한다.
        if "TrainingArguments" in code_src and not re.search(r"\bfp16\s*=\s*True", code_src):
            C(f"C4 {nb.name}: TrainingArguments 에 fp16=True 가 없음 — "
              f"커스텀 AMP(GradScaler)·accelerate 를 쓰는지 확인")

    # ── C5 산문 수치 vs 실측 ───────────────────────────────────────────────
    blob = "\n".join(exec_texts)
    measured = set(NUM.findall(blob))
    # 반올림 표기도 실측으로 인정 (1.000020 → 1.0000, 0.5580 → 0.558)
    for raw in list(measured):
        measured.add(raw.rstrip("0").rstrip("."))
    # 타 챕터 실측 풀 — "Ch 10=0.9030" 같은 교차인용은 이 챕터 출력엔 당연히 없다.
    # 어디에도 없는 값과 구분해야 사람이 볼 것이 남는다.
    elsewhere: dict[str, str] = {}
    for ex in sorted(EXECUTED.glob("*.ipynb")):
        if any(ex.name == nb.name for nb in nbs):
            continue
        try:
            _, _, texts = notebook_outputs(ex)
        except Exception:
            continue
        for v in NUM.findall("\n".join(texts)):
            elsewhere.setdefault(v, ex.stem)
            elsewhere.setdefault(v.rstrip("0").rstrip("."), ex.stem)

    def check_line(where: str, line: str) -> None:
        for val in sorted(set(NUM.findall(line))):
            # continue 여야 한다. return 이면 한 줄에 값이 여럿일 때 첫 일치 뒤가 묻힌다
            # (실제로 "DistilBERT(0.562) 가 sklearn TF-IDF(0.542)" 에서 0.562 를 놓쳤다).
            short = val.rstrip("0").rstrip(".")
            if val in measured or short in measured:
                continue
            src = elsewhere.get(val) or elsewhere.get(short)
            if src:
                C(f"C5[교차인용] {where}: {val} 는 이 챕터가 아닌 {src} 실측 — "
                  f"인용 대상이 맞는지 확인 — {line.strip()[:70]}")
            else:
                C(f"C5[미확인] {where}: {val} 가 어떤 실행본에도 없음 — {line.strip()[:80]}")

    for p in pages:
        for i, line in enumerate(prose_only(p.read_text(encoding="utf-8")).splitlines(), 1):
            check_line(f"{p.name}:{i}", line)
    for nb in nbs:                      # 노트북 마크다운 셀에도 실측 주장이 산다
        for idx, line in notebook_prose(nb):
            check_line(f"{nb.name} cell[{idx}]", line)

    # ── C6 그림 참조 ───────────────────────────────────────────────────────
    for p in pages:
        for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
            ref = m.group(1)
            if ref.startswith("http"):
                V(f"C6 외부 이미지 금지(E4) {p.name}: {ref}")
                continue
            target = (p.parent / ref).resolve()
            if not target.exists():
                V(f"C6 그림 파일 없음 {p.name}: {ref}")

    # ── C7 교차참조 ────────────────────────────────────────────────────────
    for p in pages:
        prose = strip_code_fences(p.read_text(encoding="utf-8"))
        for m in SECTION_REF.finditer(prose):
            C(f"C7 {p.name}: '부록 §{m.group(1)}' 참조 — 부록 *페이지* 에 그 절이 실재하는지 "
              f"직접 확인하세요 (노트북에만 있고 페이지엔 없는 사례가 실제로 있었습니다)")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report["violations"] else 0

    print(f"■ Ch {n} ({cdir.name}) 기계 검사")
    appendix_note = f"  (+ 부록 {len(appendix_nbs)}개)" if appendix_nbs else ""
    print(f"  본편 실행본 출력 셀 {main_with_out}개{appendix_note}  |  "
          f"페이지 `▶ 실행 결과` {markers}개  |  페이지 {len(pages)}개")
    if report["violations"]:
        print(f"\n❌ 위반 {len(report['violations'])}건")
        for v in report["violations"]:
            print(f"   - {v}")
    else:
        print("\n✅ 기계 검사 위반 없음")
    if report["candidates"]:
        print(f"\n🔎 사람이 판단할 후보 {len(report['candidates'])}건 "
              f"(실행마다 달라지는 값은 결함이 아님 — 판단 기준은 SKILL.md 참조)")
        for c in report["candidates"][:40]:
            print(f"   - {c}")
        if len(report["candidates"]) > 40:
            print(f"   … 외 {len(report['candidates']) - 40}건")
    print("\n다음: check_wikidocs_md.py 로 E1~E9 를 돌리고, 그 다음 사람이 산문을 읽습니다.")
    return 1 if report["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
