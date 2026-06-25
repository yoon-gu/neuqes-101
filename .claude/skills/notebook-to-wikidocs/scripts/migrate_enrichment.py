#!/usr/bin/env python3
"""④ 산문(코드 설명·위 코드 읽기·결과 해석) 마이그레이션.

재변환(build_wikidocs.py)은 pages/*.md 를 노트북에서 새로 찍어내 손으로 넣은 ④ 산문을
지운다. 이 스크립트는 **재변환 직전(git)의 기존 pages** 에서 ④ 를 추출해, **새로 변환된
pages** 의 같은 코드셀 자리에 다시 붙인다. 키는 '노트북 코드셀의 코드 텍스트'.

핵심 아이디어
- ④ 는 모두 '특정 코드셀'에 매달려 있다: (A) 코드 앞 설명, (B) 코드 조각 뒤 '위 코드 읽기',
  (C) 출력 뒤 '결과 해석'. 코드셀 본문(정규화)을 키로 OLD→NEW 이식하면 분할 구조가 달라져도
  안전하다.
- (B)는 한 코드셀이 여러 조각으로 쪼개져 있을 수 있다. OLD 조각 경계를 그대로 복원해
  NEW 의 단일 펜스를 같은 지점에서 다시 쪼개고 그 사이에 '위 코드 읽기'를 끼운다.
- (A)는 마커가 없어 노트북 마크다운과 구분이 필요하다 → 노트북 마크다운에 없는 산문만 ④로 본다.

사용:  python3 migrate_enrichment.py <ch_num> --old-ref HEAD
       (OLD 는 git ref 에서, NEW 는 워킹트리 pages/ 에서 읽는다)
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PAGES = ROOT / "pages"


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _is_real_codedesc(p: str) -> bool:
    """④A(코드 앞 짧은 설명)인지. 표/인용/과장 길이는 구버전 노트북 잔재일 수 있어 제외."""
    if "|" in p:                       # 마크다운 표
        return False
    if p.lstrip().startswith(">"):     # 인용/콜아웃(부록 안내 등)
        return False
    if p.lstrip().startswith(("- ", "* ", "1.")):  # 목록
        return False
    return len(p) <= 400               # 2-3문장 분량 상한


# --------------------------------------------------------------------------- #
# 노트북에서 코드셀·마크다운 모으기
# --------------------------------------------------------------------------- #
def chapter_slug(num: int):
    """레포 루트 NN_slug/ 디렉터리에서 챕터 슬러그. (부록 노트북은 디렉터리가 없음)"""
    for d in sorted(ROOT.glob(f"{num:02d}_*")):
        if d.is_dir():
            return d.name[3:]
    return None


def load_notebook(num: int):
    # 메인 노트북만(부록 노트북 12_..._data_scaling.ipynb 등 제외). 폴더 슬러그로 특정.
    slug = chapter_slug(num)
    main = ROOT / "executed" / f"{num:02d}_{slug}.ipynb" if slug else None
    if not (main and main.exists()):
        # 폴백: 가장 짧은 이름(메인이 가장 짧음), appendix 제외
        cands = sorted((p for p in (ROOT / "executed").glob(f"{num:02d}_*.ipynb")
                        if "appendix" not in p.name), key=lambda p: len(p.name))
        if not cands:
            sys.exit(f"executed/{num:02d}_*.ipynb 없음")
        main = cands[0]
    nb = json.load(open(main))
    code_norms, md_norms = [], []
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and src.strip():
            code_norms.append(_norm(src))
        elif c["cell_type"] == "markdown" and src.strip():
            md_norms.append(_norm(src))
    md_all = "".join(md_norms)
    return code_norms, md_all


# --------------------------------------------------------------------------- #
# 페이지 → 블록 토큰 스트림
# --------------------------------------------------------------------------- #
# 블록 종류: code / output / walk / interp / header / prose
def parse_blocks(text: str):
    lines = text.splitlines()
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```python"):
            j = i + 1
            buf = []
            while j < n and lines[j].strip() != "```":
                buf.append(lines[j]); j += 1
            blocks.append(("code", "\n".join(buf)))
            i = j + 1
        elif s == "**▶ 실행 결과**":
            # 마커 + 뒤따르는 출력물(텍스트 펜스/이미지)을 흡수. 이미지 출력
            # (`![...](..png)`)도 포함해야 그 뒤 '결과 해석'을 코드셀에 정확히 매단다.
            # 단 코드 펜스(```python)는 흡수하지 않는다(다음 셀을 삼키면 안 됨).
            out_lines = [ln]
            j = i + 1
            while j < n:
                k = j
                while k < n and not lines[k].strip():  # 사이 빈 줄 허용
                    k += 1
                if k >= n:
                    j = k; break
                t = lines[k].strip()
                if t.startswith("```") and not t.startswith("```python"):
                    out_lines.extend(lines[j:k+1]); j = k + 1
                    while j < n and lines[j].strip() != "```":
                        out_lines.append(lines[j]); j += 1
                    if j < n:
                        out_lines.append(lines[j]); j += 1
                elif t.startswith("!["):
                    out_lines.extend(lines[j:k+1]); j = k + 1
                else:
                    break
            blocks.append(("output", "\n".join(out_lines).rstrip()))
            i = j
        elif s.startswith("**위 코드 읽기**"):
            j = i
            buf = []
            while j < n and lines[j].strip():
                buf.append(lines[j]); j += 1
            blocks.append(("walk", "\n".join(buf)))
            i = j
        elif s == "**결과 해석**":
            # 표준 형식: 마커 단독 줄 + 빈 줄 + 한 문단. 다음 빈 줄 전까지만 흡수해
            # 뒤따르는 노트북 산문을 결과 해석에 끌어들이지 않는다(이식 시 중복 방지).
            buf = [ln]; j = i + 1
            while j < n and not lines[j].strip():
                buf.append(lines[j]); j += 1
            while j < n and lines[j].strip():
                buf.append(lines[j]); j += 1
            blocks.append(("interp", "\n".join(buf).rstrip()))
            i = j
        elif s.startswith("#"):
            blocks.append(("header", ln)); i += 1
        elif not s:
            i += 1
        else:
            j = i
            buf = []
            while j < n and lines[j].strip():
                buf.append(lines[j]); j += 1
            blocks.append(("prose", "\n".join(buf)))
            i = j
    return blocks


# --------------------------------------------------------------------------- #
# OLD 블록 → 코드셀별 ④ 패키지
# --------------------------------------------------------------------------- #
def extract_enrichment(old_blocks, code_norms, md_all):
    """{정규화코드셀: {'A':[..], 'chunks':[(code,walk), ..], 'C':[..]}}"""
    enrich = {}
    cellset = set(code_norms)
    i, n = 0, len(old_blocks)
    pending_prose = []   # 직전 코드/출력/헤더 이후 쌓인 산문 (A 후보)
    while i < n:
        kind, payload = old_blocks[i]
        if kind == "code":
            # 이 코드셀을 (조각 합쳐) 완성될 때까지 모은다
            chunks = []          # (code, walk)
            acc = ""
            j = i
            matched = None
            while j < n and old_blocks[j][0] == "code":
                code = old_blocks[j][1]
                walk = None
                if j + 1 < n and old_blocks[j+1][0] == "walk":
                    walk = old_blocks[j+1][1]
                    nxt = j + 2
                else:
                    nxt = j + 1
                chunks.append([code, walk])
                acc = _norm(acc + code)
                if acc in cellset:
                    matched = acc
                    j = nxt
                    break
                j = nxt
            if matched is None:
                # 합쳐도 노트북 셀과 안 맞으면(이례적) 첫 조각만 키로
                matched = _norm(chunks[0][0])
            # A(코드 앞 설명): 마커가 없어 OLD 페이지의 구버전·발산 내용과 섞일 수 있다.
            # 1차로 표/인용/목록/과장 길이를 제외하고, 2차로 apply 단계의 중복 가드
            # (노트북 본문과 겹치면 건너뜀)로 구버전 부활(예: Ch7 옛 표)을 막는다.
            A = [p for p in pending_prose
                 if _norm(p) not in md_all and _is_real_codedesc(p)]
            # C: 이 셀 출력 뒤 결과 해석
            C = []
            k = j
            # 출력 블록(있으면) 건너뛰고 결과 해석 수집
            if k < n and old_blocks[k][0] == "output":
                k += 1
            while k < n and old_blocks[k][0] == "interp":
                C.append(old_blocks[k][1]); k += 1
            enrich.setdefault(matched, {"A": A, "chunks": chunks, "C": C})
            pending_prose = []
            i = j
        elif kind == "output":
            pending_prose = []
            i += 1
        elif kind == "interp":
            i += 1
        elif kind == "header":
            pending_prose = []
            i += 1
        else:  # prose
            pending_prose.append(payload)
            i += 1
    return enrich


# --------------------------------------------------------------------------- #
# NEW 페이지에 ④ 재부착
# --------------------------------------------------------------------------- #
def resplit_code(new_code: str, chunks):
    """NEW 단일 펜스 코드를 OLD 조각 경계대로 다시 쪼갠다. (코드, walk) 리스트 반환.

    경계는 OLD 조각의 *정규화 내용*으로 맞춰(빈 줄 위치에 안 흔들림), 각 조각은
    앞뒤 빈 줄을 제거해 ```python 바로 아래가 빈 줄로 시작하지 않게 한다(조각 사이는
    '위 코드 읽기'로 분리되므로 경계 빈 줄이 불필요)."""
    if len(chunks) == 1:
        return [(new_code.strip("\n"), chunks[0][1])]
    new_lines = new_code.split("\n")
    out = []
    pos = 0
    for idx, (ocode, walk) in enumerate(chunks):
        if idx == len(chunks) - 1:
            piece = "\n".join(new_lines[pos:])
        else:
            target = _norm(ocode)
            acc, k = "", pos
            while k < len(new_lines) and _norm(acc) != target:
                acc += new_lines[k]
                k += 1
            piece = "\n".join(new_lines[pos:k])
            pos = k
        out.append((piece.strip("\n"), walk))
    return out


def _a_overlaps_orig(a: str, orig_norm: str) -> bool:
    """(A) 의 실질 줄(>=30자)이 이미 노트북 본문(재빌드본)에 있으면 True.
    수식·코드·표 행 등 노트북과 부분 겹침을 잡아 중복 삽입을 막는다."""
    for ln in a.splitlines():
        s = ln.strip()
        if len(s) >= 30 and _norm(s) in orig_norm:
            return True
    return False


def apply_enrichment(new_text: str, enrich, stats, orig_norm=""):
    blocks = parse_blocks(new_text)
    out = []
    for idx, (kind, payload) in enumerate(blocks):
        if kind == "code":
            key = _norm(payload)
            pkg = enrich.get(key)
            if not pkg:
                out.append(("code", payload)); continue
            # (A) 코드 앞 설명 — 펜스 바로 앞에 삽입.
            # 직전 산문과 같거나, 노트북 본문과 겹치면(구버전 잔재·중복) 건너뜀.
            for a in pkg["A"]:
                prev_prose = out and out[-1][0] == "prose" and _norm(out[-1][1]) == _norm(a)
                if prev_prose or _a_overlaps_orig(a, orig_norm):
                    continue
                out.append(("prose", a)); stats["A"] += 1
            # (B) 조각 분할 + 위 코드 읽기
            pieces = resplit_code(payload, pkg["chunks"])
            for pcode, walk in pieces:
                out.append(("code", pcode))
                if walk:
                    out.append(("walk", walk)); stats["B"] += 1
            # (C) 는 다음 output 뒤에 넣어야 하므로 표시만
            out.append(("__pending_C__", pkg["C"]))
        elif kind == "output":
            out.append(("output", payload))
            # 직전에 __pending_C__ 있으면 출력 뒤에 결과 해석
        else:
            out.append((kind, payload))
    # __pending_C__ 를 바로 뒤 output 뒤로 재배치
    final = []
    pending_C = None
    for kind, payload in out:
        if kind == "__pending_C__":
            pending_C = payload
            continue
        final.append((kind, payload))
        if kind == "output" and pending_C:
            for c in pending_C:
                final.append(("interp", c)); stats["C"] += 1
            pending_C = None
    # 남은 C(출력 없는 셀) 그냥 코드 뒤에 — 드묾
    if pending_C:
        for c in pending_C:
            final.append(("interp", c)); stats["C"] += 1
    return render_blocks(final)


def render_blocks(blocks):
    parts = []
    for kind, payload in blocks:
        if kind == "code":
            parts.append("```python\n" + payload + "\n```")
        else:
            parts.append(payload)
    return "\n\n".join(parts).strip() + "\n"


# --------------------------------------------------------------------------- #
def git_show(ref: str, relpath: str):
    try:
        return subprocess.check_output(["git", "show", f"{ref}:{relpath}"],
                                       cwd=ROOT, stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        return None


def _exec_fingerprint(nb: dict) -> str:
    """실행본 지문: 코드셀 소스 + 텍스트 출력(이미지는 존재만). 결과가 바뀌면 달라진다."""
    parts = []
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        outs = []
        for o in c.get("outputs", []):
            if "text" in o:
                outs.append("".join(o["text"]))
            data = o.get("data", {})
            if "text/plain" in data:
                outs.append("".join(data["text/plain"]))
            if any(k.startswith("image/") for k in data):
                outs.append("<image>")   # base64 는 매 실행 달라지므로 존재만 비교
        parts.append(_norm(src) + "||" + _norm("".join(outs)))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("num", type=int)
    ap.add_argument("--old-ref", default="HEAD")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="실행본이 바뀌어도 ④ 보존 강행")
    args = ap.parse_args()
    num = args.num

    code_norms, md_all = load_notebook(num)
    slug = chapter_slug(num)

    # 실행본 변경 가드: 현재 executed 노트북이 --old-ref 대비 바뀌었으면(재실행으로
    # 결과가 달라졌으면) 옛 ④(특히 결과 해석)는 더 이상 맞지 않으므로 이식을 건너뛴다.
    # → 페이지는 갓 재변환된 상태(④ 없음)로 두고 새로 작성한다. (--force 로 강행)
    if not args.force:
        old_nb_txt = git_show(args.old_ref, f"executed/{num:02d}_{slug}.ipynb")
        if old_nb_txt:
            cur_nb = json.load(open(ROOT / "executed" / f"{num:02d}_{slug}.ipynb"))
            if _exec_fingerprint(cur_nb) != _exec_fingerprint(json.loads(old_nb_txt)):
                print(f"[skip] Ch {num}: executed 실행본이 {args.old_ref} 대비 변경됨 "
                      f"— ④ 보존 생략(새로 작성 필요). 강행하려면 --force")
                return

    # 표준 페이지(개요+실습/해부/변형/정리)만 대상. 부록(`-data_scaling`·`-lambda_sweep`
    # 등 비표준 접미사)은 별도 노트북에서 만든 페이지라 건드리지 않는다.
    std_names = {f"{num:02d}-{slug}.md"} | {
        f"{num:02d}-{slug}-{suf}.md" for suf in ("practice", "anatomy", "variation", "wrapup")}

    # OLD: git ref 의 표준 NN-*.md 페이지만 이어붙여 ④ 추출
    old_names = subprocess.check_output(
        ["git", "show", f"{args.old_ref}:pages"], cwd=ROOT
    ).decode().splitlines()
    old_pages = [nm for nm in old_names if nm in std_names]
    old_text = "\n\n".join(filter(None, (git_show(args.old_ref, f"pages/{nm}")
                                         for nm in sorted(old_pages))))
    old_blocks = parse_blocks(old_text)
    enrich = extract_enrichment(old_blocks, code_norms, md_all)
    print(f"[OLD] ④ 추출: {len(enrich)} 코드셀 "
          f"(walk {sum(len(v['chunks'])>0 and sum(1 for _,w in v['chunks'] if w) for v in enrich.values())}, "
          f"A {sum(len(v['A']) for v in enrich.values())}, "
          f"C {sum(len(v['C']) for v in enrich.values())})")

    # NEW: 워킹트리의 표준 NN-*.md 각각에 적용 (부록 페이지 제외)
    stats = {"A": 0, "B": 0, "C": 0}
    new_pages = [p for p in sorted(PAGES.glob(f"{num:02d}-*.md")) if p.name in std_names]
    # 재빌드본(노트북 본문) 전체를 정규화해 (A) 중복 삽입 가드에 사용
    orig_norm = _norm("".join(p.read_text() for p in new_pages))
    for p in new_pages:
        new_text = p.read_text()
        merged = apply_enrichment(new_text, enrich, stats, orig_norm)
        if not args.dry_run:
            p.write_text(merged, encoding="utf-8")
    print(f"[NEW] 재부착: A={stats['A']} B(위 코드 읽기)={stats['B']} C(결과 해석)={stats['C']}"
          + ("  (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
