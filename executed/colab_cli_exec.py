#!/usr/bin/env python3
"""Colab VM 위에서 도는 실행본 생성기 — colab-cli `colab run` 의 대상 스크립트.

로컬 래퍼 `run_via_cli.sh` 가 다음처럼 호출한다(직접 실행할 일은 없음):

    colab run --keep -s <세션> --gpu T4 --timeout 36000 \
        colab_cli_exec.py <REPO> <BRANCH> <TARGET> [--force]

`colab run` 이 이 파일을 T4 VM 커널에 올려 `__main__` 으로 실행하면:
  1) 본인 fork 를 clone (REPO 는 본인 fork — 원본 push 권한 불필요)
  2) 대상 챕터를 NotebookClient 로 끝까지 실행
  3) 출력 포함 `executed/<폴더>.ipynb` 를 VM 에 저장(실패해도 다음 챕터 계속)
  4) 다운로드할 폴더 목록을 `/content/wikidocs_manifest.txt` 에 기록
그 뒤 래퍼가 `colab download` 로 결과를 로컬로 가져온다.

`run_on_colab.ipynb` 의 실행 로직(소스 해시 멱등·`executed_from` 도장·챕터별 소요시간)을
그대로 미러링한다. 차이는 단 하나: **GitHub PAT push 대신 결과를 로컬로 download → 로컬에서
커밋**한다(PAT 불필요). 둘 중 한쪽 로직을 고치면 다른 쪽도 함께 맞춘다.

TARGET:
  stale          없거나 clean 노트북이 바뀐 챕터만 (기본)
  all            전 챕터
  gpu            07 이상(GPU) 전부
  "1,7,24"       쉼표 목록 — 번호 또는 폴더명("24_gpt_tinystories")
"""
import sys
import base64
import subprocess
import hashlib
import datetime
import time
import threading
from pathlib import Path

PER_CELL_TIMEOUT = 60 * 60  # 셀당 최대 실행 시간(초). GPU 학습 챕터 여유 있게.


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *pkgs], check=True)


def fmt_dur(sec):
    """초 → '3분 5초' / '42초' 형태."""
    m, s = divmod(int(round(sec)), 60)
    return f"{m}분 {s}초" if m else f"{s}초"


def main():
    if len(sys.argv) < 3:
        print("usage: colab_cli_exec.py <REPO> <BRANCH> [TARGET] [--force]", file=sys.stderr)
        sys.exit(2)
    repo = sys.argv[1]
    branch = sys.argv[2]
    target = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "stale"
    force = "--force" in sys.argv

    _pip("nbclient", "nbformat")
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    # 1) 본인 fork clone (이미 있으면 최신화)
    work = Path("/content") / repo.split("/")[-1]
    if not work.is_dir():
        # 얕은 클론(--depth 1) — 챕터별 새 VM 방식에서 클론 비용을 줄인다. 히스토리 불필요.
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", branch,
                        f"https://github.com/{repo}.git", str(work)], check=True)
    else:
        subprocess.run(["git", "-C", str(work), "checkout", "-q", branch], check=True)
        subprocess.run(["git", "-C", str(work), "pull", "-q", "--depth", "1"], check=False)

    exec_dir = work / "executed"
    exec_dir.mkdir(exist_ok=True)

    # 2) 챕터 탐색 + 해시(멱등 판단) — run_on_colab.ipynb 와 동일
    def chapters():
        out = []
        for d in sorted(work.glob("[0-9][0-9]_*")):
            nb = d / (d.name + ".ipynb")
            if nb.exists():
                out.append((d.name, nb))
            # 부록 노트북(<폴더>/<폴더>_*.ipynb) — 키는 노트북 stem
            for sub in sorted(d.glob(d.name + "_*.ipynb")):
                out.append((sub.stem, sub))
        return out

    def source_hash(nb_path):
        """clean 노트북의 셀 소스(출력 제외) 해시 — 내용이 바뀌면 달라진다."""
        nb = nbformat.read(nb_path, as_version=4)
        h = hashlib.sha256()
        for c in nb.cells:
            h.update(c.cell_type.encode()); h.update(b"\0")
            h.update((c.source or "").encode()); h.update(b"\0")
        return h.hexdigest()

    def executed_hash(folder):
        p = exec_dir / (folder + ".ipynb")
        if not p.exists():
            return None
        try:
            nb = nbformat.read(p, as_version=4)
            return nb.metadata.get("executed_from", {}).get("source_sha256")
        except Exception:
            return None

    def is_stale(folder, nb_path):
        return source_hash(nb_path) != executed_hash(folder)

    all_ch = chapters()
    print(f"발견한 챕터: {len(all_ch)}", flush=True)

    # 3) 대상 결정
    def base_set(t):
        if t == "all":
            return all_ch
        if t == "gpu":
            return [(f, p) for f, p in all_ch if int(f[:2]) >= 7]
        if t == "stale":
            return all_ch  # staleness 필터는 아래에서
        # 쉼표 목록(번호 또는 폴더명)
        keys = {x.strip().zfill(2) if x.strip().isdigit() else x.strip()
                for x in t.split(",") if x.strip()}
        return [(f, p) for f, p in all_ch if f in keys or f[:2] in keys]

    base = base_set(target)
    sel = base if force else [(f, p) for f, p in base if is_stale(f, p)]
    print(f"실행 대상: {len(sel)}개  (TARGET={target!r}, FORCE={force})", flush=True)

    # 4) 실행 → executed/<폴더>.ipynb 저장 (실패해도 다음 챕터 계속)
    written = []
    total_t0 = time.time()
    for f, p in sel:
        print(f"\n=== 실행: {f} ===", flush=True)
        nb = nbformat.read(p, as_version=4)
        client = NotebookClient(
            nb, timeout=PER_CELL_TIMEOUT, kernel_name="python3",
            resources={"metadata": {"path": str(p.parent)}},  # 챕터 폴더에서 실행
            allow_errors=False,
        )
        status = "ok"
        t0 = time.time()
        # 하트비트 — nbclient 가 자식 커널에서 챕터를 돌리는 동안 이 부모 커널은 무출력이라,
        # 원격 실행 도구(colab run/exec)가 "죽은 세션"과 "조용히 일하는 중"을 구분 못 한다.
        # 20초마다 한 줄 찍어 살아있음을 알린다 → 작은 --timeout 으로도 정상 챕터가 안 끊기고,
        # VM 이 죽으면 출력이 멈춰 그 timeout 안에 빠르게 실패한다(장시간 hang 방지).
        stop_hb = threading.Event()

        def _heartbeat():
            n = 0
            while not stop_hb.wait(20):
                n += 1
                print(f"  ··· {f} 실행 중 ({n * 20}s)", flush=True)

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()
        try:
            client.execute()
        except CellExecutionError as e:
            status = "error: " + str(e).splitlines()[-1][:120]
            print("  ⚠️", status, flush=True)
        except Exception as e:  # 커널 타임아웃 등
            status = "fail: " + str(e)[:120]
            print("  ⚠️", status, flush=True)
        finally:
            stop_hb.set()
        elapsed = time.time() - t0
        nb.metadata["executed_from"] = {
            "source_sha256": source_hash(p),
            "executed_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "runtime": "colab-t4",
            "status": status,
            "elapsed_sec": round(elapsed, 1),
        }
        nbformat.write(nb, exec_dir / (f + ".ipynb"))
        # 전송용 base64 사본 — colab download 는 .ipynb 를 Jupyter Contents API 가 dict 로
        # 돌려줘 repr 로 깨뜨린다(작은따옴표). base64(평문)로 받아 로컬에서 무손실 디코드한다.
        (exec_dir / (f + ".ipynb.b64")).write_text(
            base64.b64encode(nbformat.writes(nb).encode("utf-8")).decode("ascii"))
        written.append((f, status, elapsed))
        print(f"  → 저장 executed/{f}.ipynb  [{status}]  ⏱ {fmt_dur(elapsed)}", flush=True)

    total_elapsed = time.time() - total_t0
    print("\n=== 요약 (챕터별 소요 시간) ===", flush=True)
    for f, s, dt in written:
        print(f"  {f:<28}{fmt_dur(dt):>10}   {s}", flush=True)
    print(f"  {'합계 (' + str(len(written)) + '개 챕터)':<28}{fmt_dur(total_elapsed):>10}", flush=True)

    # 5) 다운로드 목록 — 래퍼가 이 파일을 읽어 colab download 한다(없는 챕터도 빈 파일로 남겨 download 실패 방지)
    manifest = Path("/content/wikidocs_manifest.txt")
    manifest.write_text("\n".join(f for f, _, _ in written) + ("\n" if written else ""))
    print(f"\nMANIFEST {manifest} ({len(written)}개)", flush=True)


if __name__ == "__main__":
    main()
