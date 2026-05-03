"""Copy a chapter notebook to _local/ with Mac-friendly patches applied.

The original .ipynb stays untouched — patched copies live under `_local/`,
which is gitignored. Open the patched copy in JupyterLab or VS Code
(kernel: `neuqes-101 (3.12, MPS)`) for local Mac verification.

Patches applied to code cells:
- `fp16=True` → `fp16=False`        (HF Trainer would auto-disable on MPS,
                                     but explicit is cleaner and silences a
                                     warning)
- `!nvidia-smi` → `!command -v` 가드 (Mac엔 nvidia-smi 없음 → 친절 메시지)
- `!pip install ...` → comment-out  (venv에 이미 설치돼 있음)

Optional with `--quick`:
- `range(N)` (N≥100) → `range(N//10)` for fast iteration
- `num_train_epochs=N` → `num_train_epochs=1`

Examples:
    python3 _drafts/_local_patch.py 14            # Ch 14 only
    python3 _drafts/_local_patch.py 14 --quick    # + reduce sizes/epochs
    python3 _drafts/_local_patch.py all           # every chapter
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCAL_DIR = REPO / "_local"


def patch_cell_source(src: str, quick: bool) -> str:
    """Apply Mac-friendly patches to one code cell's source string."""
    # 1. fp16 — Mac MPS에서 비활성
    src = src.replace("fp16=True", "fp16=False")

    # 2. !pip install ... → comment (venv에 이미 설치)
    src = re.sub(
        r"^(\s*)!pip install (.*)$",
        r"\1# (skipped in _local) !pip install \2",
        src,
        flags=re.MULTILINE,
    )

    # 3. !nvidia-smi → graceful fallback (해당 라인 통째 교체)
    src = re.sub(
        r"^(\s*)!nvidia-smi.*$",
        r"\1!command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo '(no nvidia-smi - local Mac)'",
        src,
        flags=re.MULTILINE,
    )

    if quick:
        # 4. .select(range(N)) — N이 100 이상이면 1/10 로 축소 (최소 50)
        def shrink_range(m: re.Match) -> str:
            n = int(m.group(1))
            return f".select(range({max(50, n // 10)}))" if n >= 100 else m.group(0)

        src = re.sub(r"\.select\(range\((\d+)\)\)", shrink_range, src)

        # 5. num_train_epochs=N → num_train_epochs=1
        src = re.sub(r"num_train_epochs\s*=\s*\d+", "num_train_epochs=1", src)

    return src


SETUP_CELL_SOURCE = (
    "# === _local Mac compatibility setup (auto-injected by _drafts/_local_patch.py) ===\n"
    "# macOS + ipykernel + safetensors mmap = SIGBUS during model loading on M-series.\n"
    "# Force PreTrainedModel.from_pretrained to default use_safetensors=False so all\n"
    "# from_pretrained / pipeline() calls below load via .bin (no mmap).\n"
    "import os\n"
    "os.environ.setdefault('SAFETENSORS_FAST_GPU', '0')\n"
    "import transformers as _tf\n"
    "_orig_fp = _tf.PreTrainedModel.from_pretrained.__func__\n"
    "def _patched_fp(cls, *args, **kw):\n"
    "    kw.setdefault('use_safetensors', False)\n"
    "    return _orig_fp(cls, *args, **kw)\n"
    "_tf.PreTrainedModel.from_pretrained = classmethod(_patched_fp)\n"
    "del _patched_fp, _tf"
)


def patch_notebook(nb_path: Path, out_path: Path, quick: bool) -> dict:
    """Read .ipynb, patch code cells, write to out_path. Return per-pattern hit counts."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    stats = {"cells_patched": 0, "fp16": 0, "pip": 0, "nvidia": 0, "range": 0, "epoch": 0,
             "setup_injected": 0}

    # Inject the setup cell at the very top so the monkey-patch runs before any imports
    setup_cell = {
        "cell_type": "code",
        "id": "local_setup",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": SETUP_CELL_SOURCE,
    }
    nb["cells"].insert(0, setup_cell)
    stats["setup_injected"] = 1

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)

        before = src
        if "fp16=True" in src:
            stats["fp16"] += 1
        if re.search(r"^\s*!pip install", src, re.MULTILINE):
            stats["pip"] += 1
        if re.search(r"^\s*!nvidia-smi", src, re.MULTILINE):
            stats["nvidia"] += 1
        if quick and re.search(r"\.select\(range\(\d+\)\)", src):
            stats["range"] += 1
        if quick and re.search(r"num_train_epochs\s*=\s*\d+", src):
            stats["epoch"] += 1

        new_src = patch_cell_source(src, quick)
        if new_src != before:
            cell["source"] = new_src
            cell["outputs"] = []
            cell["execution_count"] = None
            stats["cells_patched"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return stats


def find_chapters(pattern: str) -> list[Path]:
    """Find chapter notebooks matching pattern. 'all' = every <NN>_<slug>/<NN>_<slug>.ipynb."""
    if pattern == "all":
        return sorted(REPO.glob("[0-9][0-9]_*/[0-9][0-9]_*.ipynb"))

    # Try direct prefix first: 14 → 14_*/14_*.ipynb
    matches = sorted(REPO.glob(f"{pattern}*/{pattern}*.ipynb"))
    if matches:
        return matches

    # Fallback — any folder containing the pattern as substring
    return sorted(p for p in REPO.glob("[0-9][0-9]_*/*.ipynb") if pattern in p.parent.name)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Mac-friendly patcher: copies chapter notebooks to _local/ with safe overrides.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("chapter", help="chapter prefix (e.g., '14', '14_auxiliary') or 'all'")
    ap.add_argument(
        "--quick", action="store_true",
        help="also shrink .select(range(N)) and num_train_epochs for fast iteration",
    )
    args = ap.parse_args(argv)

    matches = find_chapters(args.chapter)
    if not matches:
        print(f"[error] no chapter notebook matching '{args.chapter}'", file=sys.stderr)
        return 1

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    mode = "quick" if args.quick else "full"
    print(f"Patching {len(matches)} notebook(s) into _local/ ({mode}):")
    for nb_path in matches:
        rel = nb_path.relative_to(REPO)
        out_path = LOCAL_DIR / rel
        s = patch_notebook(nb_path, out_path, args.quick)
        print(f"  {rel}")
        print(f"    → {out_path.relative_to(REPO)}")
        print(f"    cells_patched={s['cells_patched']}  "
              f"fp16={s['fp16']} pip={s['pip']} nvidia={s['nvidia']}"
              + (f" range={s['range']} epoch={s['epoch']}" if args.quick else ""))

    print()
    print("Open with one of:")
    if len(matches) == 1:
        rel_local = (LOCAL_DIR / matches[0].relative_to(REPO)).relative_to(REPO)
        print(f"  .venv/bin/jupyter lab {rel_local}")
        print(f"  code {rel_local}   # then 'Select Kernel' → neuqes-101 (3.12, MPS)")
    else:
        print(f"  .venv/bin/jupyter lab _local/")
        print(f"  code _local/       # then open the .ipynb you want")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
