#!/usr/bin/env bash
# executed/ 실행본을 google-colab-cli 로 로컬에서 자동 생성한다 (macOS/Linux 전용).
#
# 브라우저(run_on_colab.ipynb)에서 "모두 실행"을 손으로 누르는 대신, 터미널 한 줄로
# **챕터마다 새 VM** 할당 → 실행 → 결과 회수까지 한다. 결과를 로컬 executed/ 로 받으므로
# **GitHub PAT 가 필요 없다**(평소 git 으로 커밋).
#
# **챕터당 독립 VM** 으로 순차 실행한다(병렬 아님 — 동시 VM 1개). VM 격리로 한 챕터 실패가 다음에 안 번지고
# resume 가 단순하다.
#   ※ colab-cli keep-alive 버그(issue #14, v0.5.11 이하)가 있으면 VM 이 ~11분에 idle-prune 돼 무거운 챕터가
#     실패한다. issue #14 수정본(keep-alive 가 TFE 터널 핑)부터는 ~90분까지 유지돼 전 챕터(예: 25·27, 19~24분)가 완주한다.
#
# 사용 (REPO 는 git origin 에서 자동 인식 — 보통 그냥 실행하면 됨)
#   ./executed/run_via_cli.sh               # 인자 없음 → 전 챕터
#   ./executed/run_via_cli.sh 7 24          # 해당 챕터만 (번호 또는 폴더명)
#   ./executed/run_via_cli.sh 07_bert_pipeline
#   FORCE=1 ./executed/run_via_cli.sh 7     # 로컬에 ok 여도 강제 재실행
#   GPU=L4 ./executed/run_via_cli.sh        # GPU 종류 변경
#   REPO=other/repo ./executed/run_via_cli.sh   # origin 외 다른 저장소를 쓸 때만
#
# 동작
#   - **resume**: 로컬에 이미 status=ok 인 챕터는 skip(FORCE=1 로 무시). 끊겨도 다시 돌리면 이어서.
#   - **재시도**: 일시 드롭(Connection lost)은 챕터당 1회 자동 재시도.
#   - **OVER_CAP**(기본 빈 값): 옛 버전(keep-alive 미수정) 안전장치 — 전체 실행 시 특정 챕터를 스킵하려면
#     OVER_CAP 에 폴더명을 넣는다(인자로 명시하면 무시하고 시도). 수정본에선 불필요(전 챕터 완주).
#   - VM 은 챕터마다 끝나면 자동 종료 → 컴퓨트 유닛 낭비 없음.
#
# 사전 준비(최초 1회)
#   1) 설치:  uv tool install "git+https://github.com/googlecolab/google-colab-cli"
#        ※ issue #14(keep-alive) 수정 포함 버전 필요. PyPI 최신(v0.5.11)엔 아직 미반영 → 위처럼 git 에서 설치.
#          수정이 릴리스되면 `uv tool install google-colab-cli` 로 충분.
#   2) 인증:  colab --auth=oauth2 whoami           # 동의 화면 "모두 선택"(refresh 때 6스코프 전체 요구).
#        - 과금 방지로 **결제수단 없는 무료 계정** 권장. 깨진 토큰: rm ~/.config/colab-cli/token.json 후 재인증.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"   # executed/
ROOT="$(cd "$HERE/.." && pwd)"          # 레포 루트
EXECPY="$HERE/colab_cli_exec.py"
OUTDIR="$HERE"

# REPO: 미지정 시 로컬 git `origin` 리모트에서 자동 인식(CLI 는 VM 에서 clone 만 하므로 origin 이면 충분).
#   다른 저장소를 쓰려면 REPO=owner/repo 로 override.
REPO="${REPO:-}"
if [ -z "$REPO" ]; then
    REPO="$(git -C "$ROOT" remote get-url origin 2>/dev/null | sed -E 's#\.git/?$##; s#^.*github\.com[:/]+##')"
fi
BRANCH="${BRANCH:-master}"
GPU="${GPU:-T4}"                  # T4 | L4 | G4 | H100 | A100
FORCE="${FORCE:-}"               # 비어있지 않으면 로컬 ok 여도 재실행
# 전체 실행 시 스킵할 챕터(공백 구분 폴더명). 기본 빈 값 — issue #14 수정본에선 모든 챕터가 완주한다.
# 옛 버전(keep-alive 미수정)을 쓸 거면 11분 초과 챕터를 여기 지정: OVER_CAP="25_gpt2_continual_pretrain 27_ko_gpt2_continual_pretrain"
OVER_CAP="${OVER_CAP:-}"

printf '%s' "$REPO" | grep -qE '^[^/]+/[^/]+$' || {
    echo "✗ REPO 를 자동 인식하지 못했습니다(git origin 확인). 직접 지정: REPO=you/neuqes-101 $0 [챕터...]"; exit 1; }
command -v colab >/dev/null 2>&1 || { echo "✗ colab-cli 미설치 → uv tool install \"git+https://github.com/googlecolab/google-colab-cli\" (issue #14 수정본)"; exit 1; }
[ -f "$EXECPY" ] || { echo "✗ $EXECPY 없음(같은 폴더의 colab_cli_exec.py 필요)"; exit 1; }

# 어떤 경로로 끝나든 진행 중 VM 은 정리(유닛/시간 소모 방지).
CUR_SESSION=""
cleanup() { [ -n "$CUR_SESSION" ] && colab stop -s "$CUR_SESSION" >/dev/null 2>&1 || true; }
trap cleanup EXIT

is_ok() {  # 로컬에 유효 JSON + executed_from.status=ok 면 0
    python3 -c 'import json,sys
try:
    nb=json.load(open(sys.argv[1])); sys.exit(0 if nb["metadata"].get("executed_from",{}).get("status")=="ok" else 1)
except Exception: sys.exit(1)' "$1" 2>/dev/null
}

resolve() {  # 인자(번호/폴더명) → 챕터 폴더명. 못 찾으면 1.
    local spec="$1" nn m
    [ -d "$ROOT/$spec" ] && { echo "$spec"; return 0; }
    if printf '%s' "$spec" | grep -qE '^[0-9]+$'; then
        nn=$(printf '%02d' "$spec")
        m=$(cd "$ROOT" && ls -d "${nn}"_*/ 2>/dev/null | head -1 | sed 's#/##')
        [ -n "$m" ] && { echo "$m"; return 0; }
    fi
    return 1
}

run_one() {  # $1=folder — 새 VM 1개로 한 챕터 실행 → executed/<folder>.ipynb 회수 → VM 종료
    local folder="$1" sess b64
    sess="wd-$(echo "$folder" | tr '_' '-')"
    CUR_SESSION="$sess"
    # --keep: run 자동 stop 전에 download 해야 함. --timeout 120: 실행기 하트비트(20s)로 정상 챕터는
    #   안 끊기고, VM 이 죽으면 출력이 멈춰 이 시간 안에 빠르게 실패(장시간 hang 방지). --force: VM 클론의 기존 실행본 무시.
    colab run --keep -s "$sess" --gpu "$GPU" --timeout 120 \
        "$EXECPY" "$REPO" "$BRANCH" "$folder" --force \
        || echo "  (colab run 비정상 종료)"
    # .ipynb 대신 base64 사본을 받아 로컬 디코드(colab download 의 .ipynb→dict→repr 깨짐 회피).
    b64="$(mktemp)"
    colab download -s "$sess" "/content/${REPO##*/}/executed/$folder.ipynb.b64" "$b64" >/dev/null 2>&1 \
        && python3 -c 'import base64,sys;open(sys.argv[2],"wb").write(base64.b64decode(open(sys.argv[1],"rb").read()))' "$b64" "$OUTDIR/$folder.ipynb" \
        || true
    rm -f "$b64"
    colab stop -s "$sess" >/dev/null 2>&1 || true
    CUR_SESSION=""
}

# 대상 챕터 목록 — 인자 있으면 그 챕터만(EXPLICIT), 없으면 전체.
chapters=()
if [ "$#" -gt 0 ]; then
    EXPLICIT=1
    for spec in "$@"; do
        if f=$(resolve "$spec"); then chapters+=("$f"); else echo "⚠️ 알 수 없는 챕터 인자: $spec (건너뜀)"; fi
    done
else
    EXPLICIT=0
    while IFS= read -r f; do chapters+=("$f"); done < <(cd "$ROOT" && ls -d [0-9][0-9]_*/ 2>/dev/null | sed 's#/##' | sort)
fi
total=${#chapters[@]}
[ "$total" -gt 0 ] || { echo "✗ 대상 챕터가 없습니다(인자 확인, 또는 레포 루트에 NN_slug/ 폴더가 있는지)"; exit 1; }

mode=$([ "$EXPLICIT" -eq 1 ] && echo "지정 ${total}개" || echo "전체 ${total}개")
echo "===== CLI 실행 ($mode, repo=$REPO@$BRANCH, $(date '+%H:%M:%S')) ====="

ok=0; skip=0; fail=0; i=0
for folder in "${chapters[@]}"; do
    i=$((i + 1))
    if [ -z "$FORCE" ] && is_ok "$OUTDIR/$folder.ipynb"; then
        echo "[$i/$total] $folder — 이미 ok, skip (FORCE=1 로 강제)"; skip=$((skip + 1)); continue
    fi
    if [ "$EXPLICIT" -eq 0 ] && echo " $OVER_CAP " | grep -q " $folder "; then
        echo "[$i/$total] $folder — 11분 캡 초과(OVER_CAP), 자동 스킵 → 브라우저 러너/원본 실행본으로 충당"; skip=$((skip + 1)); continue
    fi
    echo ""
    echo "===== [$i/$total] $folder — 새 VM ($(date '+%H:%M:%S')) ====="
    run_one "$folder"
    if ! is_ok "$OUTDIR/$folder.ipynb"; then
        echo "  ↻ $folder 1차 미완 — 일시 드롭 대비 1회 재시도 ($(date '+%H:%M:%S'))"
        run_one "$folder"
    fi
    if is_ok "$OUTDIR/$folder.ipynb"; then
        echo "  ✓ $folder"; ok=$((ok + 1))
    else
        echo "  ✗ $folder — 미완(11분 초과 또는 연결 끊김; 2회 시도)"; fail=$((fail + 1))
    fi
done

echo ""
echo "===== 종료 ($(date '+%H:%M:%S')): ok=$ok skip=$skip fail=$fail / total=$total ====="
echo "로컬 executed/*.ipynb:"; ls -1 "$OUTDIR"/[0-9][0-9]_*.ipynb 2>/dev/null | sed 's#.*/#  #'
echo "skip/fail(무거운) 챕터는 브라우저 러너(run_on_colab.ipynb)나 원본 저장소 실행본으로 충당하세요."
echo "이어서 변환:  python3 .claude/skills/notebook-to-wikidocs/scripts/build_wikidocs.py <챕터>"
