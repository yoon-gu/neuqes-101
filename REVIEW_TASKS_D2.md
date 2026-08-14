# D2 검수 작업 목록 (2026년 9월 출판물 검수)

> 기준 문서: [REVIEW_PLAN_2026-09.md](./REVIEW_PLAN_2026-09.md)
>
> 담당자: D2
>
> 작성일: 2026-08-14

이 문서는 전체 검수 계획(`REVIEW_PLAN_2026-09.md`) 중 D2가 실제로 처리해야 하는 항목만 뽑아 개인 체크리스트로 정리한 것입니다. 진행 현황의 단일 출처는 여전히 GitHub Project이며, 이 문서는 D2 개인의 작업 순서를 놓치지 않기 위한 보조 자료입니다.

## 1. 담당 범위 요약

- **기본 담당** (페어 D, D1과 함께 독립 검수): Ch 27~34, 8챕터 × 3형식(Notebook/WikiDocs/EPUB) = **24건**.
- **3차 검수 담당**: Ch 25 (기본 담당은 페어 C) — 3형식 = **3건**.
- 개인 총 검수 건수: **27건**.
- 기본 담당 챕터끼리는 D2가 3차 검수자로 배정되지 않습니다 (3차 검수자는 항상 기본 담당과 다른 페어).

## 2. 기본 담당 체크리스트 (Ch 27~34)

> D2 결정: Ch 27~30을 원래 계획(2주 분량, 8/3~8/16)과 달리 **이번 주(8/10~8/16)에 한 번에 진행**합니다. 원래 일정보다 부하가 큰 주이므로, 금요일 주간 회의에서 `Actual Hours`를 확인해 Ch 31 이후 일정을 당길지 그대로 둘지 판단합니다(계획서 §3: 소요시간이 15% 이상 벌어지면 조정).

| 주차 | 챕터 | 폴더 / 내용 | Notebook | WikiDocs | EPUB | 비고 |
|---|---|---|---|---|---|---|
| **8/10~8/16 (이번 주)** | Ch 27 | `27_ko_gpt2_continual_pretrain` — KoGPT2 continual pretraining | [ ] | [ ] | [ ] | |
| **8/10~8/16 (이번 주)** | Ch 28 | `28_sft` — KoGPT2 SFT | [ ] | [ ] | [ ] | |
| **8/10~8/16 (이번 주)** | Ch 29 | `29_benchmark_eval` — 영어/한국어 분야별 벤치마크 평가 | [ ] | [ ] | [ ] | 평가 노트북 — 실행 시간 특히 확인 |
| **8/10~8/16 (이번 주)** | Ch 30 | `30_dpo` — DPO (`DPOTrainer`) | [ ] | [ ] | [ ] | |
| 8/17~8/23 | Ch 31 | `31_grpo` — GRPO (`GRPOTrainer`) | [ ] | [ ] | [ ] | |
| 8/24~8/30 | Ch 32 | `32_diffusion_intro` — Diffusion 패러다임 (기본 샘플러) | [ ] | [ ] | [ ] | |
| 8/31~9/6 | Ch 33 | `33_diffusion_train` — Diffusion 샘플러(carry-over + 반복억제) | [ ] | [ ] | [ ] | |
| 9/7~9/13 | Ch 34 | `34_ko_diffusion` — 한국어 Diffusion (80/10/10 마스킹) | [ ] | [ ] | [ ] | |

각 칸은 D2 본인의 PASS 여부입니다. D1의 PASS와 별개로 독립적으로 기록합니다(같이 읽고 한 번만 기록하지 않음).

## 2-1. 산출물 파이프라인과 검수 시 확인할 것 (Ch 27~30 공통)

세 산출물(Notebook/WikiDocs/EPUB)은 하나의 원본에서 순서대로 파생됩니다. 검수할 때 이 순서를 거슬러 확인하면 됩니다.

| 단계 | 위치 | 설명 |
|---|---|---|
| ① 원본 노트북 | `<NN>_<slug>/<NN>_<slug>.ipynb` (예: `27_ko_gpt2_continual_pretrain/27_ko_gpt2_continual_pretrain.ipynb`) | 루트 README의 Colab 배지가 여기로 연결됩니다. **Notebook 검수 대상** — 실제 Colab T4에서 30분 내 끝까지 실행되는지 확인합니다. |
| ② 실행본 | `executed/<NN>_<slug>.ipynb` (예: `executed/27_ko_gpt2_continual_pretrain.ipynb`) | 원본을 실행해 출력 셀까지 채운 버전. `colab-cli`를 설치·로그인해뒀다면 `executed/run_via_cli.sh`로 브라우저 없이 CLI에서 실행할 수 있습니다 ([google-colab-cli](https://github.com/googlecolab/colab-cli) 참고). |
| ③ WikiDocs 산출물 | `pages/<NN>-<slug>*.md` + `TOC.md` | 실행본을 `.claude/skills/notebook-to-wikidocs` 스킬로 변환한 결과. **WikiDocs 검수 대상.** Ch 27~30은 이미 `pages/27-ko_gpt2_continual_pretrain*.md` ~ `pages/30-dpo*.md`와 `TOC.md`의 해당 항목이 존재합니다. |
| ④ EPUB | 위 WikiDocs 산출물을 빌드 | **EPUB 검수 대상.** 빌드 시점의 소스가 최신 원본/실행본과 일치하는지 확인합니다. |

**검수 중 수정이 필요할 때의 순서** (계획서 §9의 Blocker/Major 처리에도 적용):

1. 원본 노트북(①)을 먼저 고칩니다.
2. `executed/run_via_cli.sh` 등으로 재실행해 실행본(②)을 갱신합니다.
3. `notebook-to-wikidocs` 스킬로 `pages/*.md`와 `TOC.md`(③)를 다시 생성합니다.
4. 필요 시 EPUB(④)을 다시 빌드합니다.

역순으로(예: WikiDocs md만 손으로 고치고 원본은 그대로 두는 식) 고치면 다음 재변환 때 수정 내용이 조용히 사라집니다 — 실제로 `7c6eb0a` 커밋이 "재변환 시 TOC.md 부록 항목이 조용히 삭제되던 버그"를 다룬 적이 있으니 주의합니다. 챕터 이슈의 `Source Revision` 필드는 항상 ①을 고친 커밋 SHA와 일치해야 합니다.

## 3. 3차 검수 담당 (Ch 25)

- 기본 담당: 페어 C, 예정 주차 8/31~9/6.
- 3차 검수(D2)는 페어 C의 기본 검수가 끝난 뒤 시작하며, 원칙상 **다음 주 수요일(9/9)까지 완료**합니다.
- 대상: `25_gpt2_continual_pretrain` — GPT2 continual pretraining (영어, Ch 27의 영어판 대응 챕터).

| 항목 | 상태 |
|---|---|
| Notebook 3차 PASS | [ ] |
| WikiDocs 3차 PASS | [ ] |
| EPUB 3차 PASS | [ ] |

## 4. 검수 기록 남기는 형식

챕터 검수 이슈에 아래 형식으로 결과를 남깁니다 (`REVIEW_PLAN_2026-09.md` §8).

```text
Notebook 1차 PASS
검수자: @<D2의 GitHub 핸들>
기준 커밋: abc1234
환경: Google Colab T4
실행 시간: 24분
발견 이슈: #123, #124
```

문제를 발견하면 챕터 이슈 댓글에 묻어두지 않고 결함 이슈를 별도로 만듭니다.

```text
[Ch 30][Notebook][Major] DPO loss 수식과 코드 구현 불일치
```

## 5. 완료 조건 (D2 개인 관점)

- [ ] 기본 담당 24건 모두 PASS 기록 또는 결함 이슈로 등록
- [ ] Ch 25의 Notebook/WikiDocs/EPUB 3차 검수 PASS
- [ ] 본인이 발견한 Blocker·Major가 모두 재검수까지 완료됨
- [ ] 매주 금요일 `Actual Hours` 필드 갱신

## 6. 참고

- 전체 계획, 일정, GitHub Project 구성: [REVIEW_PLAN_2026-09.md](./REVIEW_PLAN_2026-09.md)
- 심각도 기준(Blocker/Major/Minor)과 처리 기한: 위 문서 §9
- 라벨 규칙: 위 문서 §7
- 검수 기준 사례(과거 결함 사례 정리): `REVIEW_GUIDE_2026-09.pdf`
