# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 레포 정체성

이 레포는 **Hugging Face 입문 커리큘럼 (18챕터)** 의 산출물을 담는 곳입니다. 코드 라이브러리가 아니라 **Google Colab 노트북 학습 자료** 가 결과물입니다. 챕터마다 레포 루트에 자체 폴더가 있고, 그 안에 노트북과 요약 README.md가 함께 들어갑니다 — 예: `01_tfidf/01_tfidf.ipynb` + `01_tfidf/README.md`. 노트북은 학습자가 **Google Colab T4 (16GB VRAM)** 에서 그대로 열어 30분 이내에 끝까지 실행할 수 있어야 합니다.

루트 `README.md`에 **챕터별 변화추적표 + Colab 열기 버튼** 이 있습니다. 이 표가 챕터 메타정보(파일명, Loss/Head 변화 등)의 **단일 출처(source of truth)** 입니다. 새 챕터를 만들거나 파일명을 바꾸면 README.md의 행도 함께 갱신해야 Colab 버튼이 깨지지 않습니다.

빌드/테스트 명령은 없습니다. 챕터 노트북의 검증은 "T4 Colab에서 30분 이내에 끝까지 돌아가는가"가 유일한 기준입니다.

## 절대 위반 금지 제약

1. **변경점 한 가지 원칙** — 한 챕터는 직전 챕터 대비 **딱 한 가지** 만 바뀝니다. 변화의 축은 셋:
   - **모델 축**: sklearn → DistilBERT → KLUE-BERT → 작은 BERT
   - **태스크 축**: Regression → Binary → Multi-class → Multi-label
   - **Loss 축**: MSELoss → BCEWithLogitsLoss → CrossEntropyLoss → BCEWithLogitsLoss(per-label) → +Auxiliary(Combined)

   한 챕터에서 한 축만 변하고 나머지는 고정. 두 축이 동시에 바뀌면 학습자가 효과를 분리해 이해할 수 없습니다. **Auxiliary는 task 신설이 아니라 loss에 보조 항을 더하는 변화** 이므로 Loss 축 끝에 위치합니다 — Ch 12/16의 메인 task는 직전 챕터(Multi-label)와 동일하고, 새 보조 헤드 + λ 가중치만 추가됩니다.
2. **T4 30분 제약** — 학습 코드는 30분 안에 끝나야 합니다. 기본 가이드: Yelp 5,000 샘플 / `max_length=128` / `batch_size=16-32` / 1-2 에폭 / `fp16=True`.
3. **bf16 사용 금지** — T4(Compute Capability 7.5)는 bf16 미지원. **항상 `fp16=True`**. Flash Attention 2도 미지원.
4. **용어 통일** — PyTorch/Hugging Face 용어를 메인으로, sklearn 용어는 괄호 안 보조로. 예: `BCEWithLogitsLoss` (sklearn: log loss).
5. **FAQ 답변까지 작성** — FAQ 섹션은 5-7개 질문 + **각 답변을 함께** 작성합니다. 필요하면 답변 안에 짧은 코드 스니펫(코드 레벨 안내)을 포함해 학습자가 바로 적용할 수 있게 합니다. 실무 : 이론 ≈ 6 : 4 비율은 유지.

## 챕터 표준 구조 (노트북 셀 순서)

각 `.ipynb`는 마크다운 셀과 코드 셀이 섞인 형태로, 다음 순서를 지킵니다. 빠뜨리는 섹션이 있으면 안 됩니다. 첫 셀은 Colab 환경 셋업(필요한 `pip install`, GPU 확인)이 들어갑니다.

1. **📊 추적표** — 현재 챕터까지의 누적 Loss/Output Head 변화 추적표 (마크다운 셀). 현재 챕터 행을 강조. 전체 표는 `README.md`를 참조.
2. **🔄 변경점 (Diff from Ch N-1)** — 이전 / 이번 / 왜 바꾸는가.
3. **📐 Loss 함수의 변화** — 해당 챕터에 변화가 있을 때만. 수식 + 직관 + 코드 한 줄 변화.
4. **🔤 토크나이저 노트** — **모든 챕터에 포함**. 같은 문장이 어떻게 토큰화되는지 예시, 다음 챕터에서 바뀌는지 여부.
5. **🚀 실습 → 🔬 해부 → 🛠️ 변형** — 3단 구조 ("먼저 돌려보기 → 안에서 무슨 일이 → 직접 재현"). 코드 셀로 구성.
6. **📦 등장한 라이브러리 정리** — 해당 챕터에서 새로 등장한 것만.
7. **🎯 체크포인트 질문** 3-4개.
8. **❓ FAQ** 5-7개 — 실무 : 이론 ≈ 6 : 4. 각 질문에 **답변까지** 작성. 필요한 곳은 짧은 코드 스니펫 포함.
9. **🚀 삽질 코너** (선택) — 일부러 틀린 코드로 에러 메시지 학습.
10. **다음 챕터 예고**.

## 전체 추적표 (Loss / Head / Tokenizer / Data)

18챕터의 변화 흐름과 Colab 링크가 담긴 단일 출처는 루트 `README.md`의 "챕터별 변화추적표" 입니다. 챕터를 작성할 때:
- 노트북 도입부 추적표는 README.md 표에서 **현재 챕터까지의 행만 잘라** 그대로 사용 (현재 행 강조).
- 챕터 추가/이름 변경/Loss 변화 시 README.md의 해당 행을 함께 갱신.
- 챕터 폴더 경로 규약: 레포 루트의 `<NN>_<slug>/` (zero-pad 두 자리). 폴더 안에 같은 이름의 노트북(`<NN>_<slug>.ipynb`)과 요약 `README.md`가 함께 들어감. **상위 `notebooks/` 디렉터리는 두지 않음** — 챕터 폴더가 루트에 직접 위치.

Phase 구분:
- **Phase 0 (Ch 1-5)**: sklearn으로 태스크/loss의 본질. BERT 등장하지 않음.
- **Phase 1 (Ch 6-12)**: DistilBERT(영어)로 같은 태스크들을 다시. Auxiliary loss로 마무리.
- **Phase 2 (Ch 13-16)**: 한국어로 압축 재방문 (klue/bert-base). **회귀 챕터는 생략** — 영어 Phase 1에서 이미 다뤘기 때문에 Binary부터 시작.
- **Phase 3 (Ch 17-18)**: 토크나이저를 직접 학습. 사전학습 의존 없는 경험. **Phase 3가 클라이맥스가 되도록 토크나이저 시각을 Ch 1부터 일관되게 추적.**

## 작업 흐름

1. **미정 사항 1건은 작성 시작 전 사용자와 확정**:
   - **Ch 16 보조 태스크 (한국어 Auxiliary)**: 라벨 개수 회귀(MSE)가 1순위 — Ch 12와 구조가 닮아 변경점이 "데이터 + 언어"로만 한정됨.
   - (참고) **한국어 회귀 챕터는 의도적으로 생략**됨. Phase 2는 Binary(Ch 13)부터 시작.
2. **챕터 단위 검증 사이클** — 한 챕터씩 다음 순서로 진행:
   1. Claude가 해당 챕터 폴더(`<NN>_<slug>/`)와 노트북(`<NN>_<slug>.ipynb`) 작성. 폴더 안 `README.md`에 챕터 요약을 함께 작성.
   2. **Claude가 곧바로 git commit + push.** Colab 버튼이 GitHub의 master 브랜치를 가리키므로 푸시 전엔 사용자가 노트북을 열 수 없음. 따라서 노트북 작성 직후 자동으로 커밋·푸시. 커밋 메시지는 챕터 단위로 의미 있게(`Add Ch <N>: <slug>` 등). 수정·재빌드 후에도 동일하게 커밋·푸시.
   3. 사용자가 README.md의 Colab 버튼으로 노트북을 열어 **직접 끝까지 실행해 검증**. T4에서 30분 내에 끝까지 정상 동작하는지가 합격 기준.
   4. 검증 통과 시 **루트 README.md 표의 `진행` 열을 `—` → `✅ Done`** 으로 갱신. 실패 시 사용자가 피드백을 주면 노트북 수정 → 다시 커밋·푸시. **`✅ Done`은 사용자 검증 후에만 갱신** — Claude가 임의로 표시하지 않음.
   5. 다음 챕터로 진행. 검증 미통과 챕터를 두고 다음 챕터로 넘어가지 않음.
3. Ch 6은 이미 작성된 영어판 베이스 파일 `_drafts/06_bert_pipeline_base.md`(원본 파일명: `01_pipeline_intro.md`)를 출발점으로. 내용은 그대로 살리고, 챕터 번호 보정·누적 추적표·🔤 토크나이저 노트·❓ FAQ를 추가해 Ch 6 노트북으로 변환.

## 톤과 표기

- **존댓말** (사용자 선호).
- 코드 주석은 한국어. 변수명/함수명은 영어.
- 라이브러리 이름은 백틱: `transformers`, `datasets`, `tokenizers`.
- 수식은 LaTeX: `$L = -\sum y \log p$`.
- 챕터 파일명: `<NN>_<slug>/<NN>_<slug>.ipynb` 두 자리 zero-pad.

## 두 방식 동등성 (Ch 9 핵심)

Ch 9는 binary를 **두 방식 모두** 다루고 동등성을 시연합니다:
- 방식 A: `num_labels=1`, sigmoid, `BCEWithLogitsLoss` (sklearn 호환)
- 방식 B: `num_labels=2`, softmax, `CrossEntropyLoss` (BERT 표준)
- z = z₁ - z₀로 두면 방식 B가 방식 A의 리파라미터화임을 보임.

이후 한국어 binary(Ch 13)는 방식 B만 사용 — 동등성을 이미 다뤘기 때문.

## `Trainer` `problem_type` 자동 매핑 (코드 작성 시 활용)

`AutoModelForSequenceClassification.from_pretrained(...)` 호출 시:
- `num_labels=1` 단독 → 회귀로 추정해 `MSELoss`. 명시적으로는 `problem_type="regression"`.
- `problem_type="single_label_classification"` → `CrossEntropyLoss` (기본값, multi-class 표준).
- `problem_type="multi_label_classification"` → `BCEWithLogitsLoss`. 라벨은 multi-hot float 텐서여야 함.
- Ch 12, 16의 보조 헤드처럼 자동 매핑을 못 쓸 때만 `Trainer.compute_loss` 오버라이드.
