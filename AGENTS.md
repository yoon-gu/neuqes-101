# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 레포 정체성

이 레포는 **Hugging Face 입문 커리큘럼 (19챕터)** 의 산출물을 담는 곳입니다. 코드 라이브러리가 아니라 **Google Colab 노트북 학습 자료** 가 결과물입니다. 챕터마다 레포 루트에 자체 폴더가 있고, 그 안에 노트북과 요약 README.md가 함께 들어갑니다 — 예: `01_tfidf/01_tfidf.ipynb` + `01_tfidf/README.md`. 노트북은 학습자가 **Google Colab T4 (16GB VRAM)** 에서 그대로 열어 30분 이내에 끝까지 실행할 수 있어야 합니다.

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
3. **📐 Loss 함수의 변화** — 해당 챕터에 변화가 있을 때만. 수식 + **수치 예시 2-3개(작은 표 권장)** + 직관 + 코드 한 줄 변화. 수치 예시는 그 loss의 핵심 성질(MSE의 비선형 페널티, BCE의 확률 양극단 폭증 등)이 드러나도록 고른다.
4. **🔤 토크나이저 노트** — **모든 챕터에 포함**. 같은 문장이 어떻게 토큰화되는지 예시, 다음 챕터에서 바뀌는지 여부.
5. **🚀 실습 → 🔬 해부 → 🛠️ 변형** — 3단 구조 ("먼저 돌려보기 → 안에서 무슨 일이 → 직접 재현"). 코드 셀로 구성.
6. **📦 등장한 라이브러리 정리** — 해당 챕터에서 새로 등장한 것만.
7. **🎯 체크포인트 질문** 3-4개.
8. **❓ FAQ** 5-7개 — 실무 : 이론 ≈ 6 : 4. 각 질문에 **답변까지** 작성. 필요한 곳은 짧은 코드 스니펫 포함.
9. **🚀 삽질 코너** (선택) — 일부러 틀린 코드로 에러 메시지 학습.
10. **다음 챕터 예고**.

## 전체 추적표 (Loss / Head / Tokenizer / Data)

19챕터의 변화 흐름과 Colab 링크가 담긴 단일 출처는 루트 `README.md`의 "챕터별 변화추적표" 입니다. 챕터를 작성할 때:
- 노트북 도입부 추적표는 README.md 표에서 **현재 챕터까지의 행만 잘라** 그대로 사용 (현재 행 강조).
- 챕터 추가/이름 변경/Loss 변화 시 README.md의 해당 행을 함께 갱신.
- 챕터 폴더 경로 규약: 레포 루트의 `<NN>_<slug>/` (zero-pad 두 자리). 폴더 안에 같은 이름의 노트북(`<NN>_<slug>.ipynb`)과 요약 `README.md`가 함께 들어감. **상위 `notebooks/` 디렉터리는 두지 않음** — 챕터 폴더가 루트에 직접 위치.

Phase 구분:
- **Phase 0 (Ch 1-6)**: sklearn으로 태스크/loss의 본질. BERT 등장하지 않음. (Ch 4는 sigmoid↔softmax 동등성을 sklearn binary로 시연하는 다리 챕터.)
- **Phase 1 (Ch 7-14)**: DistilBERT와 Trainer로 Loss/Head 구조를 재정식화합니다. Auxiliary loss로 마무리.
- **Phase 2 (Ch 15-18)**: 한국어로 압축 재방문 (klue/bert-base). **회귀 챕터는 생략** — 영어 Phase 1에서 이미 다뤘기 때문에 Binary부터 시작.
- **Phase 3 (Ch 19-20)**: 토크나이저를 직접 학습. 사전학습 의존 없는 경험. **Phase 3가 클라이맥스가 되도록 토크나이저 시각을 Ch 1부터 일관되게 추적.**

## 작업 흐름

1. **미정 사항 1건은 작성 시작 전 사용자와 확정**:
   - **Ch 17 보조 태스크 (한국어 Auxiliary)**: 라벨 개수 회귀(MSE)가 1순위 — Ch 13과 구조가 닮아 변경점이 "데이터 + 언어"로만 한정됨.
   - (참고) **한국어 회귀 챕터는 의도적으로 생략**됨. Phase 2는 Binary(Ch 15)부터 시작.
2. **챕터 단위 검증 사이클** — 한 챕터씩 다음 순서로 진행:
   1. Codex가 해당 챕터 폴더(`<NN>_<slug>/`)와 노트북(`<NN>_<slug>.ipynb`) 작성. 폴더 안 `README.md`에 챕터 요약을 함께 작성.
   2. **Codex가 곧바로 git commit + push.** Colab 버튼이 GitHub의 master 브랜치를 가리키므로 푸시 전엔 사용자가 노트북을 열 수 없음. 따라서 노트북 작성 직후 자동으로 커밋·푸시. 커밋 메시지는 챕터 단위로 의미 있게(`Add Ch <N>: <slug>` 등). 수정·재빌드 후에도 동일하게 커밋·푸시.
   3. 사용자가 README.md의 Colab 버튼으로 노트북을 열어 **직접 끝까지 실행해 검증**. T4에서 30분 내에 끝까지 정상 동작하는지가 합격 기준.
   4. 검증 통과 시 **루트 README.md 표의 `진행` 열을 `—` → `✅`** 으로 갱신. 실패 시 사용자가 피드백을 주면 노트북 수정 → 다시 커밋·푸시. **`✅`은 사용자 검증 후에만 갱신** — Codex가 임의로 표시하지 않음.
   5. 다음 챕터로 진행. 검증 미통과 챕터를 두고 다음 챕터로 넘어가지 않음.
3. Ch 7은 이미 작성된 영어판 베이스 파일 `_drafts/06_bert_pipeline_base.md`(원본 파일명: `01_pipeline_intro.md`, 베이스 파일 자체는 이름 그대로 둠)를 출발점으로. 내용은 그대로 살리고, 챕터 번호 보정·누적 추적표·🔤 토크나이저 노트·❓ FAQ를 추가해 Ch 7 노트북으로 변환.

## 톤과 표기

- **존댓말** (사용자 선호).
- PR 제목, PR 본문, PR 설명 코멘트는 항상 **한글**로 작성합니다. Git commit 메시지는 기존 흐름에 맞춰 영어를 써도 되지만, GitHub에 노출되는 PR 글은 한글을 기본으로 합니다.
- 코드 주석은 한국어. 변수명/함수명은 영어.
- 라이브러리 이름은 백틱: `transformers`, `datasets`, `tokenizers`.
- 수식은 LaTeX: `$L = -\sum y \log p$`.
- 출판 원고에서는 유니코드 수학 기호를 그대로 쓰지 않고 LaTeX 명령으로 표기합니다. 예: `λ` → `$\lambda$`, `Δ` → `$\Delta$`, `≈` → `$\approx$`, `≤`/`≥` → `$\le$`/`$\ge$`, `×` → `$\times$`, `→` → `$\to$` 또는 문장 부호. 코드 블록·출력 문자열에서는 가급적 ASCII 이름(`lambda`, `delta`, `<=`, `->`)을 씁니다.
- 챕터 파일명: `<NN>_<slug>/<NN>_<slug>.ipynb` 두 자리 zero-pad.

## 출판용 LaTeX 원고 규칙 (`book/`)

현재까지 작성된 노트북을 대중서 원고로 묶을 때는 `book/` 아래 LaTeX 프로젝트를 유지합니다. 원천은 챕터 노트북이며, `book/tools/notebook_to_tex.py --execute`로 `book/chapters/`의 장 원고를 재생성합니다. 새 챕터를 추가한 뒤에는 스크립트의 `CHAPTERS` 목록과 `book/main.tex`의 `\input{...}`를 함께 갱신하고, `latexmk -xelatex book/main.tex`로 PDF 빌드를 확인합니다. 출력이 필요 없는 빠른 점검만 할 때는 `--execute` 없이 변환할 수 있지만, 최종 원고는 출력과 해석이 포함되도록 `--execute`를 사용합니다.

조판과 문체는 다음 원칙을 지킵니다.
- 본문 폰트는 `NanumGothic`, 코드 폰트는 `NanumGothicCoding`, 빌드는 XeLaTeX 기준입니다.
- 장 표기는 `Ch 3`보다 `3장`을 우선합니다. `Chapter N`, `Ch N`, `챕터` 같은 표현은 출판 원고에서는 `N장`, `장`으로 윤문합니다.
- 구어체 표현은 공식 원고 톤으로 바꿉니다. 예: `삽질 코너` → `오류 실험`, `떡밥` → `후속 논점`, `그냥` → 문맥에 맞게 `단순히`/`직접`, `뱉다` → `출력하다`.
- FAQ 질문은 `subsubsection`으로 올리지 않고 `faqBox`로 묶습니다. 질문과 답변의 글자 크기가 튀지 않아야 합니다.
- 다음 장 힌트와 장 끝 예고는 `previewBox`로 묶어 “미리보기” 컨셉으로 보여줍니다. 미리보기 박스에만 제목 앞 기호를 허용하고, 다른 박스 제목 앞에는 특수문자를 붙이지 않습니다.
- 박스 폭은 본문 폭 기준으로 일관되게 유지합니다. 미리보기 박스가 `quote` 안에 들어가 폭이 달라지지 않게 합니다.
- 인라인 코드와 `texttt`/verbatim 계열 항목은 회색 음영 코드 스타일로 보이게 합니다. 코드 셀과 마크다운 코드 펜스는 `lstlisting` 블록으로 구분합니다.
- 마크다운 표는 LaTeX 표로 제대로 파싱되게 전처리합니다. 수식 안의 절댓값 기호처럼 `|`가 표 구분자로 오인되는 경우를 주의하고, 넓은 표는 `adjustbox`로 페이지 폭 안에 맞춥니다. `\toprule()`, `\midrule()`, `\bottomrule()`처럼 불필요한 괄호가 남지 않게 합니다.
- 표시 수식은 번호가 있는 `equation` 환경으로 만들고, 설명 문장에서 `\eqref{...}`로 참조합니다. 수식 글꼴은 본문과 어울리는 sans-serif 계열로 통일합니다.
- 유니코드 수학 기호나 특수 기호는 PDF 폰트에서 누락될 수 있으므로 가능한 한 LaTeX 명령으로 변환합니다. 본문·표·박스 제목에서는 `$\lambda$`, `$\Delta$`, `$\approx$`, `$\pm$`, `$\le$`, `$\ge$`, `$\times$`처럼 쓰고, 코드/출력 블록에서는 ASCII 표기를 우선합니다.
- `토크나이저 노트`, `실습`, `해부`처럼 코드가 이어지는 섹션에는 코드 뒤에 산문형 `위 코드 읽기` 설명을 붙입니다. 박스로 만들지 말고 본문 문단으로 넣으며, 행 번호와 실제 코드 조각을 함께 적어 독자가 시선을 덜 왕복하게 합니다. 모든 줄을 기계적으로 설명하지 말고 핵심 의미 단위만 설명합니다. `import`는 주요 패키지 묶음만 간단히 설명하고, 단순 `print()`/`display()` 출력 확인 줄은 굳이 설명하지 않아도 됩니다.
- 최종 책 원고에서는 코드 출력도 함께 보여줍니다. 코드 블록 다음에는 `위 코드 읽기`, 그 다음에는 `출력`, `출력 해석` 순서가 되게 하며, 출력은 너무 길면 핵심 줄만 남깁니다. 출력 해석은 shape, accuracy, MSE/MAE, 확률, classification report, confusion matrix처럼 독자가 의미를 해석해야 하는 지점을 짧게 풀어줍니다.
- `plt.show()`가 있는 실습 코드는 원칙적으로 대응하는 그림을 `book/assets/figures/`에 생성하거나 노트북 출력에서 추출해 `bookfigurelabel`로 본문에 추가합니다. 그림은 코드 블록과 가장 가까운 위치에 배치하고, 그림 목록에 들어가도록 캡션과 label을 함께 붙입니다.
- 모든 그림은 본문에서 반드시 `그림~\ref{fig:...}` 형태로 참조합니다. 그림만 배치하고 끝내지 말고, 바로 아래나 가까운 문단에 `그림 읽기` 또는 해석 문단을 두어 해당 그림의 label을 링크로 호출하고 핵심 해석을 설명합니다.

## 두 방식 동등성 (Ch 4 / Ch 10 핵심)

binary 분류는 두 방식 모두 가능:
- 방식 A: 1차원 출력 + sigmoid + `BCEWithLogitsLoss` (sklearn 호환)
- 방식 B: 2차원 출력 + softmax + `CrossEntropyLoss` (BERT 표준)
- z = z₁ − z₀로 두면 방식 B가 방식 A의 리파라미터화임을 보임.

- **Ch 4** (sklearn): 동일 binary 데이터에 두 방식을 학습해 `predict_proba`가 일치함을 시연 + 수학적 동등성을 식과 코드로 직접 보임.
- **Ch 10** (BERT): 같은 동등성을 BERT에서 다시 확인 (10a/10b 한 노트북). Ch 4의 sklearn 실험을 BERT로 일반화하는 위치.
- 한국어 binary(Ch 14)는 방식 B만 사용 — 동등성은 Ch 4·10에서 이미 다뤘기 때문.

## `Trainer` `problem_type` 자동 매핑 (코드 작성 시 활용)

`AutoModelForSequenceClassification.from_pretrained(...)` 호출 시:
- `num_labels=1` 단독 → 회귀로 추정해 `MSELoss`. 명시적으로는 `problem_type="regression"`.
- `problem_type="single_label_classification"` → `CrossEntropyLoss` (기본값, multi-class 표준).
- `problem_type="multi_label_classification"` → `BCEWithLogitsLoss`. 라벨은 multi-hot float 텐서여야 함.
- Ch 13, 17의 보조 헤드처럼 자동 매핑을 못 쓸 때만 `Trainer.compute_loss` 오버라이드.
