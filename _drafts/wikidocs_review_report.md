<!-- 자동 생성 검토 리포트 — neuqes-101 WikiDocs 페이지 (Ch 1-34, 134페이지) -->
<!-- 검토일: 2026-06-21 | issueCount=69 | bySeverity={'높음': 20, '중간': 29, '낮음': 20} -->

# neuqes-101 WikiDocs 페이지 검토 종합 리포트

**총 이슈 60건** (높음 19 · 중간 28 · 낮음 13, 중복 통합 후). Ch 1-34 전 구간의 본문·실습·wrapup 페이지와 `executed/*.ipynb` 실행본을 전수 대조한 결과, **챕터별 핵심 수치(loss·accuracy·F1·AUC·파라미터)는 실행본과 거의 모두 일치**하며 톤(존댓말)·용어 통일·범위 하이픈·출력 영문화 규약도 대체로 잘 지켜졌습니다. 다만 책이 20챕터에서 34챕터로 성장하는 동안 **갱신이 누락된 교차 참조 번호와 stale prose**가 패턴으로 남아, 사실정확성 이슈가 집중되어 있습니다. 전반 품질은 **B+ (구조·서술은 견고, 메타데이터 정합성은 일괄 보정 필요)**.

## 종합 평가

| 차원 | 상태 | 요약 |
|---|---|---|
| **통일성** | 양호 (꼬리표류 미갱신) | 용어(`BCEWithLogitsLoss` (sklearn: log loss))·존댓말·출력 영문화는 일관. 단 '전체 20챕터'(Ch 1-18)·'Phase 3 Ch 19-20'(Ch 16-18)·DistilBERT 66M/67M 혼용이 옛 표기로 잔존. print() 한국어 출력 59곳(21파일)이 가장 광범위한 규약 위반. |
| **사실정확성** | 주의 필요 | 챕터 핵심 수치는 정확하나 (1) 교차 참조 챕터 번호 off-by-one 다수, (2) overview/wrapup 기대 수치가 실측과 벌어짐(Ch 21·23), (3) 학습 step·시간 prose가 stale(Ch 24-28), (4) Ch 32 토크나이저/모델 사양 전면 모순. |
| **흐름** | 양호 (예고 일부 오류) | Phase 경계·-100/마스킹 thread·축 변화 서술은 탄탄. 단 Ch 3 다음 챕터 예고가 통째로 Ch 5 내용, Ch 33·34 wrapup에 '다음 주제 미정' 메타 노출. |

## 🔴 높음 (즉시 수정)

| 챕터 | 페이지 | 문제 | 수정안 |
|---|---|---|---|
| Ch 1 | 01-tfidf-wrapup.md:90 | Phase·챕터 번호 전부 오기. "Phase 2(Ch 13-16) ... Phase 3(Ch 18)" — 실제 Phase 2=Ch 15-18, Phase 3=Ch 19-23 | "Phase 2(Ch 15-18, 한국어) ... Phase 3(Ch 19부터)"로 수정 |
| Ch 2 | 02-sklearn_regression-wrapup.md:83 | "Ch 2 → Ch 4 (5클래스 분류)" — 5클래스는 Ch 5, Ch 4는 softmax binary | "Ch 2 (회귀) → Ch 5 (5클래스 분류)" |
| Ch 3 | 03-sklearn_binary.md:75 + 다음 챕터 예고 | 다음 챕터 예고 전체가 Ch 4가 아니라 **Ch 5(5클래스) 내용**. 실제 Ch 4는 softmax binary(2차원, 동등성 시연) | 예고를 "Ch 4. sklearn Softmax Binary — 같은 이진 데이터를 2차원 softmax+CE로"로 교체. line 75 "5차원" → "2차원", "K=2 유지" |
| Ch 6 | 06-sklearn_multilabel.md:77,126 / -wrapup.md:5,6 | BERT multi-label 후속을 4곳에서 'Ch 12'로 지칭 — DistilBERT multi-label은 **Ch 13**(Ch 12=multi-class) | 네 곳 'Ch 12' → 'Ch 13' (주석 `# PyTorch (Ch 13 이후, multi-label)` 포함) |
| Ch 8 | 08-tokenizer_datasets.md:153 | 챕터-태스크 매핑 off-by-one: "Ch 11 multi-class, Ch 12 multi-label" — 실제 Ch 11=binary softmax, Ch 12=multi-class, Ch 13=multi-label | "Ch 10 binary(sigmoid), Ch 11 binary(softmax), Ch 12 multi-class, Ch 13 multi-label"로 수정 |
| Ch 8 | 08-...md:184 / -practice.md:427,684,691,734 | auxiliary loss를 5곳에서 'Ch 13'으로 지칭 — 보조 loss는 **Ch 14**(Ch 13=multi-label) | 다섯 곳 'Ch 13' → 'Ch 14' |
| Ch 11 | 11-bert_binary_softmax.md:94 | 파라미터 합계 두 값이 실행본·산식 모두 불일치. 방식 A=66,955,010(틀림)/방식 B=66,955,778(어디에도 없는 값) | A 합계 → **66,954,241**, B 합계 → **66,955,010** (executed/11 출력과 일치) |
| Ch 12 | 12-bert_multiclass.md:94 | Ch 11의 잘못된 합계를 승계 + K=5 합계도 산식과 어긋남 | K=2 → **66,955,010**, K=5 → **66,957,317** (executed/12·13과 일치) |
| Ch 18 | 18-ko_auxiliary.md:86 | 사용자 향 본문에 내부 규약 파일명 `CLAUDE.md` 노출 (메타 누출 금지 위반) — 직접 확인됨 | 괄호의 "CLAUDE.md 의 ... 규칙" 삭제, 내용만 풀어 서술 |
| Ch 21 | 21-en_bert_classify.md:78,276 / wrapup:146 | overview 기대치(acc 75-85% / AUC 0.85-0.92, Ch 10=92-95%)가 실측(0.6490/0.7079, Ch 10=0.9030)과 정면 모순. practice는 정직하게 보고 | overview·wrapup을 'acc 약 0.65, AUC 약 0.71'로, Ch 10 인용은 '약 0.90'으로 정정 |
| Ch 22 | 22-ko_bert_pretrain.md:26,41 | Ch 20 데이터를 `yelp_polarity` text로 오기 — 실제 Wikitext-103(README·Ch 20·21·23 모두 일치) | line 26·41을 'Wikitext-103 paragraphs (일반 도메인, Salesforce/wikitext)'로 수정 |
| Ch 23 | 23-ko_bert_classify.md:77,234 / wrapup:138 | overview 기대치(acc 65-75%/AUC 0.75-0.85)가 실측(0.5420/0.5585, 동전 던지기 수준)과 모순. practice:617은 정직 | 'acc 약 0.54, AUC 약 0.56'로 낮추고 짧은 사전학습(0.2분) 한계 명시 |
| Ch 23 | 23-ko_bert_classify.md:17,181,183 | MLM epoch이 overview 안에서 '1 epoch'(line 17·181·183)과 '3 epoch'(187·실행본) 자가당착. 실행본=`MLM_EPOCHS=3` | line 17·181·183을 '3 epoch'으로 통일 |
| Ch 25 | 25-gpt2_continual_pretrain.md:5,46,61,214,265 | step '약 500-800/460', 'T4 8-10분' — 실행본 global_step 3242, 19.22분 | '약 3,200 step (51,863 chunks / eff. batch 16)', 'T4 약 19분'으로 갱신 |
| Ch 26 | 26-ko_tiny_gpt.md:58 | 변경점 표가 Ch 24 lr을 '5e-4'로, Ch 26을 '(그대로)'로 표기 — 실측 Ch 24=3e-4, Ch 26=5e-4(서로 다름). '변경점 한 가지' 원칙도 위배 | 'Ch 24: 3e-4 | Ch 26: 5e-4'로 고치고 lr 미세조정을 명시(또는 노트북을 일치시킴) |
| Ch 27 | 27-ko_gpt2_continual_pretrain.md:5,58,73,291 | '약 1 epoch (수백 step)', 'T4 8-12분' — 실행본 global_step 3033, 17.08분 | '약 3,000 step (48,513 chunks / eff. batch 16)', 'T4 약 17분' |
| Ch 32 | 32-diffusion_intro.md:37,89,91,95,100,124,128,146 | **한 페이지 안에서 두 토크나이저 공존** — BPE 2048 직접 학습(line 37/124)과 WordPiece bert-base-uncased vocab 30,522(line 89/91/146)가 모순. 실행본·README는 BPE 2048(직접 학습) — 직접 확인됨 | 30,522/WordPiece/bert-base-uncased 서술 전부 BPE 2048로 통일. baseline ln(30522)=10.33 → ln(2048)=7.62, line 146 섹션 재작성, line 37 '(bert-base-uncased 가져옴)' 삭제 |
| Ch 32 | 32-diffusion_intro.md:174,206,208 | 모델/학습 수치 불일치: '13M params, max_steps=1500, batch 32, T4 13-15분, loss 4-6' — 실행본 3.79M/30000 step/batch 64/18.92분/loss 3.71. Ch 33 회고도 3.79M·30000 step | 13M→3.79M, 1500→30000, 32→64, '약 19분', '약 3.7'로 정정 |
| Ch 32 | 32-diffusion_intro-wrapup.md:9,80 | FAQ Q6 답변 전체가 사실 반대 — "WordPiece 내장 [MASK](id 103) 바로 사용"이라지만 실제는 BPE 직접 학습 + special_token [MASK](id 2) 추가 | Q6를 'BPE를 쓰면서 [MASK]는 어떻게 마련했나'로 재구성, 답변을 'BPE 2048 직접 학습 + special_tokens로 [MASK] 추가'로 교체. 표 line 9도 수정 |

## 🟡 중간

| 챕터 | 페이지 | 문제 | 수정안 |
|---|---|---|---|
| Ch 1-18 | 18개 overview | 도입부 추적표 안내가 '전체 20챕터 표'(옛 표기). Ch 19+는 이미 '전체 챕터 표'로 통일됨 | 18개 파일 '전체 20챕터 표는' → '전체 챕터 표는' 일괄 치환 |
| Ch 3 | 03-sklearn_binary-wrapup.md:33 | softmax 방식 B를 'Ch 9에서 다룰'이라 함 — Ch 9는 BERT 회귀, 방식 B=Ch 11 | "Ch 11에서 다룰 '방식 B'"로 수정 |
| Ch 3 | 03-sklearn_binary-wrapup.md:77 | "긍정 ≈ 60%" — 같은 챕터 실측 49.4%(0=2044/1=1996)와 챕터 내부 모순 — 직접 확인됨 | "긍정 ≈ 49%, 거의 반반"으로 수정 |
| Ch 7 | 07-bert_pipeline.md:135 / -wrapup.md:135 | klue/bert-base 등장을 'Ch 14'로 지칭 — 실제 Ch 15(Phase 2 시작), Ch 14는 영어 보조 loss | 두 곳 'Ch 14' → 'Ch 15' |
| Ch 16, 17 | 16-ko_multiclass.md:74 / 17-ko_multilabel.md:112 | 토큰 길이 연쇄 과대 추정: Ch 16 '25-30'(실측 mean 15.8/max 27), Ch 17 '50-60'(실측 mean 30/max 41) | Ch 16 '평균 약 16(최대 27)', Ch 17 '약 2배인 평균 약 30(최대 41)' |
| Ch 16-18 | 16:70, 17:108, 18-wrapup:123 | 'Phase 3 (Ch 19-20)' — README·Ch 19·23은 Ch 19-23 | 세 곳 'Ch 19-20' → 'Ch 19-23' |
| Ch 19 | 19-tokenizer_training.md:179 | 범위에 물결표 '수십~백여 언어' (하이픈 규약 위반) | '수십-백여 언어' 또는 '수십에서 백여 개 언어' |
| Ch 20 | 20-en_bert_pretrain.md:1,137,247 | Yelp를 '영화 리뷰'로 반복 오기 (Yelp=식당/업체, 영화 리뷰=NSMC) | 'Yelp 리뷰(식당·업체)'로 정정 |
| Ch 20 | 20-en_bert_pretrain.md:214,238,239 | loss/perplexity 목표 'loss 5 이하 / ppl 50-500' — 실측 loss 7.13/ppl 1247 | 'loss 약 7', 'ppl 약 1,200개 후보'로 정정 |
| Ch 21 | 21-en_bert_classify.md:1,133,233 / wrapup:51,159 | Ch 20과 동일 Yelp='영화 리뷰' 오기 | 'Yelp 리뷰(식당·업체)'로 정정. wrapup:159는 'Yelp(영어)/NSMC(한국어)' 분리 |
| Ch 21 | 21-en_bert_classify.md:233 | MLM loss '4-6 / 수백 후보' — 실측 7.20 / ppl 1339.60 | 'loss 약 7', '약 1,300개 후보'로 정정 |
| Ch 23 | 23-ko_bert_classify-wrapup.md:131 | FAQ Q6에 고아 수치 '정확도 89%'(어느 셋업과도 불일치; ours 0.542/ref 0.86) | 'klue/bert-base 기준 약 86% 천장' 등 맥락에 맞게 교체 |
| Ch 24 | 24-gpt_tinystories.md:5,284 | 학습 시간 '약 18분 / 15-18분' — 실측 0.87분 (약 20배 과대). practice는 정직 | 'T4 약 1분 (1500 step)', 상단 총합 하향 |
| Ch 26 | 26-ko_tiny_gpt.md:5,219 | Ch 24와 동일 시간 과대(18분 vs 실측 0.90분) | 'T4 약 1분'으로 수정 |
| Ch 26, 27 | 26:76 / 27:87 | alignment를 'Ch 29-30'으로 표기 — 정답 DPO=Ch 30/GRPO=Ch 31, Ch 29는 벤치마크 평가 | 'Ch 30-31'로 수정 |
| Ch 28 | 28-sft.md:5 | SFT 학습 '약 12-18분' — 실측 2.38분(188 step, 약 6배 과대) | '약 2-3분 (3,000 샘플 1 epoch, 188 step)' |
| Ch 30 | 30-dpo-variation.md:8 / -wrapup.md:67 | DPO β 기본값을 '1'로 오기 — 본문·실행본·추적표는 일관되게 0.1 | 두 곳 '1' → '0.1' |
| Ch 31 | 31-grpo.md:242 / -wrapup.md:117 | 'ref-free (beta=0)'라 설명 — 실행본 GRPOConfig beta=0.04(KL 앵커) | '작은 KL 앵커(beta=0.04)'로 수정, wrapup Q6 코드도 조정 |
| Ch 31 | 31-grpo.md:309 | 부록 링크 'appendix_qwen_grpo_hpo.ipynb' — 실제 파일 31_grpo_appendix.ipynb (깨진 링크) | 실제 경로로 정정 |
| Ch 33 | 33-diffusion_train-wrapup.md:114 | Ch 34 예고가 '다룰 모델은 확정되는 대로' 미정 메타 노출 + 실제 내용(한국어 diffusion+80/10/10)과 불일치 | 실제 Ch 34 내용으로 교체, '확정되는 대로' 삭제 |
| Ch 34 | 34-ko_diffusion-wrapup.md:126 | Phase 5 예고에 '정해지는 대로 안내하겠습니다' 미정 메타 노출 | 문구 삭제, AR·diffusion 회고 방향만 단정 서술 |
| Ch 9-15, 20-23 | 다수 (특히 15:34/120) | DistilBERT body를 '67M'(Ch 9-14)과 '약 66M'(Ch 20·21·23)으로 혼용. Ch 15는 한 파일 내 자기모순 | 전 챕터 body='약 66M'으로 통일, 헤드 포함 총합만 '약 67M(헤드 포함)' |
| Ch 2-34 | 21파일 59곳 | print()/plt 문자열 안 한국어 (출력 영문화 규약 위반). 빈도: 29-practice(12), 27-wrapup(7), 30-practice(6) | print/plot 한국어를 영문으로, 설명은 마크다운·주석으로 이동 |

## 🟢 낮음

- **Ch 8** (08-...md:140,146,150) — 'Ch 9-13 모든 분류 학습'이 회귀(Ch 9)를 분류에 포함. 'Ch 9-14 모든 Trainer 학습' 또는 '분류는 Ch 10-13'으로.
- **Ch 7-14 overview** — '전체 20챕터 표' 문구(상단 Ch 1-18 일괄 수정에 흡수).
- **Ch 15** (15-ko_binary.md:69) — distilbert 토큰화 예시가 '글자 단위 11+'이나 실행본은 자모 단위 14토큰. '자모(초·중·종성) 단위'로 정정.
- **Ch 19** (19-...md:32,98) — Ch 20 데이터를 'Yelp text'로 표기, 실제 Wikitext-103. 'Wikitext-103(일반 도메인 위키)'으로 통일.
- **Ch 20** (20-...md:147) — 블록 수 '약 1,000-2,000' → 실측 5,352(약 68만 토큰).
- **Ch 20** (20-...md:5) — MLM 시간 '15-20분' → 실측 0.4분. 전체 소요는 데이터 다운로드 지배(약 5-8분)로 재산정.
- **Ch 22** (22-...md:5) — 한국어 MLM 시간 '15-20분' → 실측 0.3분.
- **Ch 23** (23-...-wrapup.md:34-37) — FAQ Q1 스니펫이 NSMC 길이를 재면서 Yelp 변수(`ds_train_full`) 복붙. NSMC 데이터 객체(`df_train['document']`)로 정정.
- **Ch 24** (24-...md:287,133,149) — 도달 loss '약 2.5-3.0' → 실측 train_loss 3.83. '약 3.5-4.0(1분 기준)' 또는 누적평균임을 명시.
- **Ch 24** (24-...md:38,40,46) — Phase 4 범위를 'Ch 24-30'과 'Ch 24-31'로 혼용. 'Ch 24-31'로 통일.
- **Ch 28** (28-sft.md:29) — KoAlpaca 샘플 '약 3-5K' → 실측 정확히 3,000. '약 3K (3,000 샘플)'.
- **Ch 31** (31-grpo.md:203,244) — 'SFT 워밍스타트'·'난이도 필터' 두 절이 모두 '## 5'로 중복. 본문 참조(§2.5/§4.5)에 맞춰 정정.
- **Ch 33** (33-...-anatomy.md:46) — 4-gram 반복률(억제 없음)이 한 곳만 0.173, 나머지·실행본은 0.177. 0.177로 통일.
- **Ch 34** (34-...-anatomy.md:53) — 순진한 diffusion 고정-t acc가 0.081과 0.084로 혼재. 본 노트북 기준 0.081로 통일(부록 0.078은 별도).
- **Ch 34** (34-...-wrapup.md:120) — 학습 시간 '20.11분' → 실행본 20.20분.
- **Ch 34** (34-...-practice.md:243) — 조건부 생성 print 헤더가 영어 'Once upon a time'인데 실제 프롬프트는 '옛날 옛날에'(Ch 33 복사 잔재). 헤더 정정/중복 제거.
- **Ch 21** (21-...-practice.md:818-825) — Ch 10 ref 주석 데이터셋명 오기('yelp_polarity' → 실제 Yelp/yelp_review_full 별점 이진화) + acc 0.93(실측 0.9030). 데이터셋명·수치 정정.

## 패턴별 일괄 수정 제안

여러 페이지에 걸친 **같은 원인의 이슈는 묶어서 한 번에 닫는 것**이 효율적입니다. 단일 출처 README가 이미 정답을 들고 있어 대부분 일괄 치환으로 해결됩니다.

1. **'전체 20챕터' → '전체 챕터' (Ch 1-18 overview 18개 파일)** — 책이 34챕터로 자랐으나 안내 문구 미갱신. Ch 19+ 형태와 통일.
2. **'Phase 3 (Ch 19-20)' → 'Phase 3 (Ch 19-23)' (Ch 16:70, 17:108, 18-wrapup:123)** — Phase 3가 2챕터였던 시절의 옛 범위.
3. **alignment 챕터 번호 'Ch 29-30' → 'Ch 30-31' (Ch 26:76, 27:87)** — DPO=Ch 30/GRPO=Ch 31, Ch 29=벤치마크 평가.
4. **챕터 cross-reference off-by-one (Ch 6·7·8)** — multi-label은 Ch 13(not 12), auxiliary는 Ch 14(not 13), klue/bert-base는 Ch 15(not 14). 20챕터 시절 매핑이 화석으로 잔존. README 추적표 행과 대조해 일괄 보정.
5. **파라미터 합계 승계 오류 (Ch 11→Ch 12)** — Ch 11의 틀린 합계가 Ch 12로 전파. A=66,954,241 / K=2=66,955,010 / K=5=66,957,317 (executed/9-13 출력 기준)으로 동시 수정.
6. **overview/wrapup 기대 수치 ↔ practice 실측 불일치 (Ch 20·21·23)** — '성공을 앞세운다'로 작성된 옛 낙관 예측만 stale. practice 페이지는 이미 정직하므로 그 톤·수치에 overview·wrapup을 맞춤. MLM loss/ppl도 동일(실측 loss~7, ppl~1,200-1,300).
7. **학습 step·시간 prose stale (Ch 24-28)** — '실행 노트북 기준 정리' 커밋이 prose까지 닿지 못함. scratch(24/26)는 시간 약 20배 과대, continual(25/27)은 step 4-7배 과소·시간 2배 과소, SFT(28)는 시간 약 6배 과대. 각 practice의 실측(elapsed/global_step)으로 일괄 갱신.
8. **Yelp='영화 리뷰' 오기 (Ch 20·21)** — 영화 리뷰는 NSMC(Ch 22-23), Yelp는 식당·업체 리뷰. 도메인 구분 복원.
9. **DistilBERT 66M/67M 혼용 (Ch 9-15, 20-23)** — body는 '약 66M' 단일 값, 헤드 포함만 '약 67M(헤드 포함)'.
10. **범위 물결표 → 하이픈 / '약'** — Ch 19:179 '수십~백여', Ch 7:80 '90~100%', Ch 7-practice:282 '30초~1분' 등. approx의 '~67M'은 '약 67M'으로 바꿔 ~ 사용 자체를 축소. (코드의 비트연산 ~, `~/.cache`, 't ~ U()' 분포 표기는 위반 아님.)
11. **print() 한국어 출력 (21파일 59곳)** — 가장 광범위한 규약 위반. 출력 문자열은 영문, 한국어 설명은 마크다운/주석으로.
12. **내부 메타 누출 (Ch 18:86)** — 사용자 향 페이지의 `CLAUDE.md` 직접 인용 1곳. 파일명/규칙명만 제거하고 내용은 유지.
13. **미정 메타 노출 (Ch 33·34 wrapup)** — '확정되는 대로'·'정해지는 대로 안내' 운영 상태 노출. 확정된 방향만 서술.

## 검토 범위·방법

- **범위**: Ch 1-34 전 34챕터의 모든 페이지 유형(overview/practice/anatomy/variation/wrapup)을 6개 담당 그룹(1-6, 7-14, 15-19, 20-23, 24-29, 30-34)으로 나눠 검토하고, 횡단 통일성을 별도로 점검.
- **방법**: 각 페이지의 수치·챕터 참조·용어·출력 규약을 `executed/*.ipynb` 실행본 출력 및 단일 출처 `README.md` 추적표와 **전수 대조**. loss·accuracy·F1·AUC·positive rate·파라미터 수·step·소요 시간·perplexity를 셀 출력과 직접 비교(예: Ch 2 MSE 1.5565, Ch 3 acc 0.8639/positive 49.4%, Ch 10 acc 0.9030, Ch 11 동등성 corr 0.9904, Ch 13 micro_f1 0.8398, Ch 21 acc 0.6490, DPO 0.500→0.844, GRPO 0.875→0.891 등 일치 확인).
- **편집장 추가 검증**: Ch 3 '긍정 60%', Ch 6 'Ch 12' 4곳, Ch 18 `CLAUDE.md` 누출, Ch 32 BPE 2048 ↔ WordPiece 30,522 자기모순을 실제 파일에서 직접 재확인 — 모두 이슈 JSON과 일치.
- **결과**: 핵심 수치 자체의 오류는 거의 없고(Phase 0 sklearn 구간은 전수 일치), 발견된 60건은 대부분 (a) 책 성장 과정의 메타데이터 미갱신, (b) practice는 정직하나 overview/wrapup만 옛 예측 잔존, (c) Ch 32의 토크나이저 교체 흔적 미정리에 집중. 단일 출처(README)·후속 챕터·실행본이 정답을 보유해 **대부분 일괄 치환으로 종결 가능**.