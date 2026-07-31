# 2026년 9월 출판물 검수 계획

> 기준일: 2026-07-31
>
> 목표일: 2026-09-30
>
> 대상: Notebook, WikiDocs, EPUB
>
> 제외: PDF는 이번 마일스톤에서 검수하지 않고 별도 백로그로 관리합니다.

## 1. 목표와 완료 기준

34개 챕터의 Notebook, WikiDocs, EPUB을 8명이 교차 검수합니다.

- Ch 1~23: 산출물마다 2명이 검수합니다.
- Ch 24~34: GPT 이후 구간으로, 산출물마다 3명이 검수합니다.
- Ch 1~23: `23장 x 3형식 x 2명 = 138건`
- Ch 24~34: `11장 x 3형식 x 3명 = 99건`
- 총 검수 횟수: 237건

2026년 9월 30일 완료 조건은 다음과 같습니다.

- [ ] Ch 1~23의 Notebook, WikiDocs, EPUB이 각각 `2/2 PASS`
- [ ] Ch 24~34의 Notebook, WikiDocs, EPUB이 각각 `3/3 PASS`
- [ ] 열린 Blocker와 Major가 0건
- [ ] 모든 수정사항의 재검수가 완료됨
- [ ] 최종 Git 커밋과 EPUB 빌드 버전이 기록됨
- [ ] 이월한 Minor가 있다면 사유와 후속 이슈가 기록됨

## 2. GitHub 운영 원칙

진행 현황의 단일 출처는 GitHub Project로 둡니다. 별도의 스프레드시트에서 상태를 중복 관리하지 않습니다.

- **GitHub Project**: 전체 일정, 담당자, 상태, 마감일을 관리합니다.
- **마일스톤**: 9월 30일까지 닫아야 할 이슈와 완료율을 집계합니다.
- **챕터 검수 이슈**: 챕터별 세 산출물의 검수 기록과 승인을 보관합니다.
- **결함 이슈**: 검수 중 발견한 개별 문제를 수정하고 재검수하는 단위입니다.
- **Pull Request**: 결함 수정 내용을 리뷰하고 결함 이슈를 닫는 단위입니다.

GitHub Project 이름과 마일스톤은 다음과 같이 통일합니다.

```text
Project: neuqes-101 출판 검수 · 2026-09
Milestone: 2026-09-30 출판 검수 완료
Due date: 2026-09-30
```

GitHub Projects는 이슈와 Pull Request를 표, 보드, 로드맵으로 관리하고 사용자 정의 필드와 자동화를 적용할 수 있습니다.

- [GitHub Projects 소개](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)
- [Project 보기 구성](https://docs.github.com/en/issues/planning-and-tracking-with-projects/customizing-views-in-your-project)
- [Milestone 소개](https://docs.github.com/en/enterprise-cloud@latest/issues/using-labels-and-milestones-to-track-work/about-milestones)

## 3. 팀과 담당 범위

실제 GitHub 사용자명이 확정되면 아래 `A1`~`D2`를 GitHub 핸들로 교체합니다.

| 페어 | 작업자 | 기본 담당 | 특성 |
|---|---|---:|---|
| A | A1, A2 | Ch 1~11 | 장 수는 많지만 상대적으로 실행과 기술 난도가 낮음 |
| B | B1, B2 | Ch 12~19 | Multi-class, Multi-label, Auxiliary, 한국어 BERT |
| C | C1, C2 | Ch 20~26 | 토크나이저, BERT 사전학습, GPT scratch |
| D | D1, D2 | Ch 27~34 | Continual pretraining, SFT, DPO, GRPO, Diffusion |

기본 담당 페어의 두 사람은 담당 챕터의 Notebook, WikiDocs, EPUB을 모두 독립적으로 검수합니다. 함께 읽고 한 번만 기록하지 않고, 각자 PASS 또는 수정 요청을 남깁니다.

### Ch 24~34 세 번째 검수자

세 번째 검수자는 기본 담당 페어와 다른 사람으로 지정하고 세 산출물을 모두 확인합니다.

| 챕터 | 기본 담당 | 세 번째 검수자 |
|---|---|---|
| Ch 24 | C | D1 |
| Ch 25 | C | D2 |
| Ch 26 | C | A1 |
| Ch 27 | D | A2 |
| Ch 28 | D | B2 |
| Ch 29 | D | C1 |
| Ch 30 | D | B2 |
| Ch 31 | D | C1 |
| Ch 32 | D | B1 |
| Ch 33 | D | C2 |
| Ch 34 | D | C2 |

이 배정은 실제 노트북의 셀 수, 코드량, 실행 기록을 반영했으며 예상 작업량 차이를 약 10% 안쪽으로 맞춘 안입니다. 첫 2주의 실제 소요시간이 15% 이상 벌어지면 아직 시작하지 않은 챕터 또는 세 번째 검수자를 조정합니다.

## 4. 일정

### 4.1 준비와 본 검수

| 기간 | A | B | C | D |
|---|---|---|---|---|
| 7/31~8/2 | GitHub Project, 마일스톤, 필드, 이슈 준비 | GitHub Project, 마일스톤, 필드, 이슈 준비 | GitHub Project, 마일스톤, 필드, 이슈 준비 | GitHub Project, 마일스톤, 필드, 이슈 준비 |
| 8/3~8/9 | Ch 1~2 | Ch 12 | Ch 20 | Ch 27~28 |
| 8/10~8/16 | Ch 3~4 | Ch 13 | Ch 21 | Ch 29~30 |
| 8/17~8/23 | Ch 5~6 | Ch 14 | Ch 22 | Ch 31 |
| 8/24~8/30 | Ch 7~8 | Ch 15~16 | Ch 23~24 | Ch 32 |
| 8/31~9/6 | Ch 9~10 | Ch 17 | Ch 25 | Ch 33 |
| 9/7~9/13 | Ch 11 | Ch 18~19 | Ch 26 | Ch 34 |

Ch 24~34의 세 번째 검수는 기본 담당 페어의 검수가 끝난 뒤 시작하며, 원칙적으로 다음 주 수요일까지 완료합니다.

### 4.2 동결과 완료

| 마감 | 상태 |
|---|---|
| 9/13 | Ch 1~34의 기본 1·2차 검수 완료, 모든 Blocker와 Major 등록 |
| 9/20 | Ch 24~34의 3차 검수 완료, Blocker 0건, Major 0건, 내용 동결 |
| 9/27 | 전체 재검수와 EPUB 최종 확인 완료, 출판물 동결 |
| 9/28 | 최종 EPUB 빌드와 Git 리비전 기록 |
| 9/29 | 링크, 목차, 이미지, 누락 여부 최종 점검 |
| 9/30 | Project와 마일스톤 완료 확정 |

9월 20일 이후에는 새로운 내용 추가나 큰 구조 변경을 하지 않습니다. 오탈자, 링크, EPUB 렌더링처럼 결과를 제한적으로 바꾸는 수정만 허용합니다.

## 5. GitHub Project 구성

### 5.1 필드

| 필드 | 형식 | 값 또는 용도 |
|---|---|---|
| Status | Single select | Backlog / This week / Reviewing / Fixing / Re-review / Ready / Done |
| Pair | Single select | A / B / C / D |
| Phase | Single select | Phase 0 / 1 / 2 / 3 / 4 / 5 |
| Required Reviews | Single select | 2 / 3 |
| Third Reviewer | Text | Ch 24~34의 세 번째 검수자 |
| Notebook | Single select | 0/N / Reviewing / Fixing / N/N PASS |
| WikiDocs | Single select | 0/N / Reviewing / Fixing / N/N PASS |
| EPUB | Single select | 0/N / Reviewing / Fixing / N/N PASS |
| Severity | Single select | None / Minor / Major / Blocker |
| Estimate | Number | 사전 난이도 점수 |
| Actual Hours | Number | 실제 검수시간 합계 |
| Source Revision | Text | 검수 기준 Git 커밋 SHA |
| Target Date | Date | 챕터 완료 예정일 |

`N`은 Ch 1~23에서는 2, Ch 24~34에서는 3입니다. 세부 승인자는 Project 필드가 아니라 챕터 이슈의 체크리스트와 댓글에 기록합니다.

### 5.2 보기

| 보기 | 형식 | 용도 |
|---|---|---|
| 전체 현황 | Table | 34개 챕터와 세 산출물의 PASS 현황 확인 |
| 이번 주 | Board | `Status` 기준으로 현재 작업만 확인 |
| 담당자별 | Table | Assignee 또는 Pair로 그룹화해 작업량 확인 |
| 3차 검수 | Table | `Required Reviews: 3`만 표시 |
| Blocker·Major | Table | `Severity: Blocker, Major`만 표시 |
| 재검수 대기 | Board | 수정은 끝났지만 재검수가 필요한 항목 확인 |
| 9월 로드맵 | Roadmap | `Target Date` 기준 일정 확인 |

### 5.3 기본 자동화

- 검수 라벨이 붙은 이슈를 Project에 자동 추가합니다.
- 이슈가 닫히면 `Status`를 `Done`으로 변경합니다.
- `Blocker` 또는 `Major`가 열려 있으면 챕터를 `Ready`로 옮기지 않습니다.
- 닫힌 이슈를 자동 보관하더라도 9월 30일 전에는 검수 이력을 확인할 수 있게 보관 시점을 늦춥니다.

처음부터 복잡한 GitHub Actions를 만들지 않습니다. 1주간 수동으로 운영한 뒤 반복 입력이 확인된 부분만 자동화합니다.

## 6. 이슈 구성

### 6.1 챕터 검수 이슈

챕터당 상위 이슈를 하나씩 만들어 총 34개를 Project에 넣습니다.

```text
[검수] Ch 01 TF-IDF — Notebook/WikiDocs/EPUB
[검수] Ch 02 sklearn Regression — Notebook/WikiDocs/EPUB
...
[검수] Ch 34 한국어 Diffusion — Notebook/WikiDocs/EPUB
```

챕터 이슈 본문은 다음 형식을 사용합니다.

```markdown
## 기준 버전

- Source commit:
- WikiDocs 확인 URL:
- EPUB build:
- Required reviews: 2 또는 3

## Notebook

- [ ] 1차 검수자 PASS
- [ ] 2차 검수자 PASS
- [ ] 3차 검수자 PASS — Ch 24~34만 사용

## WikiDocs

- [ ] 1차 검수자 PASS
- [ ] 2차 검수자 PASS
- [ ] 3차 검수자 PASS — Ch 24~34만 사용

## EPUB

- [ ] 1차 검수자 PASS
- [ ] 2차 검수자 PASS
- [ ] 3차 검수자 PASS — Ch 24~34만 사용

## 완료 조건

- [ ] Blocker 0건
- [ ] Major 0건
- [ ] 수정사항 재검수 완료
- [ ] 최종 버전 기록
```

Ch 1~23에서는 세 번째 검수 항목을 삭제하거나 `해당 없음`으로 표시합니다.

### 6.2 결함 이슈

문제를 발견하면 챕터 이슈의 댓글에 묻어두지 않고 결함마다 별도 이슈를 만듭니다.

```text
[Ch 14][Notebook][Major] Auxiliary loss 계산과 본문 설명 불일치
[Ch 25][WikiDocs][Minor] 코드 블록의 들여쓰기 손상
[Ch 32][EPUB][Blocker] 수식과 코드가 화면 밖으로 잘림
```

결함 이슈에는 다음을 기록합니다.

- 대상 챕터와 산출물
- 검수한 버전 또는 Git 커밋
- 실제 결과와 기대 결과
- 재현 절차
- 스크린샷 또는 오류 로그
- 심각도
- 수정 담당자와 재검수자
- 연결된 챕터 검수 이슈

수정한 사람이 자신의 수정만 보고 이슈를 닫지 않습니다. 문제를 처음 등록한 사람이나 지정된 재검수자가 확인한 뒤 닫습니다.

## 7. 라벨

| 분류 | 라벨 |
|---|---|
| 검수 범위 | `qa`, `publication-review` |
| 산출물 | `format:notebook`, `format:wikidocs`, `format:epub` |
| 심각도 | `severity:blocker`, `severity:major`, `severity:minor` |
| 문제 유형 | `type:runtime`, `type:content`, `type:layout`, `type:link`, `type:typo` |
| 후속 범위 | `scope:pdf`, `status:later` |

Pair, Phase, 일정, 상태는 Project 필드로 관리합니다. 같은 정보를 라벨과 Project 필드 양쪽에 중복 저장하지 않습니다.

## 8. 검수 기록과 Pull Request

검수자는 챕터 이슈에 다음 형식으로 결과를 남깁니다.

```text
Notebook 1차 PASS
검수자: @username
기준 커밋: abc1234
환경: Google Colab T4
실행 시간: 24분
발견 이슈: #123, #124
```

수정 Pull Request에는 결함 이슈를 연결합니다.

```markdown
Fixes #123
Related to #42
```

`#123`은 결함 이슈, `#42`는 챕터 검수 이슈입니다. 챕터 검수 이슈를 `Fixes`로 연결하면 수정 하나가 병합될 때 챕터 전체가 너무 일찍 닫힐 수 있으므로 사용하지 않습니다.

검수 후 원천 노트북 내용이 바뀌면 해당 챕터의 WikiDocs와 EPUB PASS도 무효화하고 다시 검수합니다. 챕터 이슈의 `Source commit`과 `EPUB build`가 최종 검수 버전과 반드시 같아야 합니다.

## 9. 심각도와 처리 기한

| 심각도 | 기준 | 처리 원칙 |
|---|---|---|
| Blocker | 실행 실패, 장 누락, 핵심 내용 오류, EPUB 열기 실패 | 즉시 작업 중단 후 우선 수정 |
| Major | 잘못된 수식·코드·출력, 중요한 설명 누락, 표·그림 손상 | 해당 주 또는 다음 주 초까지 수정 |
| Minor | 오탈자, 작은 줄바꿈, 표현 통일 | 내용 동결 전 일괄 수정 가능 |

Blocker와 Major가 하나라도 열려 있으면 해당 챕터는 `Ready` 또는 `Done`으로 이동할 수 없습니다.

## 10. 주간 운영

### 월요일

- 이번 주 챕터를 `This week`로 이동합니다.
- 담당자와 검수 기준 리비전을 확인합니다.
- 이전 주 미해결 Blocker와 Major를 먼저 확인합니다.

### 화요일~목요일

- 두 기본 검수자가 독립적으로 검수합니다.
- 발견한 문제를 결함 이슈로 등록합니다.
- Ch 24~34는 기본 검수가 끝나면 세 번째 검수자에게 넘깁니다.

### 금요일

- 30분 이내 주간 회의를 진행합니다.
- Blocker, Major, 일정 지연만 논의합니다.
- `Actual Hours`를 갱신하고 다음 주 작업량을 조정합니다.
- Project 상태를 업데이트합니다.

개별 오탈자와 문장 표현은 주간 회의에서 하나씩 논의하지 않고 담당자가 이슈에서 처리합니다.

## 11. 시작 체크리스트

- [ ] 8명의 GitHub 사용자명과 A1~D2 매핑 확정
- [ ] 8명에게 저장소와 Project 접근 권한 부여
- [ ] GitHub Project 생성 및 저장소 연결
- [ ] 마일스톤 `2026-09-30 출판 검수 완료` 생성
- [ ] Project 필드와 보기 생성
- [ ] 라벨 생성
- [ ] 챕터 검수 이슈 34개 생성
- [ ] 각 이슈에 기본 페어와 세 번째 검수자 배정
- [ ] 8/3 시작 전에 대표 챕터 하나로 검수 기준 보정
- [ ] PDF 이슈는 `scope:pdf`, `status:later`로 9월 마일스톤에서 제외

GitHub CLI로 Project를 생성하거나 수정하려면 최초 한 번 `project` 권한을 추가해야 합니다.

```bash
gh auth refresh -s project
```

권한 추가 후 Project와 챕터 이슈를 자동 생성할 수 있습니다. 외부 Project, 마일스톤, 이슈를 실제로 생성하기 전에는 제목, 담당자 GitHub 핸들, 공개 범위를 최종 확인합니다.
