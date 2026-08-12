# 결함 이슈 등록과 수정 기록

절차의 원본은 `REVIEW_PLAN_2026-09.md` §6~§8, 양식 예시는 `REVIEW_GUIDE_2026-09.pdf` 5쪽이다. 여기는 실제로 등록해 본 결과의 실무 메모다.

## 먼저 확인

```bash
gh issue list --search "Ch 12" --limit 60        # 챕터 검수 이슈 번호 (예: #36)
gh api repos/:owner/:repo/milestones --jq '.[] | "\(.number) \(.title)"'
gh label list --limit 100 | grep -E "severity|format|type:"
git rev-parse --short HEAD                        # 기준 커밋
```

라벨은 전부 이미 존재한다. 없는 라벨을 `--label` 로 주면 생성이 실패한다.

| 분류 | 라벨 |
|---|---|
| 검수 범위 | `qa`, `publication-review` |
| 산출물 | `format:notebook`, `format:wikidocs`, `format:epub` |
| 심각도 | `severity:blocker`, `severity:major`, `severity:minor` |
| 문제 유형 | `type:runtime`, `type:content`, `type:layout`, `type:link`, `type:typo` |

유형 선택이 헷갈릴 때: 목차·절 번호·링크 문제는 `type:link`(설명이 "링크·목차·이동 문제"), 소요 시간·실행 환경은 `type:runtime`, 나머지 내용·수식·코드·출력은 `type:content`.

## 제목

```
[Ch 12][Notebook][Major] "Ch 5 셋업 재현" 주장과 달리 sklearn baseline 설정이 세 곳 모두 다름
[Ch 12][WikiDocs][Minor] 부록 페이지 본문 헤딩 번호(12-3)가 TOC(12-5)와 불일치
```

## 본문 — 계획서 §6.2 의 8개 필드

`## 대상` / `## 실제 결과` / `## 기대 결과` / `## 재현 절차` / `## 로그` / `## 추가 확인` / `## 전파 범위` / `## 심각도` / `## 담당`

지킬 것:

- **대상에 기준 커밋과 환경을 쓴다.** Colab 실행을 안 했으면 "정적 검토 + `executed/`(커밋 SHA) 대조, **Colab 재실행 미수행**" 이라고 그대로 쓴다. 실행 PASS 로 오해되면 안 된다.
- **전파 범위는 각 산출물을 실제로 열어 확인하고 쓴다.** WikiDocs 에 없다고 EPUB 에도 없는 게 아니다. `pages/` 와 `book/chapters/*.tex` 를 따로 grep 한다.
- **재실행 필요 여부를 본문에 적는다.** 수정자가 가장 먼저 알아야 할 비용이다. (SKILL.md 의 수정 경로 표)
- **마크다운으로 쓴다.** 표·코드펜스가 GitHub 에서 렌더링된다. 평문 블록으로 감싸지 말 것.

```bash
gh issue create \
  --title '[Ch 12][Notebook][Major] …' \
  --body-file body.md \
  --label "severity:major" --label "format:notebook" --label "type:content" \
  --label "qa" --label "publication-review" \
  --milestone "2026-09-30 출판 검수 완료"
```

## 챕터 이슈 연결 — 교차참조는 댓글이 아니다

본문에 `#36` 을 쓰면 #36 타임라인에 **`cross-referenced` 회색 줄**이 생긴다. 그건 검수 기록이 아니다. 계획서 §9 가 요구하는 결과 기록은 따로 남긴다.

```bash
gh issue comment 36 --body-file result.md
```

이슈 번호는 생성 후에 부여되므로 **결함 이슈를 모두 만든 뒤** 번호를 모아 챕터 이슈 댓글의 `발견 이슈:` 를 채운다.

## 수정 기록

수정 후 각 결함 이슈에 댓글로 남긴다. 담을 것:

- 어느 안을 골랐는지와 **그 근거** (이슈 댓글에 다른 의견이 있었다면 어떻게 반영했는지)
- 브랜치·커밋 SHA
- 변경 지점 (파일:줄 또는 노트북 셀)
- **재실행 필요 여부**와 이유
- 범위를 넘어 고친 게 있으면 명시하고 분리 여부를 묻는다
- 마무리는 "등록자 확인 후 닫아 주세요" — 계획서 §6.2 는 수정자가 스스로 닫지 못하게 한다

PR 은 결함 이슈에만 `Fixes #NN` 을 걸고, 챕터 이슈는 `Related to #36` 으로 둔다. 챕터 이슈를 `Fixes` 로 걸면 수정 하나가 병합될 때 챕터 전체가 너무 일찍 닫힌다.

## 틀린 내용을 올렸다면

정정 댓글을 따로 단다. 본문을 조용히 고치지 않는다 — 이미 읽은 사람이 있고, 수정자가 잘못된 전파 범위를 보고 헛수고할 수 있다. 무엇이 틀렸고 무엇이 맞는지, 그래서 작업 범위가 어떻게 줄거나 느는지를 쓴다.
