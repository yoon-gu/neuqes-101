# Codex 핸드오프 노트

이 파일은 Claude → codex 로의 의도적 핸드오프 메시지를 담습니다. codex가 master 를 sync 한 뒤 한 번 훑어보면 됩니다. 처리 끝난 항목은 사용자(또는 codex)가 지워도 무방.

---

## 2026-05-04 — sklearn 1.5+ API 호환성 정리

**무엇이 바뀌었나** — 챕터 노트북·빌드 스크립트·폴더 README 전반에서 *deprecated/제거* 된 sklearn 인자 사용 제거. 코드는 모던 API로 동작하도록 정리하고, 설명문에서도 *역사적 deprecation 언급* 을 모두 빼고 *현재 동작* 만 풀어쓰기로 통일.

**영향 받은 챕터**: Ch 3 (예고문), Ch 4 (코드+FAQ), Ch 5 (코드+설명), Ch 6 (코드+FAQ Q7), Ch 7·10·11 (추적표 표기), Ch 11 (FAQ Q4 phrasing)

**핵심 룰 (앞으로 모든 챕터에 적용)**:

- `LogisticRegression(multi_class="multinomial")` ❌ → `LogisticRegression()` ✅
  - 이유: sklearn 1.5+ 에서 `multi_class` 인자가 `LogisticRegression` 에서 deprecated 됐고 1.7+ 에선 완전 제거. 모던 sklearn 은 데이터의 클래스 개수(K=2 binary / K≥3 multi-class)로 자동 분기.
  - 코드만 바꾸고 *왜 이 인자를 안 쓰는지의 역사적 설명은 노출하지 않음*.
- `LogisticRegression(multi_class="ovr")` ❌ → `OneVsRestClassifier(LogisticRegression())` ✅
  - OvR 학습은 wrapper class 로만 표현됨. multi-class·multi-label 양쪽에 통하는 표준 패턴.
- `roc_auc_score(..., multi_class="ovr")` ✅ — *이건 다른 API* 로 sklearn 1.8 도 지원. **변경 없음**.

**책(book/) 측 작업 안내**:

- 챕터 .tex 재생성 (`book/tools/notebook_to_tex.py`) 한 번 돌리시면 새 노트북 본문이 자동 반영됩니다.
- 주의 깊게 봐야 할 자리:
  - Ch 4 §실습 코드 블록 (방식 A/B 학습 셀) — 주석이 짧아짐
  - Ch 4 FAQ Q5 — 제목·본문 모두 다시 쓰여짐 ("multi_class 인자는 왜 안 쓰나요?" → "softmax 와 OvR 을 어떻게 구분하나요?")
  - Ch 6 FAQ Q7 — 같은 톤으로 단순화. 두 도구 비교 표가 한 컬럼으로 줄어듦
  - Ch 11 FAQ Q4 — phrasing 깔끔해짐
  - 추적표 행: `LogisticRegression()` (multinomial 자동) 형태로 통일

**검증**: 로컬 venv (sklearn 1.8) + nbconvert 로 Ch 3-6 모두 정상 실행 확인. Colab 환경(구 sklearn) 에서도 그대로 돌아감 — 모던 API 가 구버전과 호환되는 *상위 호환* 변경이라.

---

(이 파일은 _drafts/ 안에 있어 출판물·노트북에 노출되지 않습니다. codex 가 확인 끝낸 항목은 지우거나 'done' 표시 해 주시면 됩니다.)
