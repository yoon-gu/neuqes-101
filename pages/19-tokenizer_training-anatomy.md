같은 영어 문장 + 같은 한국어 문장을 4 토크나이저에 통과시켜 토큰 시퀀스를 직접 출력. *알고리즘에 따라 토큰 수가 어떻게 다른지*, *언어에 따라 어떻게 다른지* 동시에 관찰.

```python
SAMPLE_EN = "The food was unforgettable and the service was excellent."
SAMPLE_KO = "이 영화는 정말 재미있어요. 배우들 연기도 훌륭했습니다."


def show_tokens(tok, text, name):
    enc = tok.encode(text)
    tokens = enc.tokens
    print(f"[{name}]  #tokens = {len(tokens)}")
    print(f"  {tokens}")
    # UNK 개수
    unk_count = sum(1 for t in tokens if t == "[UNK]")
    if unk_count:
        print(f"  ! contains {unk_count} [UNK] tokens")
    print()


print("=" * 78)
print(f"ENGLISH sample: {SAMPLE_EN}")
print("=" * 78)
show_tokens(tok_en_wp, SAMPLE_EN, "en WordPiece")
show_tokens(tok_en_wl, SAMPLE_EN, "en WordLevel")

print("=" * 78)
print(f"KOREAN sample: {SAMPLE_KO}")
print("=" * 78)
show_tokens(tok_ko_wp, SAMPLE_KO, "ko WordPiece")
show_tokens(tok_ko_wl, SAMPLE_KO, "ko WordLevel")
```

**▶ 실행 결과**

```text
==============================================================================
ENGLISH sample: The food was unforgettable and the service was excellent.
==============================================================================
[en WordPiece]  #tokens = 15
  ['[CLS]', 'the', 'food', 'was', 'unf', '##orge', '##tt', '##able', 'and', 'the', 'service', 'was', 'excellent', '.', '[SEP]']

[en WordLevel]  #tokens = 10
  ['The', 'food', 'was', '[UNK]', 'and', 'the', 'service', 'was', 'excellent', '.']
  ! contains 1 [UNK] tokens

==============================================================================
KOREAN sample: 이 영화는 정말 재미있어요. 배우들 연기도 훌륭했습니다.
==============================================================================
[ko WordPiece]  #tokens = 12
  ['[CLS]', '이', '영화는', '정말', '재미있어요', '.', '배우들', '연기도', '훌륭', '##했습니다', '.', '[SEP]']

[ko WordLevel]  #tokens = 9
  ['이', '영화는', '정말', '재미있어요', '.', '배우들', '연기도', '[UNK]', '.']
  ! contains 1 [UNK] tokens
```

**해석 가이드**

- **WordPiece (영어)** — `unforgettable` 같은 드문 단어가 *여러 조각* 으로 쪼개짐. `[CLS]`, `[SEP]` 가 자동 부착되어 BERT 입력 그대로 사용 가능.
- **WordLevel (영어)** — `unforgettable` 이 학습 코퍼스에 *충분히 등장* 했다면 1 토큰, 아니면 `[UNK]`. binary 결과.
- **WordPiece (한국어)** — 조사·어미가 `##` prefix 로 분리되어 *어근 + 조사* 구조가 토큰 시퀀스에 보임.
- **WordLevel (한국어)** — 한국어는 *교착어* 라 같은 어근에 다른 조사가 붙은 어절이 모두 *다른 vocab entry* — `재미있어요` / `재미있다` / `재미있는데` 가 전부 별개 토큰. vocab 효율이 매우 낮음.
