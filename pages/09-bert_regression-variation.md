직접 다 돌리진 않고 *어떤 인자를 바꾸면 무슨 일이 일어나는지* 짚습니다.

| 바꾸는 인자 | 자주 보는 결과 |
|---|---|
| `num_train_epochs=5` (더 오래) | 학습 loss는 더 줄지만 eval은 정체/악화 (overfitting) |
| `learning_rate=2e-4` (10배 큼) | 학습 초반에 loss가 발산하거나 nan으로 감 |
| `learning_rate=2e-7` (100배 작음) | 학습이 거의 안 됨 (loss가 안 줄어듦) |
| `batch_size=4` | step 수 증가, 학습 시간 길어짐, gradient 잡음 큼 |
| `batch_size=64` | T4에서 OOM 위험 (max_length=128 + DistilBERT는 32까지가 안전) |
| `fp16=False` | VRAM 2배, 속도 느려짐, 결과는 비슷 |
| `max_length=512` | 시퀀스 길이가 4배라 attention 비용 16배 — T4 30분 초과 |

이 표가 BERT 파인튜닝의 *기본 안전대* 입니다. Ch 10 이후 모든 학습 챕터에서도 동일한 인자 범위에서 움직입니다.
