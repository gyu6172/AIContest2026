# GPU 프로파일 (T4 ↔ A100 전환 가이드)

> Colab GPU가 바뀔 때 이 파일을 보고 값만 바꿔치기. 코드 구조는 동일.
> 변경 위치: `src/train.py`, `src/inference.py` 상단 상수 + SFTConfig.

---

## 빠른 전환표

| 항목 | T4 (16GB, fp16) | A100 (40GB, bf16) | 위치 |
|---|---|---|---|
| `per_device_train_batch_size` | **1** | **8** | train.py SFTConfig |
| `gradient_accumulation_steps` | **16** | **2** | train.py SFTConfig |
| → 유효 배치 | 16 | 16 (동일 유지) | — |
| `max_steps` | 1000 | 1000 (그대로) | train.py SFTConfig |
| `MAX_SEQ_LENGTH` (train) | 2048 | 2048 | train.py:26 |
| `BATCH_SIZE` (inference) | **4** | **16** | inference.py:25 |
| `max_new_tokens` | 128 | 128 | 동일 |
| `dataset_num_proc` | 2 | 4 | train.py SFTConfig |
| fp16/bf16 | fp16 자동 | bf16 자동 | 코드가 `is_bf16_supported()`로 분기 — 손댈 필요 X |
| `optim` | adamw_8bit | adamw_8bit | 동일 |

유효 배치(=per_device × grad_accum)를 **16으로 고정**하면 학습 곡선/하이퍼파라미터(lr 등) 그대로 재현 가능.

---

## A100 적용 — 정확한 수정 지점

### 1) `src/train.py` SFTConfig 안 (현재 286~310줄 근처)

```python
per_device_train_batch_size = 8,    # T4: 1
gradient_accumulation_steps = 2,    # T4: 16
dataset_num_proc = 4,               # T4: 2
```

### 2) `src/inference.py` 상단 상수

```python
BATCH_SIZE = 16   # T4: 4
```

이 두 군데가 끝. 나머지(LoRA r, lr, max_steps, max_seq_length)는 건드리지 않는 게 안전 — 학습 동역학이 바뀜.

---

## T4로 돌아올 때

위 표의 T4 열 값으로 되돌리면 끝. 추가로 확인할 것:

- A100에서 `BATCH_SIZE=16` 으로 학습/추론한 LoRA 어댑터를 T4에서 inference만 돌릴 때도 `BATCH_SIZE=4`로 낮추지 않으면 OOM.
- `bf16/fp16` 분기는 코드가 자동 처리(`torch.cuda.is_bf16_supported()`)이므로 수동 수정 금지.

---

## A100에서 더 짜내고 싶을 때 (선택)

기본값으로도 충분하지만, 시간이 급하면:

- `per_device_train_batch_size = 16, grad_accum = 1` → step 수 줄어 더 빠름. 단 lr 재튜닝 권장(2e-4 → 1.5e-4 정도).
- `SFTConfig(packing=True)` 추가 → 짧은 샘플들을 묶어 throughput ↑. **단 우리 데이터는 instruction-tuning 형식이라 packing이 loss masking을 망칠 수 있어 기본 비추천**. 켜려면 검증 EM 먼저 비교.
- `max_new_tokens = 96` (출력 JSON이 짧음) → inference 약 20~30% 단축.

---

## 메모리 모니터링

학습/추론 중 한 번:
```python
import torch; print(torch.cuda.memory_allocated()/1e9, "GB /", torch.cuda.get_device_properties(0).total_memory/1e9, "GB")
```

A100 40GB에서 위 설정 기준 예상:
- 학습(batch 8, seq 2048, bf16, LoRA r=16, 4bit base): **~22~26 GB**
- 추론(batch 16): **~10~14 GB**

여유 있으면 batch 12로 올려도 됨.

---

## 체크리스트 (런타임 변경 직후)

- [ ] `nvidia-smi`로 GPU 종류 확인
- [ ] 위 표대로 두 파일 수정
- [ ] 첫 step 통과 후 `nvidia-smi` 한 번 더 — VRAM 70% 이하면 안전
- [ ] OOM 발생 시 `per_device_train_batch_size`만 절반으로, `grad_accum`을 두 배로 (유효 배치 유지)
