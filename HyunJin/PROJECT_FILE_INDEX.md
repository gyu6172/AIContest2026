# 📁 프로젝트 파일 경로 정리 (walkthrough.md 기준)

> [walkthrough.md](디지털경진대회/walkthrough.md) 에서 참조한 모든 파일의 **실제 저장 경로**를 정리한 문서입니다.
> 정본(canonical) 소스 코드 폴더는 **`lora_model/my_code_0514from0508/`** 입니다.

---

## 1. 소스 코드 (정본: `lora_model/my_code_0514from0508/`)

walkthrough 부록 "프로젝트 파일 구조"에 나온 `.py` 파일 전부 이 폴더에 존재합니다.

| 파일 | 역할 | 경로 |
|------|------|------|
| `preprocess.py` | 프롬프트 빌더, HTML 파서, 유틸 | `lora_model/my_code_0514from0508/preprocess.py` |
| `retrieval.py` | Few-shot 예제 검색기 (임베딩 기반) | `lora_model/my_code_0514from0508/retrieval.py` |
| `train.py` | SFT 학습 (LoRA + OOF) | `lora_model/my_code_0514from0508/train.py` |
| `build_dpo_dataset.py` | DPO 데이터셋 생성 (Hard Negative Mining) | `lora_model/my_code_0514from0508/build_dpo_dataset.py` |
| `train_dpo.py` | DPO 학습 (IPO + β=0.3 + label_smoothing) | `lora_model/my_code_0514from0508/train_dpo.py` |
| `inference.py` | 기본 추론 | `lora_model/my_code_0514from0508/inference.py` |
| `inference_tta.py` | TTA 추론 (후보 셔플 다수결) | `lora_model/my_code_0514from0508/inference_tta.py` |
| `inference_prompt_ensemble.py` | 3-Persona 앙상블 추론 | `lora_model/my_code_0514from0508/inference_prompt_ensemble.py` |
| `inference_lora_router.py` | 3-Way LoRA 라우팅 추론 | `lora_model/my_code_0514from0508/inference_lora_router.py` |

---

## 2. 데이터

| 파일 | 설명 | 경로 |
|------|------|------|
| `train.csv` | 학습 데이터 (10,307건) | `data/train.csv` |
| `test.csv` | 테스트 데이터 (4,417건) | `data/test.csv` |
| `dpo_dataset.json` | DPO 데이터셋 (2,979쌍) | `data/dpo_dataset.json` |

---

## 3. 제출 파일 (메트릭 이력표 기준: `submission/`)

| # | 제출 파일 | Exact Match | 방법 | 경로 |
|---|-----------|-------------|------|------|
| 1 | submission (7).csv | ~0.76 | SFT 단독 | `submission/submission (7).csv` |
| 2 | submission_tta | ~0.76 | + TTA | `submission/submission_tta (1).csv` |
| 3 | submission_prompt_ensemble | ~0.76 | + 3-Persona | `submission/submission_prompt_ensemble.csv` |
| 4 | **submission_super_ensemble_v2** | **0.7644** | **최종 최고점** | `submission/submission_super_ensemble_v2.csv` |
| 5 | submission_lora_router | 0.6455 | DPO 1차 ❌ | `submission/submission_lora_router.csv` |
| 6 | submission_dpo_merged | ~0.65 | DPO 2차 IPO ❌ | `submission/submission_dpo_merged.csv` |

---

## 4. LoRA 가중치

| 항목 | 경로 |
|------|------|
| SFT LoRA 가중치 | `lora_model/my_code_0514from0508/lora_model/` |

---

## ⚠️ walkthrough 부록과 실제 파일의 불일치 (3건)

1. **`inference_dpo_merge.py`**
   - 부록: `my_code_0514from0508/` 에 있다고 표기.
   - 실제: 정본 폴더(`lora_model/my_code_0514from0508/`)엔 **없음**. 루트 레벨 별도 폴더 `my_code_0514from0508/inference_dpo_merge.py` 에만 존재.

2. **`super_ensemble.py`**
   - 부록 파일 목록엔 **없음**.
   - 실제 존재: `submission/super_ensemble.py`.

3. **`lora_model_dpo/`**
   - 부록에 DPO 가중치 폴더로 표기.
   - 실제: repo 어디에도 **없음** (미존재).

---

## 참고: 소스 코드 중복본 위치

동일 `.py` 파일이 여러 폴더에 중복 존재합니다. walkthrough 기준 **정본은 `lora_model/my_code_0514from0508/`** 이며, 아래는 과거 스냅샷/백업입니다.

- `src/`, `src2/`, `src3/`
- `my_code_20260508/`
- `temp_0508/`
- `outputs/run_analysis/run_analysis/src_snapshot/`
