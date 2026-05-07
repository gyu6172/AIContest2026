# 디지털경진대회 — Web Agent Action Prediction

> 본 문서는 현재 `src/` 코드 상태를 반영한 마스터 인덱스입니다.
> 세부 사항은 각 파일/문서 링크를 참조하세요.
> *최신화: 2026-05-06*

---

## 0. 대회 개요

| 항목 | 내용 |
|------|------|
| Task | 웹 UI 에이전트의 다음 행동 예측 |
| Input | task + history + cleaned_html + candidate_elements(15개) |
| Output | op (CLICK/TYPE/SELECT) + target_id + value |
| Metric | Exact Match (3개 필드 모두 일치) |
| Train / Test | 10,307 / 4,417행 |
| 고유 site_token | 97개 |
| 제출 제한 | 100회 |

op 분포: CLICK 55.6% / TYPE 25.8% / SELECT 18.6%

---

## 1. 데이터 스키마

| 컬럼 | 설명 |
|------|------|
| id | `aac_mix_train_XXXXXX` |
| site_token | 사이트 식별자 (97개) |
| task | 작업 목표 (자연어, 입력값 포함) |
| history | 완료된 스텝들 — `Step N: [tag] label -> OP[: value]` |
| cleaned_html | 현재 페이지 HTML |
| candidate_elements | 15개 요소 JSON 배열 (`candidate_id`, `tag`, `text`, `attrs`) |
| op / target_id / value | **예측 대상** |

**주의**: SELECT 후보의 `attrs`에는 `options=A / B / C` 형태로 선택지 포함.

---

## 2. 핵심 데이터 인사이트

- **Value 추출 가능성 95.85%**: TYPE/SELECT의 value 대부분이 task 문장에 그대로 포함 → regex+fuzzy로 추출
- **요소 텍스트 매칭 21.21%**: 정답 요소 text가 task에 직접 등장하는 비율은 낮음 → 의미론적 매칭 필요 (LLM/retrieval로 처리)
- **Candidates 고정 15개**: 15-class classification으로도 접근 가능
- **Tag→Op 거의 결정적**: button→CLICK, input→TYPE, select→SELECT

---

## 3. 현재 파이프라인 (실제 코드 기준)

```
data/train.csv ─┬─> build_empirical_priors()
                └─> ExampleRetriever().build()

data/test.csv 행 ──> [LoRA LLM (Qwen2.5-3B)]
                         │ (target_id invalid 시)
                         ▼
                     fallback_rule_based()
                         │
                         ▼
                     enforce_consistency()
                         │
                         ▼
                    submission.csv
```

**핵심 변경 사항** (이전 문서 대비):
- Site-Template 기반 Stage 1 매핑은 **제거됨**. 대신 priors는 LLM 프롬프트의 weak tie-break 컨텍스트로만 사용.
- LLM-first 라우팅: 모델이 있으면 모든 행이 LLM으로 들어감.
- 잘못된 op/target_id/value 자동 교정용 `enforce_consistency` 단계 추가.

→ 상세: [`docs/PIPELINE_CONTEXT.md`](docs/PIPELINE_CONTEXT.md), [`docs/review.md`](docs/review.md)

---

## 4. 모듈 구성 (`src/`)

| 파일 | 역할 |
|------|------|
| `preprocess.py` | empirical priors, value 추출, rule-based fallback, consistency guard |
| `retrieval.py` | `(site_token, history_signature)` 기반 유사 예시 검색기 |
| `train.py` | Unsloth + TRL SFT, LoRA 학습, site_token 그룹 검증 |
| `inference.py` | priors+retriever 빌드 → LLM 추론 → fallback → submission.csv |

진단 스크립트(`scripts/`): `diagnostic.py` (프롬프트 길이/타깃 유효성 점검), `check.py` (default candidate 비율 점검)

---

## 5. 모델/학습 설정

- **Base**: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- **LoRA**: r=16, target=`q,k,v,o,gate,up,down_proj` (7개)
- **SFT (L4 기준)**: batch=4, grad_accum=4, max_steps=4000, lr=2e-4, optim=adamw_8bit, cosine scheduler
- **Augmentation**: Candidate 순서 셔플 2회 (원본+셔플2 = 3배 데이터)
- **max_seq_length**: 2048
- **Inference (L4 기준)**: batch=8, `do_sample=False`, `max_new_tokens=64`
- **학습 모드 2가지**:
  - 단일: `python src/train.py` (기존 GroupShuffleSplit)
  - **OOF**: `python src/train.py --oof` (3-Fold GroupKFold → 앙상블)
- **추론 모드 2가지**:
  - 단일: `python src/inference.py` (lora_model/ 사용)
  - **앙상블**: `python src/inference.py --ensemble` (lora_model_fold_0~2 다수결)

---

## 6. LLM 프롬프트 구성

다음을 포함:
- Task / History
- Retrieval로 가져온 유사 과거 예시 (k=2)
- Empirical priors (op 분포, P(op|tag), site별 빈출 패턴)
- 15개 후보 요소 (attrs 포함)
- JSON-only 출력 지시

명시적 가드: "예시의 target_id는 현재 행에 유효하지 않다", "task와 현재 후보가 priors/history와 충돌하면 task를 우선하라".

---

## 7. Consistency Guard 규칙

`enforce_consistency(pred, candidates)`:

- 잘못된 op → CLICK
- 잘못된 target_id → fallback rule
- TYPE이 input/textarea가 아닌 후보 → 적절한 input으로 교체 또는 다운그레이드
- SELECT가 select가 아닌 후보 → 적절한 select로 교체 또는 다운그레이드
- CLICK이 select 후보이고 옵션 추출 가능 → SELECT로 업그레이드
- SELECT의 value → 반드시 options 중 하나
- CLICK의 value → 항상 빈 문자열

---

## 8. Value 추출 우선순위

`extract_value_from_task(task, op, attrs)`:
1. CLICK → 빈 문자열
2. SELECT → `options=` 중 task에 등장하는 옵션
3. 따옴표 패턴
4. TYPE → label/placeholder/aria-label 뒤따르는 청크
5. 날짜 regex `\d{4}-\d{2}-\d{2}`
6. 빈 문자열

→ 상세 로직: 코드 (`src/preprocess.py`) 및 git log 참조

---

## 9. 작업 디렉토리 상태

```
HyunJin/
├── CLAUDE.md                         # (이 문서) 마스터 인덱스
├── colab_pipeline.ipynb              # Colab 실행 노트북
├── docs/
│   ├── PIPELINE_CONTEXT.md           # 코드 기반 상세 컨텍스트
│   ├── review.md                     # 흐름 리뷰
│   └── GPU_PROFILES.md               # T4/A100 프로필 스위칭
├── src/
│   ├── preprocess.py / retrieval.py
│   └── train.py / inference.py
├── scripts/
│   └── check.py / diagnostic.py      # 진단 스크립트
├── artifacts/
│   └── submission.csv / submission_pre_patch.csv
├── data/                             # train/test/somenna_submission.csv
├── lora_model/                       # 학습된 LoRA 어댑터
└── outputs/                          # 체크포인트
```

---

## 10. 제출 체크리스트

- [ ] somenna_submission.csv 행 수와 일치
- [ ] op은 CLICK/TYPE/SELECT만
- [ ] CLICK의 value는 ""
- [ ] target_id는 해당 row의 candidate 중 하나
- [ ] 결측값 없음 (`fillna("")` 적용됨)

---

## 11. 다음 개선 후보

- TYPE value 추출 정밀도 향상
- prompt/prior calibration
- fallback의 후보 랭킹 개선
- OOF 결과 기반 Error Analysis → 타겟 개선
- 최종 제출 전 full data 학습 (`FINAL_TRAIN_ON_FULL_DATA = True`)
