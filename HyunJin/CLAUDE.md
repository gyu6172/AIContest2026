# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 대회 개요

웹 UI 에이전트 다음 행동 예측. Input: task + history + cleaned_html + candidate_elements(15개). Output: op(CLICK/TYPE/SELECT) + target_id + value. Metric: Exact Match (3필드 전부 일치). Train/Test: 10,307 / 4,417행, 97개 site_token. 제출 한도 100회.

op 분포: CLICK 55.6% / TYPE 25.8% / SELECT 18.6%

현재 최고 성능: Exact 0.7516, Target 0.7825, Op 0.9728, Value 0.9627 (35회 제출)

---

## 명령어

```bash
# 학습 (단일)
python src/train.py

# 학습 (OOF 3-Fold → 앙상블용 lora_model_fold_0~2 생성)
python src/train.py --oof

# 학습 플래그
# --e2  real_web 증강 2배 부스트
# --e3  compact 후보 포맷 사용

# 추론 (단일 lora_model/)
python src/inference.py

# 추론 (앙상블 다수결)
python src/inference.py --ensemble

# 추론 플래그
# --e3  compact 후보 포맷 (train --e3와 맞춰야 함)

# 진단
python scripts/diagnostic.py   # 프롬프트 길이 / 타깃 유효성
python scripts/check_target.py # default candidate 비율
```

---

## 아키텍처

### 파이프라인 흐름

```
data/train.csv ──> ExampleRetriever().build()  (RAG 인덱스)

data/test.csv 행
  └─> build_prompt()               [RAG + A-Tree + 압축 History + Reasoning Guidelines]
        └─> LLM (Qwen3.6-35B LoRA + thinking)
              │ target_id 무효 시
              ▼
          fallback_rule_based()    [tag→op 규칙, score 기반 후보 선택]
              │
              ▼
          enforce_consistency()    [op/tag/value 가드레일]
              │
              ▼
         submission.csv + artifacts/inference_report.json
```

### HTML 타입 분기

`detect_html_type(row)` → `"workflow"` 또는 `"real_web"`. 판별 기준: HTML 내 `workflow-context` / `completed-fields` 키워드. Workflow 행은 프롬프트에 `[Workflow Status]` 블록(step 진행도, completed fields) 추가.

### 프롬프트 구조 (`build_prompt`)

1. **Universal Rerank**: CrossEncoder(`cross-encoder/ms-marco-MiniLM-L-6-v2`)로 후보 task 관련도 순 정렬
2. **RAG**: `ExampleRetriever.query()` → 유사 과거 예시 k=5개
3. **A-Tree 포맷팅**: BeautifulSoup으로 HTML 파싱 → 후보의 조상 경로+형제 노드를 인덴트 트리로 표현 (`format_as_tree`)
4. **History 압축**: AgentOccam 방식, `[tag]text→OP:value` 형태로 60~70% 토큰 절약
5. **Reasoning Guidelines**: 5단계 논리 가이드 (task 이해 → 요소 분석 → 대비 → 일관성 → 출력)
6. 출력: `<think>...</think>` + JSON `{"op", "choice", "value"}`

---

## 모델/학습 설정

| 항목 | 값 |
|------|-----|
| Base | `unsloth/Qwen3.6-35B-A3B-4bit` |
| LoRA | r=16, alpha=32, 7개 모듈(`q,k,v,o,gate,up,down_proj`) |
| 학습 모드 | **DPO 기본** (`USE_DPO = True`), SFT는 플래그로 전환 |
| DPO | beta=0.1, 1 epoch, lr=5e-5, bf16, adamw_8bit, cosine |
| SFT | 1 epoch, lr=1e-4, same optimizer |
| max_seq_length | 학습 8192 / 추론 16384 |
| Inference batch | 64 |
| max_new_tokens | 512 (thinking + CoT + JSON) |

**학습 플래그 위치**: `src/train.py` 상단 런타임 플래그 블록. 변경이 필요한 건 여기서만.

```python
FINAL_TRAIN_ON_FULL_DATA = False  # True로 바꾸면 검증 없이 전체 데이터 학습
USE_DPO = True                    # False → SFT 모드
EXCLUDE_SITES = {"site_2aa627db"} # CLICK 편향 96.5%, test에 없음 → 제외
```

---

## DPO 학습 데이터 생성 (WEPO 프레임워크)

`prepare_training_data()`:

- **Chosen**: 정답 후보 + CoT reasoning (`generate_cot_reasoning`)
- **Rejected**: `find_lca_hard_negative()` — DOM 거리(LCA) + CrossEncoder 점수 복합 스코어로 "가장 헷갈리는 오답" 선택. False-negative 필터(text/label 동일 후보 제외) 적용
- **f_op 휴리스틱**: TYPE/SELECT 정답일 때 33% 확률로 rejected_op → CLICK 변조 (기능 변별력 추가)
- **셔플 증강**: `hard_negative_shuffle()` — 랜덤 셔플 후 정답과 동일 tag의 hard negative를 정답 바로 앞에 배치. real_web은 3배 더 많이 셔플

---

## 주요 함수 관계

| 함수 | 위치 | 역할 |
|------|------|------|
| `build_prompt` | preprocess.py | 프롬프트 조립 (rerank → RAG → A-Tree → 압축 history) |
| `enforce_consistency` | preprocess.py | op/target_id/value 가드레일. `_task` 키를 임시로 받고 반환 전 제거 |
| `fallback_rule_based` | preprocess.py | LLM 실패 시 tag→op 규칙 + 점수 기반 후보 선택 |
| `extract_value_from_task` | preprocess.py | SELECT options 매칭 → 따옴표 → label 기반 TYPE → 날짜 regex 순 |
| `ExampleRetriever.query` | retrieval.py | (site, sig, html_type) 5단계 fallback으로 예시 검색 |
| `find_lca_hard_negative` | preprocess.py | DPO rejected 샘플 선택 (LCA + CrossEncoder) |
| `format_as_tree` | preprocess.py | HTML → 후보 중심 인덴트 A-Tree |
| `_compress_history` | preprocess.py | History 토큰 압축 |
| `detect_html_type` | preprocess.py | workflow vs real_web 분기 |

---

## Consistency Guard 규칙 (`enforce_consistency`)

- invalid op → CLICK
- invalid target_id → fallback_rule_based → 첫 번째 후보 (최후 안전망)
- CLICK + select 태그 → options 추출 성공 시 SELECT 업그레이드, 실패 시 button/a로 교체
- SELECT value → options 중 task와 가장 일치하는 것, 실패 시 첫 번째 옵션
- CLICK value → 항상 `""`

---

## Retriever 인덱스 구조

5단계 fallback 쿼리 (우선순위 순):
1. `(site, history_sig, html_type)` — 가장 정확
2. `(site, html_type)`
3. `(sig, html_type)`
4. `(site, sig)` — html_type 무관
5. `(site)` — 최종 fallback

Jaccard 유사도(`_jaccard`)로 task 기반 재정렬 후 top-k 반환.

---

## 데이터 스키마

| 컬럼 | 설명 |
|------|------|
| `site_token` | 사이트 식별자 (97개) |
| `task` | 자연어 목표 (value 95.85% 확률로 포함) |
| `history` | `Step N: [tag] label -> OP[: value]` |
| `cleaned_html` | 현재 페이지 HTML |
| `candidate_elements` | 15개 요소 JSON (`candidate_id`, `tag`, `text`, `attrs`) |
| `op / target_id / value` | 예측 대상 |

`attrs` 포맷: `key=value | key=value`. SELECT의 options: `options=A / B / C`.

---

## 제출 체크리스트

- [ ] `somenna_submission.csv` 행 수 일치
- [ ] op ∈ {CLICK, TYPE, SELECT}
- [ ] CLICK의 value = `""`
- [ ] target_id가 해당 row의 candidate 중 하나
- [ ] 결측값 없음 (`fillna("")`)
- [ ] artifacts/inference_report.json 확인 (fallback 비율, guard 통계)

---

## 다음 개선 후보

- TYPE value 추출 정밀도 향상
- OOF error analysis → 오류 패턴 타겟 개선
- 최종 제출 전 `FINAL_TRAIN_ON_FULL_DATA = True`로 전체 데이터 학습
