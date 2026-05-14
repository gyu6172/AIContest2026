# 2026-05-12 DPO Hard Negative 리팩토링 계획

## 1. 결론

`Project01보고서.pdf` 계열 자료의 핵심 아이디어는 우리 프로젝트에 일부 적용할 가치가 있다. 다만 보고서의 `hallucination` 라벨 구조를 그대로 가져오면 안 된다.

보고서 방식은 RAG/QA 문제에 맞춰져 있다.

- supported answer vs abstain
- insufficient answer vs hallucinated answer
- plausible but wrong hard negative

우리 프로젝트는 웹 UI action prediction 문제다.

```json
{"op": "CLICK|TYPE|SELECT", "choice": 1, "value": "..."}
```

따라서 우리에게 필요한 것은 `hallucination` 라벨이 아니라, 웹 UI grounding에 맞춘 `negative_type` 또는 `error_type`이다.

추천 적용 순서:

1. LCA false-negative filter
2. LCA + task relevance hybrid scoring
3. rejected pair에 `negative_type` 로깅
4. validation에서 효과 확인
5. 효과가 있으면 HNM weighted loss 도입
6. EMA reference는 보류

---

## 2. 보고서 방식의 객관적 평가

### 2.1 Hallucination 라벨은 SOTA인가?

`hallucination`을 명시적으로 라벨링하거나 hallucinated response를 rejected로 사용하는 DPO 계열은 RAG/QA/멀티모달 factuality 분야에서 강한 연구 흐름이다.

최근 연구 흐름:

- OPA-DPO: hallucinated response와 expert-corrected response를 on-policy preference pair로 구성
- HDPO: hallucination 원인별 preference pair 구성
- TPO: token-level preference로 hallucination 완화
- F-DPO: factuality label을 DPO에 직접 반영

하지만 `hallucination 라벨이 있음 = 모든 분야의 SOTA`는 아니다. 정확히는 다음과 같이 보는 것이 맞다.

> Hallucination/factuality label을 preference pair 설계에 반영하는 것은 최신 DPO 연구들의 중요한 구성 요소 중 하나다. 그러나 웹 UI action prediction에서는 hallucination보다 grounding error taxonomy가 더 직접적이다.

### 2.2 보고서에서 가져올 수 있는 핵심

보고서에서 가장 유용한 부분은 `hallucination`이라는 라벨명이 아니라, DPO pair를 난이도별로 구분하고 어려운 pair에 더 큰 weight를 주는 HNM 방식이다.

보고서 방식:

```text
gold_vs_abstain              -> weight 1.0
abstain_vs_hallucination     -> weight 1.5
abstain_vs_hard_negative     -> weight 2.0
```

우리 프로젝트에 맞춘 변환:

```text
fallback_random_negative     -> 낮은 weight
lca_hard_negative            -> 기본 weight
hybrid_hard_negative         -> 높은 weight
realweb_hard_negative        -> 높은 weight
op_mutated_negative          -> 보조 weight
invalid_json_or_choice       -> 별도 분석용 error_type
```

---

## 3. 우리 프로젝트용 Negative/Error Taxonomy

### 3.1 학습 데이터 생성 시 기록할 `negative_type`

DPO rejected candidate가 어떤 방식으로 만들어졌는지 기록한다.

추천 타입:

```text
lca_hard_negative
hybrid_hard_negative
fallback_random_negative
op_mutated_negative
realweb_hard_negative
workflow_hard_negative
false_negative_filtered
```

의미:

- `lca_hard_negative`: DOM/LCA 거리 기준으로 선택된 오답
- `hybrid_hard_negative`: LCA + task relevance + tag similarity 기준으로 선택된 오답
- `fallback_random_negative`: DOM 파싱 실패 또는 후보 부족으로 random fallback된 오답
- `op_mutated_negative`: TYPE/SELECT 정답을 CLICK rejected로 변조한 경우
- `realweb_hard_negative`: real_web sample에서 생성된 hard negative
- `workflow_hard_negative`: workflow sample에서 생성된 hard negative
- `false_negative_filtered`: gold와 너무 유사해 rejected에서 제외된 후보가 있었음

### 3.2 평가/분석 시 기록할 `error_type`

추론 결과 분석용으로 별도 분류한다.

추천 타입:

```text
invalid_json
invalid_choice
wrong_target
wrong_op
wrong_value
unsupported_value
empty_target_id
fallback_rule_used
```

의미:

- `invalid_json`: 모델 출력 JSON 파싱 실패
- `invalid_choice`: choice가 1~15 범위를 벗어남
- `wrong_target`: target candidate 선택 실패
- `wrong_op`: CLICK/TYPE/SELECT 선택 실패
- `wrong_value`: TYPE/SELECT value 추출 실패
- `unsupported_value`: task나 candidate option에 없는 값을 생성
- `empty_target_id`: target_id가 비어 있음
- `fallback_rule_used`: LLM 결과 대신 rule fallback이 사용됨

---

## 4. 1단계 리팩토링: LCA False-Negative Filter

### 목표

정답과 지나치게 비슷해서 실제로는 false negative일 수 있는 후보를 rejected에서 제외한다.

### 적용 위치

- `src/preprocess.py`
- `find_lca_hard_negative(row, candidates, target_id)`

### 추천 helper

```python
def _is_false_negative(target_cand: dict, cand: dict) -> bool:
    ...
```

### 권장 필터 정책

강한 제외 조건:

- `text`가 비어 있지 않고 완전히 동일
- `label`이 비어 있지 않고 완전히 동일
- `aria-label`이 비어 있지 않고 완전히 동일

주의해서 다룰 조건:

- `name` 동일
- `placeholder` 동일

`name`과 `placeholder`는 단독 일치만으로 제외하지 않는다. 같은 `placeholder=YYYY-MM-DD` 같은 값이 여러 input에서 반복될 수 있기 때문이다.

추천 정책:

```text
if text exact match:
    exclude
if label exact match:
    exclude
if aria-label exact match:
    exclude
if two or more weak attrs match:
    exclude
```

약한 attr:

```text
name
placeholder
type
```

### 성공 기준

- rejected 후보가 모두 사라져 skip되는 샘플이 늘지 않을 것
- fallback random 비율이 과도하게 증가하지 않을 것
- real_web target_id_acc가 유지 또는 상승할 것

---

## 5. 2단계 리팩토링: LCA + Task Relevance Hybrid Scoring

### 목표

LCA 거리만 보지 않고, task와의 의미적 관련성도 반영해 더 좋은 hard negative를 선택한다.

### 문제점

현재 LCA-only 방식은 다음 한계가 있다.

- DOM에서 가까워도 의미적으로 쉬운 오답일 수 있음
- DOM에서 멀어도 task상 헷갈리는 오답일 수 있음
- 같은 거리일 때만 `_candidate_match_score()`를 tie-break로 사용함

### 추천 scoring

점수 scale을 맞춰야 한다. 단순히 `20 - dist`를 쓰면 DOM 점수가 너무 커져서 task score가 거의 무시된다.

추천:

```text
dist_score  = max(0, 20 - dist) / 20
match_score = min(_candidate_match_score(c, task), 3.0) / 3.0
tag_bonus   = 0.1 if same_tag else 0.0

final_score = dist_score + match_score + tag_bonus
```

주의:

- `tag_bonus=1.0`은 너무 클 수 있다.
- score 정규화 없이 `20 - dist`를 쓰면 LCA-only와 거의 다르지 않다.
- match score가 너무 큰 후보는 false negative일 수 있으므로 filter를 먼저 적용한다.

### 성공 기준

- real_web target_id_acc 상승
- workflow 성능 유지
- fallback random 비율 증가 없음
- DPO 데이터 생성 개수 감소 없음

---

## 6. 3단계 리팩토링: negative_type 로깅

### 목표

각 DPO pair가 어떤 negative 전략으로 생성되었는지 기록한다.

### 이유

나중에 HNM weighted loss를 적용하려면 pair별 난이도 정보가 필요하다. 바로 weighted loss를 구현하지 않더라도 `negative_type`을 기록해두면 분석과 ablation이 쉬워진다.

### Dataset field 예시

```python
{
    "prompt": full_prompt,
    "chosen": chosen_output,
    "rejected": rejected_output,
    "negative_type": "hybrid_hard_negative",
    "difficulty_weight": 1.5
}
```

단, 현재 `TRL DPOTrainer`가 추가 column을 어떻게 처리하는지 확인해야 한다. 문제를 피하려면 1차로 `negative_type`은 별도 분석 로그에만 남기고, 학습 dataset에는 넣지 않는 선택지도 있다.

---

## 7. 4단계 리팩토링: HNM Weighted DPO

### 목표

어려운 negative pair에 더 큰 gradient를 부여한다.

### 보고서 방식

보고서의 HNM은 loss를 단순 평균하지 않고 weighted mean으로 계산한다.

```python
unweighted_loss = -F.logsigmoid(rewards)
weighted_loss = unweighted_loss * difficulty_weight
loss = weighted_loss.sum() / difficulty_weight.sum()
```

### 우리 프로젝트용 weight 초안

```text
fallback_random_negative  = 0.7
workflow_hard_negative    = 1.0
lca_hard_negative         = 1.0
op_mutated_negative       = 1.2
realweb_hard_negative     = 1.5
hybrid_hard_negative      = 1.5
```

주의:

- weight를 너무 크게 주면 JSON 형식 안정성이 흔들릴 수 있다.
- 처음부터 2.0 이상을 쓰지 않는다.
- real_web이 현재 병목이므로 real_web hard negative에 약간 더 weight를 주는 것은 합리적이다.

### 구현 리스크

현재 프로젝트는 `trl.DPOTrainer`를 직접 사용한다. Weighted DPO를 하려면 다음 중 하나가 필요하다.

1. `DPOTrainer` subclass로 loss 계산 override
2. TRL이 sample weight를 지원하는지 확인
3. difficulty별 oversampling으로 간접 구현

가장 안전한 1차 접근은 custom trainer가 아니라 `hybrid_hard_negative` sample을 약간 더 많이 생성하는 oversampling이다. 단, 데이터 분포 왜곡이 생길 수 있으므로 ablation이 필요하다.

---

## 8. EMA Reference는 보류

### 이유

보고서에서도 EMA 단독 성능 개선은 명확하지 않았다.

또한 우리 프로젝트는 다음 구조를 사용한다.

- Qwen3-8B
- LoRA
- 4bit quantization
- Unsloth
- TRL `DPOTrainer`

EMA reference를 넣으려면 trainer 내부와 reference log-prob 계산을 크게 바꿔야 한다. 구현 복잡도와 리스크에 비해 기대 이득이 작다.

보류 사유:

- JSON format 안정성에 악영향 가능
- reference anchor가 약해질 수 있음
- custom trainer 구현 필요
- 메모리/속도 부담 증가
- 보고서 결과에서도 EMA 개선이 작거나 없음

결론:

> EMA는 지금 적용하지 않는다. HNM까지 검증한 뒤에도 DPO 불안정성이 크면 그때 재검토한다.

---

## 9. 검증 계획

### 필수 비교군

```text
SFT only
DPO LCA baseline
DPO LCA + false-negative filter
DPO hybrid hard negative
DPO hybrid + negative_type logging
DPO hybrid + HNM weighted loss
```

### 핵심 metric

```text
Exact Match
target_id_acc
op_acc
value_acc
workflow EM
real_web EM
JSON parse failure rate
fallback random ratio
filtered false negative count
```

### 성공 기준

- real_web target_id_acc 상승
- workflow 성능 유지
- op/value accuracy 하락 없음
- JSON parse failure 증가 없음
- 학습 데이터 skip 비율 증가 없음

---

## 10. 최종 우선순위

가장 안전한 실행 순서:

```text
P0. 현재 DPO baseline 유지
P1. false-negative filter 추가
P2. LCA + task relevance hybrid scoring 추가
P3. negative_type / filtered_count 로깅
P4. validation ablation
P5. HNM weighted loss 또는 oversampling 검토
P6. EMA reference 보류
```

핵심 판단:

> 보고서의 `hallucination` 라벨은 우리 프로젝트에 그대로 맞지 않는다. 하지만 "DPO pair의 난이도를 구분하고 어려운 pair에 더 큰 학습 신호를 준다"는 HNM 아이디어는 우리 프로젝트에도 적용 가치가 있다.

