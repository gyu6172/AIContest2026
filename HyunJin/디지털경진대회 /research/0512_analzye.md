# 2026-05-12 WEPO-DPO Literature Review 분석 및 프로젝트 적용안

> 대상 문서: `docs/WEPO DPO Literature Review_latex/manuscript.pdf`  
> 관련 구현: `src/preprocess.py`, `src/train.py`  
> 핵심 주제: WEPO 기반 DPO 선호도 학습, LCA Hard Negative, 웹 UI action prediction

---

## 1. 한 줄 결론

PDF의 핵심은 다음과 같다.

> 지금 적용한 LCA 기반 DPO는 방향이 맞다. 다만 LCA 하나만 믿으면 위험하므로, LCA는 싸고 좋은 baseline으로 쓰되 false negative filter와 semantic/model-confidence negative를 섞어야 더 안전하고 SOTA에 가까워진다.

우리 프로젝트 기준으로는 현재 구현을 버릴 필요가 없다. 오히려 `15개 candidate 중 choice를 고르는 구조`와 `chosen/rejected JSON pair를 만드는 DPO 구조`가 WEPO와 잘 맞는다. 다만 LCA-only hard negative는 구조적으로 가까운 오답만 고르기 때문에, 실제로 모델이 헷갈리는 오답이나 의미적으로 비슷한 오답을 놓칠 수 있다.

---

## 2. 중학생도 이해할 수 있는 설명

웹페이지에는 버튼, 입력창, 선택창이 많다. 모델은 다음과 같은 문제를 푼다.

```json
{"op": "TYPE", "choice": 11, "value": "Mina Wilson"}
```

즉, 모델은 "무슨 행동을 할지", "15개 후보 중 몇 번 요소를 고를지", "입력하거나 선택할 값은 무엇인지"를 맞춰야 한다.

DPO는 모델에게 정답과 오답을 나란히 보여주면서 가르치는 방식이다.

- chosen: 좋은 답. 정답 candidate를 고른 JSON
- rejected: 나쁜 답. 헷갈리기 쉬운 오답 candidate를 고른 JSON

예를 들어 task가 "manager 이름을 입력하라"이고 정답이 `Manager` 입력창이라고 하자. 같은 폼 안에 `Supplier`, `Recipe card`, `Photo upload`, `Meal period` 같은 입력창이 있으면 모델은 헷갈릴 수 있다. 이런 오답을 일부러 rejected로 넣으면 모델은 "비슷하게 생긴 입력창 중에서도 task에 맞는 입력창을 골라야 한다"는 것을 배운다.

여기서 LCA는 HTML/DOM 나무에서 두 요소가 얼마나 가까운 친척인지 보는 방법이다. 같은 `<form>` 안에 있는 입력창들은 DOM 나무에서 가까운 친척이다. 그래서 LCA가 가까운 오답은 "모델이 헷갈릴 만한 오답"일 가능성이 있다.

하지만 LCA가 항상 정답은 아니다. 같은 부모 아래 있어도 전혀 다른 역할의 버튼일 수 있고, DOM에서는 멀어도 화면에서는 가까운 버튼일 수 있다. 따라서 LCA는 좋은 출발점이지만, 최종 hard negative 기준으로는 부족하다.

---

## 3. PDF의 핵심 논문 흐름

### 3.1 Tier 1: 현재 방식의 직접 기반

#### WEPO

WEPO는 웹 요소 선택 문제에 DPO를 적용한다. 정답 요소를 chosen으로 두고, 비정답 요소를 rejected로 둔다. 외부 reward model이나 사람이 만든 preference label 없이, 웹 구조를 이용해 rejected를 자동으로 만든다는 점이 중요하다.

우리 프로젝트와의 연결:

- `target_id`에 해당하는 candidate를 chosen으로 사용한다.
- DOM/LCA 거리로 가까운 오답을 rejected로 사용한다.
- rejected의 `op`와 `value`는 가능하면 chosen과 같게 유지해 `choice` 학습에 집중시킨다.

현재 구현은 이 방향과 잘 맞는다.

#### DPO

DPO는 chosen/rejected 쌍을 이용해 모델이 chosen을 더 좋아하도록 학습한다. 별도 reward model 없이 preference pair만 있으면 된다.

우리 프로젝트와의 연결:

- prompt: task, history, cleaned_html 요약, 15개 candidate
- chosen: 정답 JSON
- rejected: 오답 candidate를 고른 JSON
- 목표: 정답 candidate에 대한 log probability를 rejected보다 높게 만든다.

주의점:

- DPO는 JSON 형식을 망가뜨릴 수 있다.
- beta가 너무 크면 모델이 과하게 바뀔 수 있다.
- chosen log-probability와 JSON parse success를 같이 봐야 한다.

#### Mind2Web

Mind2Web은 웹 action prediction에서 후보 요소를 먼저 추리고, 그중 하나를 고르는 구조를 대표한다. 우리 프로젝트의 `candidate_elements 15개` 구조와 매우 비슷하다.

중요한 시사점:

- generation보다 discrimination, 즉 "보기 중 고르기"가 일반화에 유리하다.
- 후보 안의 오답들은 random negative보다 어렵고 유용한 negative다.
- top-k 후보 중 2~3등처럼 정답에 가까운 후보가 좋은 hard negative가 될 수 있다.

#### WebLINX / DMR

WebLINX는 Dense Markup Ranking을 통해 task+history와 element HTML 표현의 semantic similarity를 이용한다.

시사점:

- LCA는 구조적 거리다.
- DMR류 방식은 의미적 거리다.
- 좋은 hard negative는 구조적으로 가까운 오답뿐 아니라 task와 의미적으로도 그럴듯한 오답이어야 한다.

---

## 4. PDF의 비판적 결론: LCA Hard Negative의 장단점

### 4.1 LCA 기반 negative가 좋은 이유

웹 개발자는 관련 있는 요소를 같은 컨테이너 안에 넣는 경우가 많다.

예:

- 같은 form 안의 input들
- 같은 nav 안의 link들
- 같은 table row 안의 cell들
- 같은 product card 안의 button, price, title

따라서 DOM에서 가까운 요소는 역할이나 문맥도 비슷할 가능성이 있다. 이런 오답은 모델이 실제로 헷갈릴 수 있으므로 DPO의 rejected로 적합하다.

DPO 관점에서도 좋은 hard negative는 "모델이 그럴듯하다고 생각할 오답"이다. 너무 쉬운 오답은 학습 신호가 약하다. 예를 들어 `Manager` 입력창의 negative로 `Cancel` 버튼을 넣는 것보다, `Supplier` 입력창을 넣는 것이 더 강한 학습 신호를 준다.

### 4.2 LCA 기반 negative가 위험한 이유

#### 문제 1: false negative 위험

DOM에서 가까운 요소는 다른 task에서는 정답이 될 수 있다. 예를 들어 같은 form 안에 `first name`, `last name`, `email` 입력창이 있을 때, task가 바뀌면 각각이 정답이 된다.

이런 요소를 너무 강하게 "나쁜 답"으로 학습시키면 모델이 비슷한 폼 필드를 과도하게 피할 수 있다.

#### 문제 2: DOM 거리와 화면 거리는 다르다

Dual-View 계열 연구는 DOM tree neighbor보다 화면에서 가까운 visual/spatial neighbor가 더 좋은 경우가 많다고 말한다.

웹페이지는 DOM 구조와 실제 화면 배치가 다를 수 있다. CSS, flex, grid, absolute positioning 때문에 DOM에서는 멀어도 화면에서는 가까울 수 있고, DOM에서는 가까워도 화면에서는 멀 수 있다.

#### 문제 3: LCA depth는 너무 거친 지표다

같은 부모 아래 있어도 전혀 다른 요소일 수 있다.

예:

- `Submit` 버튼과 `Terms of Service` 링크
- footer 안의 `Privacy`, `Contact`, `Subscribe`

반대로 DOM에서는 멀지만 의미적으로 거의 같은 요소도 있다.

예:

- 여러 product card의 `Add to Cart` 버튼
- 여러 row의 `Edit` 버튼
- 여러 modal에 반복되는 `Close` 버튼

LCA는 text, aria-label, placeholder, name, option, visual appearance를 직접 반영하지 않는다.

---

## 5. 현재 우리 구현 평가

### 5.1 현재 구현의 좋은 점

현재 `src/train.py`는 DPO 모드에서 다음 구조를 만든다.

- chosen: 정답 candidate의 `choice`
- rejected: `find_lca_hard_negative()`로 고른 오답 candidate의 `choice`
- 기본적으로 rejected의 `op/value`는 chosen과 동일
- 일부 TYPE/SELECT에서 rejected op를 CLICK으로 바꾸는 f_op 휴리스틱 적용

이 설계의 가장 좋은 점은 DPO 학습 신호가 `choice`에 집중된다는 것이다. 만약 rejected의 value까지 틀리게 만들면 모델은 "요소를 잘못 골랐기 때문"인지 "값이 틀렸기 때문"인지 애매하게 배운다. 지금처럼 op/value를 최대한 유지하면 "어떤 candidate를 골라야 하는가"를 더 명확히 배운다.

또한 LCA negative는 추가 모델 없이 BeautifulSoup과 DOM 구조만으로 만들 수 있다. 토큰 비용도 늘리지 않는다. 현재 프로젝트처럼 빠르게 성능을 올려야 하는 상황에서는 구현비용 대비 효율이 좋다.

### 5.2 현재 구현의 약점

가장 큰 약점은 LCA-only라는 점이다. 현재 rejected는 "DOM에서 가까운 오답"이지 "모델이 실제로 가장 헷갈리는 오답"은 아니다.

또한 false negative filter가 없다. gold와 거의 같은 label/text/name을 가진 candidate가 rejected로 들어갈 수 있다. 이런 경우 DPO가 잘못된 선호를 학습할 위험이 있다.

그리고 candidate ordering bias도 조심해야 한다. 후보 순서를 늘 같은 방식으로 두면 모델이 의미가 아니라 위치 패턴을 배울 수 있다. 현재 shuffle augmentation이 어느 정도 완화하지만, rejected가 특정 위치에 자주 놓이는지 점검할 필요가 있다.

### 5.3 객관적 점수

- 아이디어 적합도: 8/10
- 구현비용 대비 효율: 8/10
- 안전장치: 5/10
- SOTA 잠재력: 6.5/10

평가 이유:

- WEPO/DPO 방향은 문제 구조와 잘 맞는다.
- LCA는 cheap baseline으로 좋다.
- 그러나 LCA-only는 논문 리뷰 기준으로도 불완전하다.
- false negative filter와 semantic tie-break 없이는 성능이 오히려 흔들릴 수 있다.

---

## 6. 우리 프로젝트에 적용할 추천 전략

### 6.1 1순위: LCA-DPO는 유지하되 false-negative filter 추가

가장 먼저 할 일은 rejected 후보를 고른 뒤, 이 후보가 gold와 너무 비슷하면 제외하는 것이다.

필터 기준 예시:

- tag가 같고 label/text/name/placeholder가 거의 같으면 제외
- aria-label이 같으면 제외
- select options가 거의 같고 label도 비슷하면 제외
- candidate text가 gold text와 완전히 같으면 제외

의도:

- 진짜 hard negative는 유지한다.
- 애매하거나 정답일 수도 있는 false negative는 제거한다.
- 성능 하락 위험을 줄인다.

토큰 영향:

- 프롬프트를 늘리지 않는다.
- 데이터 생성 로직만 바뀐다.
- 토큰 친화적이다.

### 6.2 2순위: LCA + semantic/task relevance hybrid scoring

LCA만으로 rejected를 고르지 말고, 다음 점수를 섞는 것이 좋다.

```text
negative_score =
  LCA proximity
  + same_tag bonus
  + attribute/text similarity
  + task relevance score
```

실용적인 방식:

1. LCA 거리 가까운 후보들을 먼저 모은다.
2. 그중 `_candidate_match_score(candidate, task)`가 높은 후보를 우선한다.
3. gold와 너무 비슷한 후보는 false-negative filter로 제외한다.
4. 최종 rejected 하나를 고른다.

이 방식은 추가 모델이 필요 없고, 현재 `preprocess.py` 안의 기존 helper를 활용할 수 있다.

### 6.3 3순위: ablation 실험

다음 실험을 반드시 나눠서 봐야 한다.

1. SFT only
2. DPO random negative
3. DPO LCA negative
4. DPO LCA + false-negative filter
5. DPO LCA + semantic/task relevance hybrid

봐야 할 metric:

- Exact Match
- target_id_acc
- op_acc
- value_acc
- workflow EM
- real_web EM
- JSON parse failure rate

특히 real_web target_id_acc가 핵심이다. workflow는 이미 쉬운 영역이므로 전체 점수만 보면 개선 효과를 잘못 판단할 수 있다.

### 6.4 4순위: model-confidence negative

가장 강한 hard negative는 현재 모델이 실제로 틀리게 고르는 후보이다.

방법:

1. SFT 또는 1차 DPO 모델로 train/validation sample을 추론한다.
2. gold가 아닌 후보 중 모델 confidence가 가장 높은 candidate를 찾는다.
3. 그 candidate를 다음 DPO round의 rejected로 사용한다.

장점:

- 모델이 실제로 헷갈리는 오답을 직접 학습한다.
- ANCE/ADORE 계열 연구와 방향이 맞다.

단점:

- inference pass가 추가되어 비용이 크다.
- 구현 복잡도가 높다.
- 대회 일정상 바로 넣기에는 부담이 있다.

따라서 지금은 1~2순위 개선 후, 시간이 남을 때 적용하는 것이 좋다.

---

## 7. 현재 코드 기준 구체 적용 포인트

### 7.1 `find_lca_hard_negative()` 개선

현재 위치:

- `src/preprocess.py`
- `find_lca_hard_negative(row, candidates, target_id)`

추천 변경:

- LCA 거리 계산 후 바로 best를 고르지 말고 후보별 score를 만든다.
- false-negative filter를 먼저 적용한다.
- 같은 거리라면 task relevance가 높은 후보를 고른다.
- gold와 지나치게 유사한 candidate는 제외한다.

예시 정책:

```text
candidate rejected 가능 조건:
- target_id와 다를 것
- DOM에서 element를 찾을 수 있을 것
- gold와 text/label/name이 완전히 동일하지 않을 것
- gold와 attribute overlap이 너무 높지 않을 것

정렬 기준:
1. DOM distance 낮을수록 우선
2. task match score 높을수록 우선
3. 같은 tag면 우선
```

### 7.2 rejected op/value 정책

현재처럼 기본적으로 chosen과 동일하게 유지하는 것이 좋다.

추천:

- 기본 rejected: `same op`, `same value`, `different choice`
- f_op 변조는 너무 많이 쓰지 않기
- TYPE/SELECT에서 CLICK 변조는 보조 실험으로만 유지

이유:

- 현재 병목은 target selection이다.
- value accuracy는 이미 높은 편이다.
- DPO 신호를 element choice에 집중시키는 편이 합리적이다.

### 7.3 token-friendly 원칙

성능이 내려가지 않는 선에서 토큰 친화적으로 하려면 다음 원칙이 좋다.

- prompt를 길게 늘리지 않는다.
- rejected 선택 로직은 데이터 생성 단계에서만 개선한다.
- candidate representation은 `compact_prompt` 실험으로만 줄인다.
- `<think>` reasoning은 너무 길게 만들지 않는다.
- HTML 전체를 넣기보다 candidate 주변 context를 유지한다.

즉, 토큰을 줄이려면 "정보를 없애기"보다 "중요한 candidate 정보만 보존하기"가 중요하다.

---

## 8. 추천 실험 계획

### Experiment A: 현재 LCA-DPO baseline

목적:

- 현재 WEPO-DPO가 SFT 대비 실제로 도움이 되는지 확인

비교:

- SFT only
- DPO LCA

성공 기준:

- real_web target_id_acc 개선
- JSON parse failure 증가 없음
- op/value accuracy 큰 하락 없음

### Experiment B: LCA + false-negative filter

목적:

- LCA negative의 위험을 줄였을 때 성능이 안정되는지 확인

비교:

- DPO LCA
- DPO LCA + filter

성공 기준:

- target_id_acc 유지 또는 상승
- real_web EM 상승
- 특정 tag/input field에서 오답 감소

### Experiment C: LCA + semantic tie-break

목적:

- 단순 DOM 거리보다 task relevance를 섞는 것이 좋은지 확인

비교:

- LCA distance only
- LCA distance + `_candidate_match_score`

성공 기준:

- real_web target_id_acc 상승
- workflow 성능 유지

### Experiment D: compact prompt

목적:

- 토큰을 줄여도 성능이 유지되는지 확인

비교:

- 기본 prompt
- `--e3` compact prompt

성공 기준:

- max_seq_length 2048 안에서 truncation 위험 감소
- 성능 하락이 없거나 매우 작음

---

## 9. PDF 자체에 대한 객관적 평가

이 PDF는 완성된 정식 논문이라기보다는 우리 프로젝트에 맞춘 literature review 문서에 가깝다.

좋은 점:

- WEPO, DPO, Mind2Web, WebLINX, Dual-View, ANCE 등 관련 아이디어를 프로젝트 관점으로 잘 연결한다.
- LCA 기반 hard negative의 장단점을 균형 있게 다룬다.
- 단순히 "LCA가 좋다"가 아니라 false negative와 visual/semantic 대안을 함께 제시한다.

주의할 점:

- `sn-bibliography.bib`가 실제 본문 인용과 연결된 참고문헌이 아니라 템플릿 예시로 보인다.
- 본문에 나온 성능 수치와 논문 요약은 원 논문으로 재검증해야 한다.
- 일부 저자명/문자 인코딩이 깨진 부분이 있다.
- 따라서 이 문서는 "구현 전략 수립용 내부 리뷰"로 쓰는 것이 적절하고, 최종 보고서나 발표에는 원 논문을 직접 인용해야 한다.

---

## 10. 최종 판단

현재 우리 로직은 방향이 맞다. 특히 `chosen = 정답`, `rejected = LCA hard negative`, `op/value 유지`라는 구조는 웹 UI action prediction의 병목인 target selection을 직접 겨냥한다.

하지만 객관적으로 보면 LCA-only는 아직 SOTA라고 하기 어렵다. SOTA에 가까워지려면 최소한 다음 두 가지가 필요하다.

1. false-negative filter
2. semantic/task relevance tie-break

이 두 가지는 추가 토큰을 거의 쓰지 않고 데이터 생성 로직만 개선하므로, 현재 프로젝트 제약과도 잘 맞는다.

추천 우선순위:

1. 현재 LCA-DPO 유지
2. false-negative filter 추가
3. LCA + task relevance hybrid scoring 추가
4. ablation으로 real_web target_id_acc 확인
5. 시간이 남으면 model-confidence negative 검토

