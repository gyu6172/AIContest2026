# 프로젝트 분석 및 향후 전략 (Project Review & Strategy)
> 최종 업데이트: 2026-05-12

본 문서는 프로젝트 진행 과정에서의 데이터 분석 결과, 성능 리뷰 및 향후 개선 방향을 기록합니다.

---

## 1. 데이터 분석 리포트

### 1.1. 주요 컬럼 특징
*   **site_token**: 사이트별로 성격이 다름. `site_2aa627db`는 CLICK 편향(96.5%)이 심해 학습에서 제외 검토.
*   **task**: 중복된 Task가 많으나 History Step에 따라 정답 Target이 달라짐.
*   **cleaned_html**: 
    - **Workflow**: `current step`, `completed-fields` 힌트가 있어 정답률 100% 가능.
    - **Real Web**: 노이즈가 많고 텍스트/속성이 비어있는 경우가 많아 Target 예측의 핵심 과제.

### 1.2. Value 추출 이슈
*   `task`와 `value` 사이의 표기 차이(띄어쓰기, 대소문자, 불필요한 따옴표 등)로 인해 단순 매칭이 어려움.
*   복잡한 문제(후보 이름 없음 등)는 규칙 기반보다 LLM의 판단이 필수적임.

---

## 2. 성능 리뷰 및 병목 분석

### 2.1. 현재 성능 지표
*   **Target Acc (0.7825)**: 가장 큰 병목 구간. 특히 Real Web 환경에서 유사한 요소들 사이의 변별력이 낮음.
*   **Value Acc (0.9627)**: 높은 편이나, `TYPE` 액션 시 task 내의 정보를 정확히 추출하지 못하는 케이스가 잔존.

### 2.2. 주요 실패 사례 (Error Analysis)
*   **시각적 유사 요소**: 같은 텍스트를 가진 버튼이 여러 개일 때 구조적 맥락 부족으로 오답 선택.
*   **텍스트 부재 요소**: 아이콘 버튼 등 텍스트가 없는 경우 grounding 실패.
*   **Action Desc 오류**: 2단계 Grounding 시 첫 번째 단계에서 동작을 잘못 정의하면 이후 단계가 모두 어긋남.

---

## 3. 향후 개선 전략

### 3.1. WEPO (Web Element Preference Optimization) 도입 (진행 중)
*   **LCA 거리 분석**: DOM 트리상에서 정답과 가장 가까운 요소를 "매력적인 오답(Hard Negative)"으로 선정.
*   **DPO (Direct Preference Optimization)**: 모델이 정답과 구조적으로 인접한 오답을 명확히 구분하도록 선호도 학습 진행.

### 3.2. Retriever 고도화
*   현재 Jaccard 유사도 기반 검색을 Semantic Search(임베딩 기반)로 전환하여 더 관련성 높은 Few-shot 예시 제공.

### 3.3. Candidate Representation 강화
*   `aria-label`, `placeholder` 등 대체 텍스트 폴백 체인을 강화하여 텍스트 없는 요소의 식별력 향상.
*   부모 및 이웃 요소의 정보를 더 컴팩트하게 압축하여 주입.

---

## 4. 실험 기록 (Log)

*   **2026-05-11**: Qwen3 Thinking Mode 활성화. Target Acc 개선 시도.
*   **2026-05-12**: WEPO 프레임워크 기반 DPO 파이프라인 설계 및 문서 구조 개편.
