# Master Plan: Hybrid Web UI Action Prediction (2026-05-01)

## 1. 개요 (Overview)
본 문서는 팀원의 데이터 인사이트와 시스템 분석 결과를 통합하여, 추론 성능(Accuracy)과 데이터 커버리지(Coverage)를 극대화하기 위한 하이브리드 추론 아키텍처 개편 계획을 담고 있다.

---

## 2. 핵심 진단 및 인사이트 (Core Insights)
- **Coverage 문제:** 현재 4,417개 행 중 3,913개만 응답됨 (누락된 504개 행이 점수 하락의 주원인).
- **사이트 이원화:** `site_token`을 통해 '정형화된 워크플로우(Workflow)'와 '비정형 웹(Raw Web)'이 구분됨.
- **태그 불일치:** History의 태그명(`textbox`, `link`)과 Candidate의 태그명(`input`, `a`)이 상이하여 매칭 성능 저하.
- **Value 노이즈:** Task에서 값을 추출할 때 따옴표, 불필요한 설명("about..."), 소수점 절삭 등의 문제로 정확도 하락.

---

## 3. 세부 실행 전략 (Detailed Strategies)

### Phase 1: 이원화 라우팅 (Workflow vs. Raw Web Split)
- **Workflow 웹 (쉬운 문제):** `data-site=` 등 구조적 힌트가 뚜렷한 사이트는 검증된 규칙 기반 엔진(Stage 1)을 최우선 적용.
- **Raw Web (어려운 문제):** 텍스트가 비어있거나 노이즈가 많은 사이트는 LLM(Qwen2.5) + RAG(Similar Examples) 문맥 추론에 의존.

### Phase 2: Coverage 100% 보장 (Safety Net)
- LLM 출력 실패나 규칙 기반 매칭 실패 시에도 반드시 유효한 `target_id`를 반환하도록 `fallback_rule_based` 함수에 최후의 안전장치 추가.
- `candidates[0]` 또는 해당 사이트의 `Empirical Prior`(과거 최빈 태그)를 강제 할당.

### Phase 3: Value 추출 및 정규화 (Value Normalization)
- `extract_value_from_task` 정규식 개편:
    - 홑따옴표/겹따옴표(`'`, `"`) 자동 제거.
    - 숫자 사이의 마침표(`.`) 및 콤마(`,`)를 보존하도록 정규식 수정 (소수점 절삭 방지).
    - "about", "for", "set to" 등의 키워드 뒤에 오는 불필요한 서술어 제거 로직 강화.

### Phase 4: History-Candidate 태그 매핑 브릿지
- 프롬프트 구성 시 History 문자열 내의 커스텀 태그를 표준 HTML 태그로 치환.
    - 예: `[textbox] -> [input]`, `[link] -> [a]`, `[combobox] -> [select]`.

### Phase 5: Semantic Name Tag 스코어링 (Fallback 고도화)
- `_candidate_match_score` 가중치 차등 부여:
    - `text` (화면 표시 텍스트) 일치: **+3.0점**
    - `aria-label / placeholder` 일치: **+2.0점**
    - `name / title` 등 속성 일치: **+1.0점**

---

## 4. 향후 작업 순서 (Action Items)
1. **`src/preprocess.py`:** Value 추출 정규식 및 따옴표 제거 로직 개선.
2. **`src/preprocess.py`:** `fallback_rule_based` 강제 할당 안전장치 추가.
3. **`src/preprocess.py`:** `_candidate_match_score` 가중치 로직 개편.
4. **`src/inference.py`:** 프롬프트 빌드 시 History 태그 정규화 로직 적용.
5. **검증:** 수정 후 `run_analysis` 재실행하여 Coverage 100% 및 Exact Match 향상 확인.
