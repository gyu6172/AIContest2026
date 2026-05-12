# Pipeline 로직 리뷰

## 전체 흐름

```
train.csv → build_site_templates() → predict_by_template() → LLM → submission.csv
```

---

## 1. 템플릿 구축 (`build_site_templates`)

**입력**: train.csv 전체 (단, train.py에서는 30%만 사용)

**핵심 로직**:
- key = `(site_token, step_num)` — 같은 사이트의 같은 스텝번호
- step_num = history에서 `Step \d+:` 패턴 개수 + 1
- 해당 key에서 `(op, label, tag)` 조합의 최빈값을 템플릿으로 저장
- label은 candidate의 `text` → 없으면 attrs에서 `aria-label` or `placeholder` 순서로 추출

**주의**: 템플릿에 value는 저장하지 않음. value는 추론 시 task에서 실시간 추출.

---

## 2. 3단계 라우팅 (`inference.py`)

### Stage 1: 템플릿 매칭 (`predict_by_template`)
- key가 템플릿에 있으면 candidates를 순회하며 `label + tag` 동시 일치 확인
- 일치 시 confidence=1.0 반환
- **conf >= 0.7** 이면 결과 확정 → LLM 건너뜀

### Stage 2: LLM 큐 (`llm_queue`)
- 템플릿 미스 + model 있으면 LLM 큐에 추가
- model 없으면 바로 Stage 3 (fallback)

### Stage 3: Rule-based Fallback (`fallback_rule_based`)
- task 키워드로 op 결정: `\btype\b|\benter\b|\binput\b` → TYPE, `\bselect\b|\bchoose\b` → SELECT, else CLICK
- word overlap 점수로 best_candidate 선택
- value는 `extract_value_from_task`로 추출

---

## 3. Value 추출 (`extract_value_from_task`)

우선순위 순서:
1. **SELECT**: attrs의 `options=A / B / C`에서 task와 일치하는 옵션
2. **따옴표**: task에서 `"..."` or `'...'` 패턴
3. **TYPE**: attrs의 label/placeholder를 task에서 찾아 뒤따라오는 텍스트 추출
4. **날짜**: `\d{4}-\d{2}-\d{2}` 패턴
5. 실패 시 `""` 반환

---

## 4. LLM 배치 처리

- BATCH_SIZE=4 (T4 OOM 방지)
- tokenizer: `padding=True, truncation=True, max_length=2048`
- 응답 디코딩: `outputs[:, inputs['input_ids'].shape[1]:]` — 생성 토큰만 슬라이싱
- hallucination 방지: `target_id not in valid_ids` → fallback_rule_based로 대체
- op 검증: CLICK/TYPE/SELECT 외 값 → CLICK으로 강제

---

## 5. 최종 결과 저장

- `results_dict`에 없는 id → CLICK + 첫 번째 candidate_id로 safe default
- `sample_submission.csv`로 merge하여 id 순서 보장
- `fillna("")`로 결측값 처리 후 저장

---

## 6. 학습 데이터 분할 (`train.py`)

```
train.csv (전체)
  ├── 30% → build_site_templates() (템플릿용)
  └── 70% → SFT 학습 데이터
              └── 템플릿으로 맞출 수 있는 건 학습 제외
                  (conf >= 0.7 & op/target_id 일치 시 skip)
```

**목적**: 쉬운 케이스는 규칙으로, 어려운 케이스만 LLM에게 학습

---

## 7. Colab 실행 시 파일 배치

```
/content/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv   ← somenna_submission.csv와 동일 내용
├── src/
│   ├── train.py
│   ├── inference.py
│   └── preprocess.py
├── lora_model/                 ← train 후 생성
├── outputs/                    ← 체크포인트
└── submission.csv              ← inference 후 생성
```

**zip 압축 시**: `HyunJin/` 안에서 `zip -r my_code.zip src/` 실행 (src/가 루트에 위치해야 함)
