# 디지털경진대회 마스터 플랜 — Web Agent Action Prediction

## 0. 대회 개요

| 항목 | 내용 |
|------|------|
| **Task** | 웹 UI 에이전트의 다음 행동 예측 |
| **Input** | task + history + cleaned_html + candidate_elements(15개) |
| **Output** | op (CLICK/TYPE/SELECT) + target_id + value |
| **Metric** | Exact Match (3개 필드 모두 일치해야 점수) |
| **Train** | 10,307행 |
| **Test 제출** | 4,417행 |
| **고유 사이트** | 97개 (site_token) |
| **제출 제한** | 100회 |

op 분포: CLICK 55.6% / TYPE 25.8% / SELECT 18.6%

---

## 1. 데이터 구조

### 1.1 컬럼 설명

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | str | `aac_mix_train_XXXXXX` |
| site_token | str | 사이트 식별자 (97개) → 같은 토큰 = 고정된 워크플로우 |
| task | str | 전체 작업 목표 (자연어, 모든 입력값 포함) |
| history | str | 이미 완료된 스텝들 |
| cleaned_html | str | 현재 페이지 HTML |
| candidate_elements | str(JSON) | 15개 상호작용 요소 배열 |
| op | str | **예측**: CLICK/TYPE/SELECT |
| target_id | str | **예측**: 요소 ID (elem_XXXXXXXX) |
| value | str | **예측**: 입력값 (CLICK이면 "") |

### 1.2 History 포맷

```
Step 1: [button] New menu update -> CLICK
Step 2: [input] Menu item -> TYPE: citrus salad
Step 3: [select] Station -> SELECT: Grab-and-go
```

패턴: `Step N: [tag] label_text -> OP[: value]`

### 1.3 Candidate Elements 포맷

```json
{
  "candidate_id": "elem_6ad3dc45",
  "tag": "input",
  "text": "Recipe card",
  "attrs": "label=Recipe card | placeholder=Recipe card | name=recipe_card | type=text"
}
```

**중요**: SELECT는 attrs에 `options=A / B / C` 형태로 선택지 포함.

---

## 2. 핵심 데이터 인사이트

### 2.1 History 분석

History는 **워크플로우 상태 머신**:
- **현재 스텝**: `len(parse_history(history)) + 1`
- **패턴**: 같은 site_token → 고정된 액션 시퀀스

**예**: site_068d6fb3
```
Step 1: CLICK "New menu update"
Step 2: TYPE menu item name (from task)
Step 3: SELECT station (from task)
Step 4: SELECT date (from task)
Step 5: CLICK "Publish"
```

### 2.2 Value 추출 가능성: 95.85%

- **TYPE/SELECT value의 95.85%가 task 문장에 그대로 포함됨**
- **전략**: task 텍스트에서 직접 추출 (regex + fuzzy matching)
- **4.15% 엣지케이스**: LLM이 처리

### 2.3 요소 텍스트 매칭: 21.21%

- **정답 요소의 텍스트가 task에 있는 비율: 21.21%**
- "Submit" vs "Publish" 같은 의미론적 차이 존재
- **해결책**: Stage 1 (Site-Template)이 이 문제를 완전히 우회 (step 번호로 직접 매핑)
- Stage 2 (fuzzy) + Stage 3 (LLM)가 의미론적 유사도 처리

### 2.4 Candidates 고정: 15개

- **모든 샘플의 후보 요소가 정확히 15개**
- **15-class classification 문제**로 접근 가능
- tag → op 상관관계 거의 결정적: button→CLICK, input→TYPE, select→SELECT

---

## 3. History 파싱 코드

```python
import re

def parse_history(history: str) -> list:
    steps = []
    if not history or not str(history).strip():
        return steps
    for line in str(history).strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'Step (\d+): \[(\w+)\] (.+?) -> (\w+)(?:: (.*))?', line)
        if m:
            steps.append({
                'step': int(m.group(1)),
                'tag': m.group(2),
                'label': m.group(3).strip(),
                'op': m.group(4).strip(),
                'value': (m.group(5) or '').strip()
            })
    return steps

def get_current_step_num(history: str) -> int:
    steps = parse_history(history)
    return (steps[-1]['step'] + 1) if steps else 1

def clean_history_context(history: str) -> str:
    """스텝 번호 재정렬만 수행. HOVER는 UI 패턴 신호로 유지."""
    steps = parse_history(history)
    lines = []
    for i, s in enumerate(steps, 1):
        val = f": {s['value']}" if s['value'] else ''
        lines.append(f"Step {i}: [{s['tag']}] {s['label']} -> {s['op']}{val}")
    return '\n'.join(lines)
```

---

## 4. Site-Template 학습

```python
from collections import Counter

def get_target_info(row):
    try:
        cands = json.loads(row['candidate_elements'])
        for c in cands:
            if c['candidate_id'] == row['target_id']:
                return c['tag'], c['text']
    except:
        pass
    return None, None

def build_site_templates(train_df):
    """각 site_token의 스텝별 가장 빈번한 (op, label, tag) 추출"""
    df = train_df.copy()
    df[['target_tag', 'target_label']] = df.apply(
        lambda r: pd.Series(get_target_info(r)), axis=1
    )
    df['step_num'] = df['history'].apply(get_current_step_num)
    
    templates = {}
    for site, grp in df.groupby('site_token'):
        template = {}
        for step, sgrp in grp.groupby('step_num'):
            op = Counter(sgrp['op']).most_common(1)[0][0]
            label = Counter(sgrp['target_label']).most_common(1)[0][0]
            tag = Counter(sgrp['target_tag']).most_common(1)[0][0]
            template[step] = {'op': op, 'label': label, 'tag': tag}
        templates[site] = template
    
    return templates
```

---

## 5. 3단계 파이프라인

```
[Input Row]
     ↓
[Stage 1: Site-Template] (step 기반 매핑, 21.21% 문제 우회)
  confidence ≥ 0.9? → return
  else → Stage 2
     ↓
[Stage 2: Rule-Based] (fuzzy + tag-based op, value extraction)
  confidence ≥ 0.7? → return
  else → Stage 3
     ↓
[Stage 3: LLM] (Qwen2.5-7B, CoT, Repair Loop)
  → json output with validation
  → fallback to Stage 2
```

### Stage 1: Site-Template

```python
def predict_by_template(row, templates, candidates):
    site = row['site_token']
    if site not in templates:
        return None, 0.0
    
    template = templates[site]
    step_num = get_current_step_num(str(row['history']))
    
    if step_num not in template:
        return None, 0.0
    
    expected = template[step_num]
    target_id = match_candidate_by_label(expected['label'], candidates)
    if not target_id:
        return None, 0.0
    
    op = expected['op']
    if op == 'CLICK':
        value = ''
    elif op == 'TYPE':
        value = extract_type_value(str(row['task']), expected['label'])
    elif op == 'SELECT':
        target_cand = next((c for c in candidates if c['candidate_id'] == target_id), None)
        value = match_select_option(str(row['task']), target_cand['attrs']) if target_cand else ''
    else:
        value = ''
    
    confidence = 0.95 if (value or op == 'CLICK') else 0.5
    return {'op': op, 'target_id': target_id, 'value': value}, confidence
```

### Stage 2: Rule-Based

```python
from difflib import get_close_matches

def match_candidate_by_label(query: str, candidates: list) -> str:
    query_n = query.lower().strip()
    
    # 1. Exact match
    for c in candidates:
        if c['text'].lower().strip() == query_n:
            return c['candidate_id']
    
    # 2. Match attrs label
    for c in candidates:
        attrs = c.get('attrs', '')
        if 'label=' in attrs:
            label_val = attrs.split('label=')[1].split(' | ')[0].lower()
            if label_val == query_n:
                return c['candidate_id']
    
    # 3. Fuzzy match
    texts = [c['text'] for c in candidates]
    matches = get_close_matches(query, texts, n=1, cutoff=0.5)
    if matches:
        for c in candidates:
            if c['text'] == matches[0]:
                return c['candidate_id']
    
    return None

def parse_select_options(attrs: str) -> list:
    if 'options=' not in attrs:
        return []
    opts_str = attrs.split('options=')[1].split(' | ')[0]
    return [o.strip() for o in opts_str.split(' / ') if o.strip()]

def match_select_option(task: str, attrs: str) -> str:
    options = parse_select_options(attrs)
    if not options:
        return ''
    task_lower = task.lower()
    
    for opt in options:
        if opt.lower() in task_lower:
            return opt
    
    matches = get_close_matches(task, options, n=1, cutoff=0.4)
    return matches[0] if matches else options[0]

def extract_type_value(task: str, field_label: str) -> str:
    # 1. 날짜
    dates = re.findall(r'\d{4}-\d{2}-\d{2}', task)
    if dates and any(kw in field_label.lower() for kw in ['date', 'time', 'when']):
        return dates[0]
    
    # 2. field_label 뒤 값
    label_lower = field_label.lower()
    task_lower = task.lower()
    idx = task_lower.find(label_lower)
    if idx != -1:
        after = task[idx + len(field_label):].strip()
        value = re.split(r'[,.]', after)[0].strip()
        if value:
            return value
    
    # 3. 따옴표 값
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', task)
    if quoted:
        return quoted[0][0] or quoted[0][1]
    
    return ''
```

### Stage 3: LLM Prompt

```
System: You are a precise web automation agent. Predict the EXACT NEXT action.

### Full Task Goal
{task}

### Already Completed Steps
{history_formatted}
(현재 스텝: {current_step_num})

### Available Elements (15 candidates)
{candidates_formatted}

각 SELECT에 options: A / B / C 형태로 선택지 포함
각 INPUT에 placeholder/label 정보 포함

### Output (JSON only)
{"op": "CLICK|TYPE|SELECT", "target_id": "elem_xxx", "value": "..."}
```

**Repair Loop** (max 3회):
1. 생성된 target_id가 유효한가?
2. SELECT면 value가 options에 있는가?
3. 실패 시: 모든 valid ID 힌트 제공 후 재시도
4. 최종 실패 시: Stage 2 결과로 fallback

---

## 6. SFT 데이터 구성

```python
def build_training_example(row):
    """train.csv 행 → SFT 학습 예시"""
    cands = json.loads(row['candidate_elements'])
    history = str(row['history'])
    
    # 입력: task + history_clean + candidates_formatted
    prompt = f"""### Task: {row['task']}
### History: {clean_history_context(history)}
### Candidates:
{format_candidates(cands)}"""
    
    # 출력: Chain-of-Thought + JSON
    step_num = get_current_step_num(history)
    output = {
        "reasoning": f"Step {step_num}: {cands[...]['text']} 선택 필요",
        "op": row['op'],
        "target_id": row['target_id'],
        "value": str(row['value']) if pd.notna(row['value']) else ''
    }
    
    return {
        "instruction": "웹 UI 에이전트의 다음 액션을 정확히 예측하세요.",
        "input": prompt,
        "output": json.dumps(output, ensure_ascii=False)
    }
```

---

## 7. LoRA Fine-Tuning 설정

```python
# Qwen2.5-7B + Unsloth 4-bit
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"  # 7개 모듈
    ],
    lora_alpha=32,
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
)

# SFT 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        save_steps=200,
        logging_steps=50,
    ),
)
```

---

## 8. 데이터 분석 스크립트

```python
import pandas as pd
import json

train = pd.read_csv('data/train.csv')

# 1. 기본 통계
print("op 분포:", train['op'].value_counts())
print("고유 site:", train['site_token'].nunique())

# 2. Value-Task 관계
def value_in_task(row):
    if row['op'] == 'CLICK':
        return True
    return str(row['value']).lower() in str(row['task']).lower()

train['in_task'] = train.apply(value_in_task, axis=1)
print(f"Value in task: {train[train['op']!='CLICK']['in_task'].mean():.2%}")

# 3. Site-template 일관성
from collections import Counter

step_nums = train['history'].apply(get_current_step_num)
site_steps = train.groupby(['site_token', step_nums])['target_label'].apply(
    lambda x: Counter(x).most_common(1)[0][1] / len(x)
)
print(f"Template consistency: {site_steps.mean():.2%}")

# 4. Tag-Op 상관도
for tag in ['button', 'input', 'select']:
    filtered = train[train['target_tag'] == tag]
    print(f"{tag}: {filtered['op'].mode()[0]} ({filtered['op'].value_counts()[filtered['op'].mode()[0]] / len(filtered):.1%})")
```

---

## 9. 구현 로드맵

### Phase 1: 베이스라인 (★★★★★)
- [ ] History 파싱 + clean 함수 검증
- [ ] Site-template 추출
- [ ] Stage 1 + Stage 2 구현
- [ ] Train 검증 세트 (80/20) 평가 → 목표 >65%
- [ ] 첫 제출 (rule-based)

### Phase 2: LLM (★★★★)
- [ ] SFT 데이터 생성 (clean_history 적용)
- [ ] Qwen2.5-7B Unsloth SFT (1000 steps)
- [ ] Repair Loop + validation 구현
- [ ] 검증 후 제출

### Phase 3: 최적화 (★★★)
- [ ] Rule + LLM 앙상블 (confidence 기반)
- [ ] 에러 분석 → 집중 개선
- [ ] 최종 제출 (20-30회)

---

## 10. 성능 예상

| 단계 | Exact Match | 비고 |
|------|------------|------|
| Stage 1 (seen site) | ~80% | op/label 결정적 |
| Stage 1+2 (seen site) | ~68-72% | value 추출 한계 |
| Stage 1+2 (all) | ~50-55% | unseen site 약함 |
| LLM SFT | ~75-80% | 의미론적 이해 |
| **Stage 1+2+3 앙상블** | **~83-88%** | 각 강점 조합 |

---

## 11. 버그 수정 (v3 노트북)

| 위치 | 버그 | 수정 |
|------|------|------|
| Cell 5 | `row['expected_output']` 없음 | 위의 `build_training_example` 사용 |
| Cell 9 | somenna_submission 읽기 | sample_submission.csv로 변경 |
| Cell 9 | Repair Loop 5개 ID만 | 모든 valid_ids 제공 |
| prompt 생성 | SELECT options 없음 | `parse_select_options` 추가 |
| LoRA | 2개 모듈만 | 7개 모듈로 확장 |

---

제출 체크리스트:
```
[ ] 모든 id 포함 (sample_submission 행 수 확인)
[ ] op: CLICK/TYPE/SELECT만
[ ] CLICK의 value = ""
[ ] target_id: 해당 row의 candidate 중 하나
[ ] 결측값 없음
```

*작성일: 2026-04-29*