import json
import re
from bs4 import BeautifulSoup

def clean_html_for_src3(html_str):
    """모델이 구조를 파악하기 좋게 최소한의 정제만 수행"""
    if not html_str: return ""
    soup = BeautifulSoup(html_str, 'lxml')
    # 불필요한 태그 제거 (script, style 등은 이미 전처리되었을 가능성이 높음)
    for tag in soup(['script', 'style']):
        tag.decompose()
    
    # 핵심 속성만 남기기
    allowed_attrs = ['id', 'name', 'type', 'value', 'placeholder', 'aria-label', 'text', 'role']
    for tag in soup.find_all(True):
        attrs = {k: v for k, v in tag.attrs.items() if k in allowed_attrs}
        tag.attrs = attrs
    
    return str(soup)

def build_src3_prompt(task, history, candidates):
    """RAG 없이 순수하게 Task, History, HTML만 제공하는 프롬프트"""
    history_str = "\n".join([f"- {h}" for h in history]) if history else "No history."
    
    # 후보 요소 리스트를 간결한 텍스트로 변환
    elements_desc = []
    for c in candidates:
        attrs_str = " ".join([f'{k}="{v}"' for k, v in c.get('attrs', {}).items()])
        elements_desc.append(f"ID {c['backend_node_id']}: <{c['tag_name']} {attrs_str}>{c.get('text', '')}</{c['tag_name']}>")
    elements_str = "\n".join(elements_desc)

    prompt = f"""You are an expert web navigation agent. Based on the task, interaction history, and visible HTML elements, predict the next action.

### TASK
{task}

### HISTORY
{history_str}

### VISIBLE ELEMENTS
{elements_str}

### INSTRUCTION
Predict the single next action in JSON format: {{"op": "CLICK/TYPE/SELECT", "target_id": number, "value": "string or null"}}.
- target_id must be one of the backend_node_id from VISIBLE ELEMENTS.
- value is only for TYPE (text to enter) or SELECT (option text).
- For CLICK, value must be null.

RESPONSE:"""
    return prompt

def generate_src3_dpo_pairs(row):
    """정답(Chosen)과 오답(Rejected) 쌍을 생성"""
    try:
        task = str(row['task'])
        
        # 데이터가 이미 객체인지 확인 (Pandas 파싱 방식에 대응)
        history = row['history']
        if isinstance(history, str):
            history = json.loads(history)
        
        candidates = row['candidate_elements']
        if isinstance(candidates, str):
            candidates = json.loads(candidates)
            
        target_id = int(row['target_id'])
        op = str(row['op'])
        
        # NaN 처리 (Pandas float NaN 대응)
        value = row['value']
        if pd.isna(value) or str(value).lower() == 'nan':
            value = None
        else:
            value = str(value)
        
        prompt = build_src3_prompt(task, history, candidates)
        
        # Chosen: 실제 정답
        chosen = json.dumps({"op": op, "target_id": target_id, "value": value})
        
        # Rejected: 하드 네거티브
        other_ids = [c['backend_node_id'] for c in candidates if int(c['backend_node_id']) != target_id]
        if other_ids:
            rejected_id = other_ids[0] 
            rejected = json.dumps({"op": op, "target_id": rejected_id, "value": value})
        else:
            # 후보가 하나뿐인 경우 액션을 변조
            alt_op = "CLICK" if op != "CLICK" else "TYPE"
            alt_val = None if alt_op == "CLICK" else "wrong_value"
            rejected = json.dumps({"op": alt_op, "target_id": target_id, "value": alt_val})

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
    except Exception as e:
        # 에러 로그 확인용 (디버깅 시 필요)
        # print(f"Error in row: {e}")
        return None

import pandas as pd # pd.isna 사용을 위해 추가
