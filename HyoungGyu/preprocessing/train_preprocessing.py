import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 1. 경로 설정
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAVE_DIR = ROOT / "HyoungGyu"

# 2. 전처리 설정
# 사용자 요청: Action 키워드(click, type, select) 보존을 위해 불용어에서 제외
DOMAIN_STOP = {"task", "step", "enter", "action", "element"}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | DOMAIN_STOP

# 사용자 요청: 숫자, 날짜, 하이픈 등을 보존하기 위한 정규식
# 기존 [A-Za-z][A-Za-z0-9_-]+ 에서 변경하여 숫자로 시작하는 토큰도 포함
TOKEN_RE = re.compile(r"[A-Za-z0-9/._-]+")

def clean_text_advanced(s):
    """
    기본적인 토큰화 및 불용어 제거를 수행하되, 숫자와 Action 키워드를 보존합니다.
    """
    if not isinstance(s, str):
        s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    
    # 소문자 변환 및 토큰 추출 (숫자 보존형 정규식 사용)
    tokens = TOKEN_RE.findall(s.lower())
    
    # 불용어 제거 (한 글자 숫자나 의미 없는 단어 필터링, 단 중요한 숫자는 보존될 수 있음)
    cleaned_tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 0]
    
    return " ".join(cleaned_tokens)

def process_html_advanced(html):
    """
    cleaned_html 고도화 전처리:
    1. <html>, <body> 태그 제거
    2. 닫는 태그 제거
    3. backend_node 관련 모든 속성 제거 (backend_nodeid, backend_node_id 등)
    4. 구조적 키워드 추출
    """
    if not isinstance(html, str):
        return ""
    
    # 1. <html> 및 <body> 오프닝 태그 제거 (속성 포함)
    html = re.sub(r'<(html|body)[^>]*>', '', html, flags=re.IGNORECASE)
    
    # 2. 닫는 태그 제거 (예: </div>, </span>)
    html = re.sub(r'</[^>]+>', '', html)
    
    # 3. backend_node 관련 모든 속성 제거 (예: backend_nodeid="104", backend_node_id="104")
    html = re.sub(r'backend_node[a-z_]*="[0-9]+"', '', html, flags=re.IGNORECASE)
    
    # 4. 불필요한 공백 정리 및 토큰화
    return clean_text_advanced(html)

def process_candidates_structured(candidates_json):
    """
    candidate_elements 구조적 평탄화 (Structured Flattening):
    [CAND] tag:X text:Y attr_name:Z [SEP] 형식
    """
    try:
        cands = json.loads(candidates_json) if isinstance(candidates_json, str) else []
    except Exception:
        cands = []
        
    structured_parts = []
    for c in cands:
        parts = ["[CAND]"]
        
        # 태그 정보
        tag = c.get("tag") or "unknown"
        parts.append(f"tag:{tag}")
        
        # 텍스트 정보
        text = c.get("text")
        if text:
            clean_text = clean_text_advanced(text)
            if clean_text:
                parts.append(f"text:{clean_text}")
        
        # 속성 정보
        attrs = c.get("attrs") or ""
        # 주요 속성 추출 (label, name, placeholder, type, options 등)
        attr_matches = re.findall(r"(\w+)=([^|]+)", attrs)
        for key, val in attr_matches:
            key = key.strip().lower()
            val_clean = clean_text_advanced(val.strip())
            if val_clean:
                parts.append(f"attr_{key}:{val_clean}")
        
        parts.append("[SEP]")
        structured_parts.append(" ".join(parts))
        
    return " ".join(structured_parts)

def main():
    print("--- [Step 0] 데이터 로딩 ---")
    train_path = DATA_DIR / "train.csv"
    df = pd.read_csv(train_path, encoding="utf-8", encoding_errors="replace")
    
    # 분석용: 원본 길이 측정
    orig_html_len = df['cleaned_html'].str.len().mean()
    orig_cand_len = df['candidate_elements'].str.len().mean()
    
    print("--- [Step 1] task/history 전처리 (숫자 및 Action 보존) ---")
    task_processed = [clean_text_advanced(x) for x in df['task']]
    history_processed = [clean_text_advanced(x) for x in df['history']]
    
    print("--- [Step 2] cleaned_html 고도화 전처리 (닫는태그/NodeID 제거) ---")
    html_processed = [process_html_advanced(x) for x in df['cleaned_html']]
    
    print("--- [Step 3] candidate_elements 구조적 평탄화 ---")
    candidate_processed = [process_candidates_structured(x) for x in df['candidate_elements']]
    
    print("--- [Step 4] 데이터 합치기 및 저장 ---")
    preprocessed_df = pd.DataFrame({
        'id': df['id'],
        'site_token': df['site_token'],
        'task': task_processed,
        'history': history_processed,
        'cleaned_html': html_processed,
        'candidate_element': candidate_processed,
        'op': df['op'],
        'target_id': df['target_id'],
        'value': df['value']
    })
    
    output_path = SAVE_DIR / "preprocessed_train_v5.csv"
    preprocessed_df.to_csv(output_path, index=False, encoding="utf-8")
    
    # 최종 결과 분석 보고용 데이터
    new_html_len = preprocessed_df['cleaned_html'].str.len().mean()
    new_cand_len = preprocessed_df['candidate_element'].str.len().mean()
    
    print(f"\n--- [최종 결과 보고] ---")
    print(f"1. 파일 저장 완료: {output_path}")
    print(f"2. HTML 최적화: {orig_html_len:.1f}자 -> {new_html_len:.1f}자 (약 {(1 - new_html_len/orig_html_len)*100:.1f}% 감소)")
    print(f"3. Candidate 최적화: {orig_cand_len:.1f}자 -> {new_cand_len:.1f}자")
    print(f"4. 샘플 확인 (1번 행):")
    print(f"   - Task: {task_processed[0]}")
    print(f"   - History (Action 확인): {history_processed[0]}")
    print(f"   - HTML (태그 확인): {html_processed[0][:100]}...")
    print(f"   - Candidate (구조 확인): {candidate_processed[0][:100]}...")

if __name__ == "__main__":
    main()
