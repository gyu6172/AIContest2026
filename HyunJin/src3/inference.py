import os
import torch
import pandas as pd
import json
from tqdm import tqdm
from unsloth import FastLanguageModel
from src3.preprocess import build_src3_prompt

# 전역 설정
MODEL_PATH = "src3_lora_model"
MAX_SEQ_LENGTH = 4096

def run_src3_inference():
    print(f"--- Starting src3 Inference with {MODEL_PATH} ---")
    
    # 1. 모델 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    # 2. 테스트 데이터 로드
    test_df = pd.read_csv("data/test.csv")
    results = []

    # 3. 루프 추론
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting Actions"):
        task = row['task']
        history = json.loads(row['history'])
        candidates = json.loads(row['candidate_elements'])
        
        # 프롬프트 생성 (RAG 없이)
        prompt = build_src3_prompt(task, history, candidates)
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 이후의 텍스트만 추출 (JSON 파싱)
        if "RESPONSE:" in response:
            json_str = response.split("RESPONSE:")[-1].strip()
        else:
            json_str = response.strip()

        try:
            # 단순 정규식으로 JSON 추출 시도
            match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if match:
                pred = json.loads(match.group())
            else:
                pred = json.loads(json_str)
        except:
            pred = {"op": "CLICK", "target_id": candidates[0]['backend_node_id'], "value": None}
            
        results.append({
            "id": idx,
            "target_id": pred.get("target_id", 0),
            "op": pred.get("op", "CLICK"),
            "value": pred.get("value", None)
        })

    # 4. 제출 파일 저장
    sub_df = pd.DataFrame(results)
    sub_df.to_csv("submission_src3.csv", index=False)
    print("Inference complete. Saved to submission_src3.csv")

if __name__ == "__main__":
    import re
    run_src3_inference()
