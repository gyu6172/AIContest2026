# -*- coding: utf-8 -*-
import csv
import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from preprocess import (
    enforce_consistency,
    choice_to_candidate_id,
    build_prompt,
)

"""
DPO-LoRA 모델의 성능을 빠르게 확인하기 위한 경량 추론 스크립트.
토너먼트 없이 모든 행을 단발성(Single-pass)으로 처리합니다.
A100 환경에 최적화되어 있습니다.
"""

MAX_SEQ_LENGTH = 16384
BATCH_SIZE = 64  # A100 80GB: 컨텍스트 확장 고려 조절

def _parse_json_safe(text: str) -> dict | None:
    import re
    candidates_json = list(re.finditer(r'\{[^{}]*\}', text, re.DOTALL))
    for m in reversed(candidates_json):
        try:
            return json.loads(m.group(0))
        except: continue
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="lora_model", help="Path to LoRA adapter")
    parser.add_argument("--input", type=str, default="data/test.csv", help="Input CSV path")
    parser.add_argument("--output", type=str, default="fast_submission.csv", help="Output CSV path")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(base_dir, args.input)
    
    print(f"Loading Model from: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"

    df = pd.read_csv(test_path)
    id_col = "id" if "id" in df.columns else df.columns[0]
    
    final_results = []
    batch = []

    def flush_batch():
        if not batch: return
        texts = [item["text"] for item in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True, do_sample=False)
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for item, resp in zip(batch, responses):
            search_text = resp.split('</think>', 1)[-1] if '</think>' in resp else resp
            data = _parse_json_safe(search_text)
            op = data.get("op", "CLICK").upper() if data else "CLICK"
            choice = data.get("choice", "") if data else ""
            value = data.get("value", "") if data else ""
            target_id = choice_to_candidate_id(choice, item["candidates"])
            
            pred = {"op": op, "target_id": target_id, "value": value}
            pred = enforce_consistency(pred, item["candidates"])
            
            final_results.append({
                "id": item["id"],
                "op": pred["op"],
                "target_id": pred["target_id"],
                "value": pred["value"]
            })
        batch.clear()

    print("Running Fast Inference...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try: candidates = json.loads(row["candidate_elements"])
        except: candidates = []
        
        prompt = build_prompt(row, candidates)
        batch.append({
            "id": row[id_col],
            "text": f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n",
            "candidates": candidates
        })
        if len(batch) >= BATCH_SIZE: flush_batch()
    
    flush_batch()
    
    print("Saving formatted submission...")
    sub_df = pd.DataFrame(final_results)
    sample_sub_path = os.path.join(base_dir, "data", "somenna_submission.csv")
    
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        final_sub = sample_sub[["id"]].merge(sub_df, on="id", how="left")
    else:
        final_sub = sub_df

    final_sub.fillna("", inplace=True)
    # CLICK 인 경우 value 강제 공백
    final_sub.loc[final_sub["op"] == "CLICK", "value"] = ""
    
    final_sub.to_csv(args.output, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Done! Saved to {args.output} ({len(final_sub)} rows)")

if __name__ == "__main__":
    main()
