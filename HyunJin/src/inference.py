import os
import csv
import torch
import pandas as pd
import json
from unsloth import FastLanguageModel
import re
from tqdm import tqdm

from preprocess import predict_by_template, build_site_templates, format_candidates_with_attrs, fallback_rule_based, extract_value_from_task

def extract_json_answers(responses):
    """JSON 형식의 답변에서 정보를 추출"""
    results = []
    for resp in responses:
        try:
            m = re.search(r'\{.*?\}', resp, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                op = data.get("op", "CLICK").upper()
                if op not in ("CLICK", "TYPE", "SELECT"):
                    op = "CLICK"
                target_id = str(data.get("target_id", ""))
                value = str(data.get("value", ""))
                if op == "CLICK": value = ""
                results.append((op, target_id, value))
            else:
                results.append(("CLICK", "", ""))
        except:
            results.append(("CLICK", "", ""))
    return results

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'train.csv')
    test_path = os.path.join(base_dir, 'data', 'test.csv')
    sample_sub_path = os.path.join(base_dir, 'data', 'sample_submission.csv')
    
    print("🔍 1. Preparing Structural Rules & Templates...")
    train_df = pd.read_csv(train_path)
    templates = build_site_templates(train_df)
    
    print("🤖 2. Waking up LLM...")
    lora_dir = os.path.join(base_dir, "lora_model")
    model, tokenizer = None, None
    
    if os.path.exists(lora_dir):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = lora_dir,
            max_seq_length = 2048,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    print("🚀 3. Starting Solid Inference...")
    test_df = pd.read_csv(test_path)
    id_col = 'id' if 'id' in test_df.columns else test_df.columns[0]
    
    results_dict = {}
    llm_queue = [] 
    
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Step 1: Routing"):
        q_id = row[id_col]
        try:
            candidates = json.loads(row['candidate_elements'])
        except:
            candidates = []
            
        pred, conf = predict_by_template(row, templates, candidates)
        
        if pred and conf >= 0.7:
            v = pred['value'] if pred['op'] != 'CLICK' else ''
            results_dict[q_id] = {'id': q_id, 'op': pred['op'], 'target_id': pred['target_id'], 'value': v}
        elif model is not None:
            cands_text = format_candidates_with_attrs(candidates)
            prompt = f"You are a web UI automation expert. Analyze the current situation and predict the next action.\n\n[Task]\n{row['task']}\n\n[History]\n{row['history']}\n\n[Candidate Elements]\n{cands_text}\n\nOutput ONLY valid JSON in the exact format:\n{{\"op\": \"CLICK|TYPE|SELECT\", \"target_id\": \"element_id\", \"value\": \"input_value_if_any\"}}"
            text = f"### Instruction:\n{prompt}\n\n### Response:\n"
            
            llm_queue.append({
                'id': q_id, 
                'text': text, 
                'candidates': candidates,
                'row': row
            })
        else:
            fb = fallback_rule_based(row, candidates)
            v = fb['value'] if fb['op'] != 'CLICK' else ''
            results_dict[q_id] = {'id': q_id, 'op': fb['op'], 'target_id': fb['target_id'], 'value': v}
            
    if llm_queue and model is not None:
        BATCH_SIZE = 4 # Colab T4 GPU 최적화
        print(f"\n🤖 LLM Processing {len(llm_queue)} cases...")
        
        for i in tqdm(range(0, len(llm_queue), BATCH_SIZE), desc="Batches"):
            batch = llm_queue[i:i+BATCH_SIZE]
            texts = [item['text'] for item in batch]
            
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                use_cache=True, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # [Decoding Fix] 생성된 토큰만 슬라이싱하여 추출
            generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            answers = extract_json_answers(responses)
            
            for item, (op, target_id, value) in zip(batch, answers):
                valid_ids = [str(c.get('candidate_id', '')) for c in item['candidates']]
                
                # Hallucination 방지 및 Fallback
                if str(target_id) not in valid_ids:
                    fb = fallback_rule_based(item['row'], item['candidates'])
                    op, target_id, value = fb['op'], fb['target_id'], fb['value']
                else:
                    # 타겟은 찾았으나 LLM의 value를 믿지 않고 정규식으로 덮어씌움! (Value Acc 복구)
                    matched_cand = next((c for c in item['candidates'] if str(c.get('candidate_id', '')) == str(target_id)), None)
                    c_attrs = str(matched_cand.get('attrs', '')) if matched_cand else ''
                    value = extract_value_from_task(item['row']['task'], op, c_attrs)
                
                results_dict[item['id']] = {'id': item['id'], 'op': op, 'target_id': target_id, 'value': value}

    print("\n💾 4. Cleaning values and Saving submission...")
    final_results = []
    
    # 템플릿, LLM, Fallback 상관없이 모든 결과에 대해 일괄 전처리 (데이터 누수 방지)
    for i, row in test_df.iterrows():
        q_id = row[id_col]
        if q_id in results_dict:
            res = results_dict[q_id]
            # 줄바꿈만 공백으로 (쉼표는 CSV quoting으로 보존)
            v = str(res['value']).replace('\n', ' ').replace('\r', ' ').strip()
            # CLICK은 value 무조건 빈값
            if res['op'] == 'CLICK':
                v = ''
            final_results.append({'id': q_id, 'op': res['op'], 'target_id': res['target_id'], 'value': v})
        else:
            # Safe default: CLICK on first candidate
            try:
                cands = json.loads(row['candidate_elements'])
                default_target = str(cands[0].get('candidate_id', '')) if cands else ''
            except:
                default_target = ''
            final_results.append({'id': q_id, 'op': 'CLICK', 'target_id': default_target, 'value': ''})
        
    sub_df = pd.DataFrame(final_results)
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        final_sub = sample_sub[['id']].merge(sub_df, on='id', how='left')
    else:
        final_sub = sub_df
        
    final_sub.fillna("", inplace=True)

    # === Final safety guards (defense in depth) ===
    # 1. CLICK은 value 강제 빈값 (train 정답 100% 검증)
    final_sub.loc[final_sub['op'] == 'CLICK', 'value'] = ''
    # 2. op 화이트리스트 외 값은 CLICK으로 정정
    final_sub.loc[~final_sub['op'].isin(['CLICK', 'TYPE', 'SELECT']), 'op'] = 'CLICK'
    # 3. 빈 target_id 방지 (이론상 없어야 하지만 안전망)
    if (final_sub['target_id'].astype(str).str.strip() == '').any():
        print(f"⚠ Warning: {(final_sub['target_id'].astype(str).str.strip()=='').sum()} rows with empty target_id")

    out_path = os.path.join(base_dir, 'submission.csv')
    # QUOTE_MINIMAL: 쉼표/따옴표가 포함된 value만 인용 (pandas 기본값을 명시)
    final_sub.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"🎉 Overhaul complete! Saved to {out_path}")
    print(f"   Rows: {len(final_sub)}")
    print(f"   op dist: {dict(final_sub['op'].value_counts())}")
    print(f"   CLICK with non-empty value: {((final_sub['op']=='CLICK') & (final_sub['value'].astype(str).str.strip()!='')).sum()}")

if __name__ == "__main__":
    main()
