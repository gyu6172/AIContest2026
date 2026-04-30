import pandas as pd
import json
import sys
import os

# Ensure src is in the path so we can import from preprocess
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import build_site_templates, predict_by_template, format_candidates_with_attrs

print("Loading data...")
train_df = pd.read_csv('data/train.csv')
print(f"Total train rows: {len(train_df)}")

print("\nBuilding templates...")
templates = build_site_templates(train_df)

correct = 0
wrong = 0
template_used = 0
target_not_in_cands = 0
total_prompt_lengths = []

print("\nRunning diagnostics...")
for _, row in train_df.iterrows():
    cands_str = row['candidate_elements']
    if pd.isna(cands_str):
        continue
    try:
        cands = json.loads(cands_str)
    except:
        continue
        
    target_id = str(row['target_id'])
    
    # 1. Check if target is even in candidates
    valid_ids = [str(c.get('candidate_id', '')) for c in cands]
    if target_id not in valid_ids:
        target_not_in_cands += 1
        
    # 2. Check template accuracy
    pred, conf = predict_by_template(row, templates, cands)
    if pred and conf >= 0.7:
        template_used += 1
        if pred['target_id'] == target_id:
            correct += 1
        else:
            wrong += 1
            
    # 3. Check prompt token lengths (rough estimate: 1 char ~= 0.25 tokens)
    cands_text = format_candidates_with_attrs(cands)
    prompt = f"You are a web UI automation expert. Analyze the current situation and predict the next action.\n\n[Task]\n{row['task']}\n\n[History]\n{row['history']}\n\n[Candidate Elements]\n{cands_text}\n\nOutput ONLY valid JSON in the exact format:\n{{\"op\": \"CLICK|TYPE|SELECT\", \"target_id\": \"element_id\", \"value\": \"input_value_if_any\"}}"
    estimated_tokens = len(prompt) * 0.25
    total_prompt_lengths.append(estimated_tokens)

print("=== DIAGNOSTIC RESULTS ===")
print(f"1. Target NOT in Candidates: {target_not_in_cands} / {len(train_df)} ({(target_not_in_cands/len(train_df)*100):.2f}%)")
print(f"   (If target is not in candidates, 100% accuracy is mathematically impossible via classification)")

print("\n2. Template Rule Engine Performance on Train Set:")
print(f"   Templates Activated: {template_used} / {len(train_df)} ({(template_used/len(train_df)*100):.2f}%)")
if template_used > 0:
    print(f"   Template CORRECT: {correct} ({(correct/template_used*100):.2f}% of activated)")
    print(f"   Template WRONG (False Positives!): {wrong} ({(wrong/template_used*100):.2f}% of activated)")
    print(f"   (These {wrong} wrong answers bypass the LLM completely and ruin our Target Accuracy)")

print("\n3. LLM Prompt Length Estimation:")
if total_prompt_lengths:
    avg_len = sum(total_prompt_lengths) / len(total_prompt_lengths)
    max_len = max(total_prompt_lengths)
    print(f"   Average estimated tokens: {avg_len:.0f}")
    print(f"   Maximum estimated tokens: {max_len:.0f}")
    if max_len > 2048:
        print(f"   WARNING: Max token length ({max_len:.0f}) exceeds current max_seq_length of 2048!")
        print(f"   Cases exceeding 2048 tokens: {sum(1 for x in total_prompt_lengths if x > 2048)}")
        
