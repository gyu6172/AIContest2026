import pandas as pd
import json
import sys
import os

# Ensure src is in the path so we can import from preprocess
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import build_empirical_priors, format_empirical_priors, format_candidates_with_attrs
from retrieval import ExampleRetriever, format_similar_examples

print("Loading data...")
train_df = pd.read_csv('data/train.csv')
print(f"Total train rows: {len(train_df)}")

print("\nBuilding empirical priors and retriever...")
priors = build_empirical_priors(train_df)
retriever = ExampleRetriever().build(train_df)

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
        
    # 2. Check task-first prompt token lengths (rough estimate: 1 char ~= 0.25 tokens)
    cands_text = format_candidates_with_attrs(cands)
    examples_text = format_similar_examples(retriever.query(row, k=2, exclude_id=row.get('id')), max_task_chars=200, max_history_chars=200)
    priors_text = format_empirical_priors(row, priors)
    prompt = f"You are a web UI automation expert. Analyze the current situation and predict the next action.\n\n[Task]\n{row['task']}\n\n[History]\n{row['history']}\n\n{examples_text}\n\n{priors_text}\n\n[Candidate Elements]\n{cands_text}\n\nOutput ONLY valid JSON in the exact format:\n{{\"op\": \"CLICK|TYPE|SELECT\", \"target_id\": \"element_id\", \"value\": \"input_value_if_any\"}}"
    estimated_tokens = len(prompt) * 0.25
    total_prompt_lengths.append(estimated_tokens)

print("=== DIAGNOSTIC RESULTS ===")
print(f"1. Target NOT in Candidates: {target_not_in_cands} / {len(train_df)} ({(target_not_in_cands/len(train_df)*100):.2f}%)")
print(f"   (If target is not in candidates, 100% accuracy is mathematically impossible via classification)")

print("\n2. LLM Prompt Length Estimation:")
if total_prompt_lengths:
    avg_len = sum(total_prompt_lengths) / len(total_prompt_lengths)
    max_len = max(total_prompt_lengths)
    print(f"   Average estimated tokens: {avg_len:.0f}")
    print(f"   Maximum estimated tokens: {max_len:.0f}")
    if max_len > 2048:
        print(f"   WARNING: Max token length ({max_len:.0f}) exceeds current max_seq_length of 2048!")
        print(f"   Cases exceeding 2048 tokens: {sum(1 for x in total_prompt_lengths if x > 2048)}")
        
