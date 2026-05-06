import pandas as pd
import json

sub = pd.read_csv('submission.csv', keep_default_na=False)
test = pd.read_csv('data/test.csv')

merged = sub.merge(test, on='id')

default_target_count = 0
for i, row in merged.iterrows():
    cands_str = row['candidate_elements']
    try:
        cands = json.loads(cands_str) if pd.notna(cands_str) else []
    except:
        cands = []
        
    target_id = str(row['target_id'])
    
    if cands and target_id == str(cands[0].get('candidate_id', '')):
        default_target_count += 1

print(f"Total rows: {len(merged)}")
print(f"Default targets: {default_target_count}")
print(f"Default rate: {default_target_count / len(merged) * 100:.2f}%")
