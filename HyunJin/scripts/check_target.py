import pandas as pd
import json

df = pd.read_csv('data/train.csv')
def check_target_in_candidates(row):
    try:
        candidates = json.loads(row['candidate_elements'])
        candidate_ids = [str(c['candidate_id']) for c in candidates]
        return str(row['target_id']) in candidate_ids
    except:
        return False

results = df.apply(check_target_in_candidates, axis=1)
print(f"Target in candidates ratio: {results.mean():.4f}")
print(f"Total rows: {len(df)}")
print(f"Missing target rows: {len(df) - results.sum()}")
