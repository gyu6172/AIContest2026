import pandas as pd
import json

def analyze_complexity(train_path, test_path):
    def get_candidate_count(x):
        try:
            # Assumes format is a JSON string of a list
            return len(json.loads(x))
        except:
            return 0

    def get_history_steps(x):
        if pd.isna(x) or x == "": return 0
        # Assumes steps are separated by newlines or 'Step X:'
        return x.count('\n') + 1

    cols = ['id', 'task', 'history', 'candidate_elements']
    
    print("Loading data for complexity analysis...")
    train_df = pd.read_csv(train_path, usecols=cols, encoding_errors='replace')
    test_df = pd.read_csv(test_path, usecols=cols, encoding_errors='replace')
    
    for name, df in [("Train", train_df), ("Test", test_df)]:
        print(f"\n--- {name} Statistics ---")
        
        # Candidate count
        df['cand_count'] = df['candidate_elements'].apply(get_candidate_count)
        print(f"Candidate Elements Count:")
        print(df['cand_count'].describe(percentiles=[.5, .75, .9, .95, .99]))
        
        # Task length
        df['task_len'] = df['task'].str.len()
        print(f"\nTask Length (chars):")
        print(df['task_len'].describe(percentiles=[.5, .75, .9, .95, .99]))
        
        # History steps
        df['history_steps'] = df['history'].apply(get_history_steps)
        print(f"\nHistory Steps Count:")
        print(df['history_steps'].describe(percentiles=[.5, .75, .9, .95, .99]))

if __name__ == "__main__":
    analyze_complexity('data/train.csv', 'data/test.csv')
