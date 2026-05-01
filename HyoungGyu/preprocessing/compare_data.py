import pandas as pd

def compare_csvs(train_path, test_path):
    print(f"Loading {train_path}...")
    # Load only necessary columns to save memory
    train_df = pd.read_csv(train_path, usecols=['id', 'site_token'], encoding_errors='replace')
    
    print(f"Loading {test_path}...")
    test_df = pd.read_csv(test_path, usecols=['id', 'site_token'], encoding_errors='replace')
    
    print("\n--- Overlap Analysis ---")
    
    # ID Overlap
    train_ids = set(train_df['id'])
    test_ids = set(test_df['id'])
    common_ids = train_ids.intersection(test_ids)
    
    print(f"Train IDs: {len(train_ids)}")
    print(f"Test IDs: {len(test_ids)}")
    print(f"Common IDs: {len(common_ids)}")
    if common_ids:
        print(f"Sample common IDs: {list(common_ids)[:5]}")
    
    # Site Token Overlap
    train_tokens = set(train_df['site_token'])
    test_tokens = set(test_df['site_token'])
    common_tokens = train_tokens.intersection(test_tokens)
    
    print(f"\nTrain Site Tokens: {len(train_tokens)}")
    print(f"Test Site Tokens: {len(test_tokens)}")
    print(f"Common Site Tokens: {len(common_tokens)}")
    if common_tokens:
        print(f"Sample common tokens: {list(common_tokens)[:5]}")

if __name__ == "__main__":
    compare_csvs('data/train.csv', 'data/test.csv')
