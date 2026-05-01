import pandas as pd
import hashlib

def get_hash(text):
    if pd.isna(text): return "None"
    return hashlib.md5(str(text).encode('utf-8')).hexdigest()

def analyze_site_content(train_path, test_path):
    cols = ['site_token', 'task', 'history', 'cleaned_html', 'candidate_elements']
    
    print("Loading datasets (this may take a while)...")
    train_df = pd.read_csv(train_path, usecols=cols, encoding_errors='replace')
    test_df = pd.read_csv(test_path, usecols=cols, encoding_errors='replace')
    
    common_sites = set(train_df['site_token']).intersection(set(test_df['site_token']))
    print(f"Common sites to analyze: {len(common_sites)}")
    
    results = []
    
    # Pick a sample of 5 sites for detailed analysis to keep it fast
    sample_sites = list(common_sites)[:5]
    
    for site in sample_sites:
        train_site = train_df[train_df['site_token'] == site]
        test_site = test_df[test_df['site_token'] == site]
        
        # Check uniqueness within site
        train_html_hashes = train_site['cleaned_html'].apply(get_hash).unique()
        test_html_hashes = test_site['cleaned_html'].apply(get_hash).unique()
        
        train_cand_hashes = train_site['candidate_elements'].apply(get_hash).unique()
        test_cand_hashes = test_site['candidate_elements'].apply(get_hash).unique()
        
        # Cross-dataset overlap
        common_html = set(train_html_hashes).intersection(set(test_html_hashes))
        common_cand = set(train_cand_hashes).intersection(set(test_cand_hashes))
        
        results.append({
            'site': site,
            'train_rows': len(train_site),
            'test_rows': len(test_site),
            'train_unique_html': len(train_html_hashes),
            'test_unique_html': len(test_html_hashes),
            'shared_html_hashes': len(common_html),
            'train_unique_cand': len(train_cand_hashes),
            'test_unique_cand': len(test_cand_hashes),
            'shared_cand_hashes': len(common_cand)
        })

    summary_df = pd.DataFrame(results)
    print("\n--- Summary for 5 Sample Sites ---")
    print(summary_df.to_string(index=False))
    
    print("\n--- Key Findings Strategy ---")
    print("1. Are HTMLs identical for the same site? (If unique count is 1, it's static)")
    print("2. Do Test HTMLs exist in Train? (If shared count > 0, we've seen this page before)")
    
    # Check one site for Task/History variation
    if not summary_df.empty:
        site_example = summary_df.iloc[0]['site']
        print(f"\nExample Tasks for {site_example} (Train):")
        print(train_df[train_df['site_token'] == site_example]['task'].head(3).values)
        print(f"\nExample Tasks for {site_example} (Test):")
        print(test_df[test_df['site_token'] == site_example]['task'].head(3).values)

if __name__ == "__main__":
    analyze_site_content('data/train.csv', 'data/test.csv')
