import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# Config
DATA_DIR = "../results/patent_cross_page_v3_results_swe_7000"
OUTPUT_MANIFEST = "../results/master_candidate_manifest_v1.csv"
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

def build_manifest():
    print("Scanning for Stage A results...")
    stage_a_files = glob.glob(f"{DATA_DIR}/*/overlap/overlap_results.csv")
    
    records = []
    unique_patents = set()

    for filepath in stage_a_files:
        patent_id = Path(filepath).parents[1].name
        unique_patents.add(patent_id)

        try:
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                records.append({
                    'pair_id': f"{patent_id}_p{row['page_a']}_p{row['page_b']}",
                    'patent_id': patent_id,
                    'page_a': row['page_a'],
                    'page_b': row['page_b'],
                    'file_a': row['file_a'],
                    'file_b': row['file_b'],
                    'stage_a_accepted': row['accepted'],
                    'stage_a_overlap_ratio': row['overlap_ratio'],
                    'human_label': 'unlabeled',  # 1=Pos, 0=Neg, u=Uncertain
                    'split': 'unassigned',
                    'selection_reason': 'stage_a_output'
                })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # Enforce Patent-Level Splits (Prevents data leakage)
    patents = list(unique_patents)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(patents)
    
    n_total = len(patents)
    n_train = int(n_total * TRAIN_FRAC)
    n_val = int(n_total * VAL_FRAC)
    
    train_pats = set(patents[:n_train])
    val_pats = set(patents[n_train:n_train + n_val])

    manifest_df = pd.DataFrame(records)
    
    def assign_split(pid):
        if pid in train_pats: return 'train'
        if pid in val_pats: return 'val'
        return 'test'
        
    manifest_df['split'] = manifest_df['patent_id'].apply(assign_split)
    manifest_df.to_csv(OUTPUT_MANIFEST, index=False)
    
    print(f"\nManifest created with {len(manifest_df)} total candidate pairs across {len(patents)} multi-page patents.")
    print(f" - Train pairs : {len(manifest_df[manifest_df['split']=='train'])}")
    print(f" - Val pairs   : {len(manifest_df[manifest_df['split']=='val'])}")
    print(f" - Test pairs  : {len(manifest_df[manifest_df['split']=='test'])}")

if __name__ == "__main__":
    build_manifest()