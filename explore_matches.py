import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')

singletons = gt[gt['matched_entity_ids'].isna()].shape[0]
print(f'Singletons: {singletons} / {gt.shape[0]} = {singletons/gt.shape[0]:.2%}')
print()

# Index for fast lookup
s2_idx = s2.set_index('entity_id')
s3_idx = s3.set_index('entity_id')
s1_idx = s1.set_index('entity_id')

# Show 5 matched examples
matched = gt[gt['matched_entity_ids'].notna()].head(5)
for _, row in matched.iterrows():
    s1_id = row['source1_entity_id']
    s1_rec = s1_idx.loc[s1_id]
    print(f"=== S1: {s1_id} ===")
    print(f"  Name: {s1_rec['business_name']}")
    print(f"  Addr: {s1_rec['business_address']}")
    print(f"  Country: {s1_rec['country']}")
    ids = str(row['matched_entity_ids']).split(',')
    for mid in ids:
        mid = mid.strip()
        if mid.startswith('S2'):
            rec = s2_idx.loc[mid]
        else:
            rec = s3_idx.loc[mid]
        print(f"  -> {mid}: Name='{rec['business_name']}' Addr='{rec['business_address']}' Country='{rec['country']}'")
    print()
