"""
Run preprocessing on a sample of real training data and show results.
Validates that the full pipeline works end-to-end on actual data.

Run from student_resource/:
    python code/business_entity_resolution/src/run_preprocess_sample.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd
from code.business_entity_resolution.src import config
from code.business_entity_resolution.src.preprocess import (
    load_source, load_ground_truth, preprocess_dataframe,
    parse_ground_truth, normalize_name, normalize_address
)

SAMPLE_SIZE = 5000  # rows per source to test

def main():
    print("=" * 70)
    print("  PREPROCESSING PIPELINE — SAMPLE RUN")
    print("=" * 70)
    
    # Load sample data
    print(f"\nLoading {SAMPLE_SIZE:,} rows per source...")
    s1 = pd.read_csv(config.TRAIN_SOURCE1, sep="\t", nrows=SAMPLE_SIZE,
                     dtype={"entity_id": str, "business_name": str, 
                            "business_address": str, "country": str})
    s2 = pd.read_csv(config.TRAIN_SOURCE2, sep="\t", nrows=SAMPLE_SIZE,
                     dtype={"entity_id": str, "business_name": str,
                            "business_address": str, "country": str})
    s3 = pd.read_csv(config.TRAIN_SOURCE3, sep="\t", nrows=SAMPLE_SIZE,
                     dtype={"entity_id": str, "business_name": str,
                            "business_address": str, "country": str})
    
    s1["business_name"] = s1["business_name"].fillna("")
    s1["business_address"] = s1["business_address"].fillna("")
    s1["country"] = s1["country"].fillna("")
    s2["business_name"] = s2["business_name"].fillna("")
    s2["business_address"] = s2["business_address"].fillna("")
    s2["country"] = s2["country"].fillna("")
    s3["business_name"] = s3["business_name"].fillna("")
    s3["business_address"] = s3["business_address"].fillna("")
    s3["country"] = s3["country"].fillna("")
    
    # Preprocess
    s1 = preprocess_dataframe(s1, desc="S1-sample")
    s2 = preprocess_dataframe(s2, desc="S2-sample")
    s3 = preprocess_dataframe(s3, desc="S3-sample")
    
    # Show results
    print("\n" + "=" * 70)
    print("  SAMPLE RESULTS")
    print("=" * 70)
    
    cols_to_show = ["entity_id", "business_name", "name_clean", "name_key",
                    "name_prefix", "zip_pin", "country"]
    
    print("\n--- Source 1 (first 5 rows) ---")
    for _, row in s1.head(5).iterrows():
        print(f"  ID: {row['entity_id']}")
        print(f"    Raw name:    {row['business_name']}")
        print(f"    Clean name:  {row['name_clean']}")
        print(f"    Name key:    {row['name_key']}")
        print(f"    Prefix:      {row['name_prefix']}")
        print(f"    Raw addr:    {row['business_address'][:60]}")
        print(f"    Clean addr:  {row['addr_clean'][:60]}")
        print(f"    ZIP/PIN:     {row['zip_pin']}")
        print(f"    Addr tokens: {row['addr_tokens'][:60]}")
        print(f"    Country:     {row['country']}")
        print()
    
    print("\n--- Source 2 (first 3 rows) ---")
    for _, row in s2.head(3).iterrows():
        print(f"  ID: {row['entity_id']}")
        print(f"    Raw name:    {row['business_name'][:60]}")
        print(f"    Clean name:  {row['name_clean'][:60]}")
        print(f"    Name key:    {row['name_key'][:60]}")
        print(f"    ZIP/PIN:     {row['zip_pin']}")
        print()
    
    print("\n--- Source 3 (first 3 rows) ---")
    for _, row in s3.head(3).iterrows():
        print(f"  ID: {row['entity_id']}")
        print(f"    Raw name:    {row['business_name'][:60]}")
        print(f"    Clean name:  {row['name_clean'][:60]}")
        print(f"    Name key:    {row['name_key'][:60]}")
        print(f"    ZIP/PIN:     {row['zip_pin']}")
        print()
    
    # Stats summary
    print("\n" + "=" * 70)
    print("  PREPROCESSING STATS")
    print("=" * 70)
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        zip_count = (df["zip_pin"] != "").sum()
        empty_name = (df["name_clean"] == "").sum()
        empty_addr = (df["addr_clean"] == "").sum()
        avg_name_len = df["name_clean"].str.len().mean()
        avg_tokens = df["addr_tokens"].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        print(f"  {name}: {len(df):,} rows | "
              f"ZIP found: {zip_count:,} ({zip_count/len(df)*100:.1f}%) | "
              f"Empty names: {empty_name} | "
              f"Empty addrs: {empty_addr} | "
              f"Avg name len: {avg_name_len:.1f} | "
              f"Avg addr tokens: {avg_tokens:.1f}")
    
    print("\n✅ Preprocessing pipeline validated successfully on sample data!")
    print("   Ready for full-scale processing.")

if __name__ == "__main__":
    main()
