# Phase 3: Feature Engineering Plan

## 1. Goal & Requirements
The goal of Phase 3 is to take the raw text of the candidate pairs generated in Phase 2 and transform them into **numeric feature vectors** that a machine learning model (like LightGBM or XGBoost) can understand. 

For each candidate pair `(S1_Entity, S2/S3_Entity)`, we must compute mathematical similarities between their respective `business_name`, `business_address`, `zip_pin`, and blocking keys.

### 1.1 Requirements
1. **String Distance Metrics:** Compute edit distances that account for typos.
2. **Token-based Metrics:** Compute overlap of words to handle reordering (e.g., "Cafe Roma" vs "Roma Cafe").
3. **Phonetic Metrics:** Check if words sound the same.
4. **Missing Value Handling:** Robustly handle missing addresses or ZIP codes.
5. **High Performance:** Feature extraction must process tens of millions of pairs quickly using optimized C/C++ backed libraries.

## 2. Feature Strategy

We will extract the following categories of features:

### 2.1 Name Similarity Features
| Feature Name | Algorithm / Library | Purpose |
|--------------|---------------------|---------|
| `name_jaro_winkler` | `rapidfuzz.distance.JaroWinkler` | Heavily weights matches at the beginning of the string. Excellent for business names. |
| `name_levenshtein_ratio` | `rapidfuzz.distance.Levenshtein` | Captures character-level typos and substitutions. |
| `name_token_jaccard` | Custom Set Intersection | Handles out-of-order words (`len(A & B) / len(A | B)`). |
| `name_token_sort_ratio`| `rapidfuzz.fuzz.token_sort_ratio` | Handles completely scrambled words. |
| `name_length_diff` | `abs(len(A) - len(B))` | Captures extreme abbreviations. |

### 2.2 Address Similarity Features
| Feature Name | Algorithm / Library | Purpose |
|--------------|---------------------|---------|
| `addr_levenshtein` | `rapidfuzz.distance.Levenshtein` | Character-level typos in addresses. |
| `addr_token_jaccard` | Custom Set Intersection | Very robust to missing tokens like "floor" or "suite". |
| `addr_cosine_sim` | TF-IDF + Cosine | Re-use our TF-IDF vectors from Phase 2 for a direct numerical similarity score. |

### 2.3 Exact & Heuristic Features
| Feature Name | Logic | Purpose |
|--------------|-------|---------|
| `zip_match` | `1` if exact match, `0` if different, `-1` if missing | Powerful geographic constraint. |
| `state_match` | Extract state from tokens and compare | Broad geographic constraint. |
| `has_null_address` | Boolean | Lets the model know it shouldn't penalize `addr_levenshtein = 0` if one address is missing. |

## 3. Potential Problems & Mitigation

### Problem 1: Computational Bottleneck
* **The Issue:** Computing Levenshtein distance natively in Python for 50 million pairs will take hours.
* **The Solution:** We **must** use `rapidfuzz`, which provides C++ optimized string distances. We will also use `pandas` vectorized operations (`apply` is often too slow, so we will use list comprehensions or `numpy.vectorize`). We can also process in chunks using `multiprocessing`.

### Problem 2: Handling Missing Data
* **The Issue:** Many records lack addresses. If we compute string distance between an empty string and a valid address, it results in `0` similarity. The ML model might incorrectly learn that "0 address similarity means it's not a match," even though it could be a perfect name match!
* **The Solution:** Instead of leaving similarity as `0`, we will impute missing similarities with `-1` (or `np.nan` since LightGBM handles NaNs natively). We also include explicit binary flags like `is_address_missing = True`.

### Problem 3: Memory Exhaustion
* **The Issue:** Expanding 50 million candidate pairs into a DataFrame with 15 float64 features will consume over 10GB of RAM.
* **The Solution:** 
  1. Cast similarity float scores to `float32`.
  2. Cast boolean flags to `int8`.
  3. Write features directly to disk (Parquet or TSV) in chunks of 5 million pairs.

## 4. Development Workflow
1. **Feature Engineering Script:** Create `features.py`.
2. **Test on Sample:** Run the feature extractor on the 20k validation candidate pairs we generate in Phase 2.
3. **Distribution Check:** Plot histograms of `name_jaro_winkler` for **True Matches** vs **False Matches**. We want to see a clear separation (bimodal distribution).
4. **Scale Up:** Apply the script to the full `candidate_pairs.tsv` in memory-safe chunks.
