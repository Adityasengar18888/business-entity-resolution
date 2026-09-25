"""
Data loading and text preprocessing module.

Handles:
- Efficient TSV loading with proper dtypes
- Text normalization (lowercase, unicode, punctuation)
- Abbreviation expansion
- Non-Latin script transliteration
- Address component extraction (ZIP/PIN, city, state)
"""
import re
import unicodedata
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from tqdm import tqdm

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:
    sanscript = None

try:
    from unidecode import unidecode
except ImportError:
    # Fallback: strip non-ASCII entirely
    def unidecode(text):
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

from . import config


# ============================================================
# 1. Data Loading
# ============================================================

def load_source(filepath: str) -> pd.DataFrame:
    """
    Load a source TSV file with proper dtypes and encoding.
    
    Args:
        filepath: Path to the .tsv file
        
    Returns:
        DataFrame with columns: entity_id, business_name, business_address, country
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype={"entity_id": str, "business_name": "string", "business_address": "string", "country": "category"},
        encoding="utf-8",
        na_values=["", "nan", "null", "None", "NaN"],
        keep_default_na=True,
    )
    # Fill missing values
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")
    
    print(f"  Loaded {filepath}: {len(df):,} records, "
          f"null names={df['business_name'].eq('').sum():,}, "
          f"null addrs={df['business_address'].eq('').sum():,}")
    return df


def load_ground_truth(filepath: str) -> pd.DataFrame:
    """
    Load ground truth file with proper handling of empty match lists.
    
    Returns:
        DataFrame with columns: source1_entity_id, matched_entity_ids (str or NaN)
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype={"source1_entity_id": str, "matched_entity_ids": str},
        encoding="utf-8",
        keep_default_na=True,
    )
    print(f"  Loaded ground truth: {len(df):,} S1 entities, "
          f"singletons={df['matched_entity_ids'].isna().sum():,}")
    return df


def load_train_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all training data files."""
    print("Loading training data...")
    s1 = load_source(config.TRAIN_SOURCE1)
    s2 = load_source(config.TRAIN_SOURCE2)
    s3 = load_source(config.TRAIN_SOURCE3)
    gt = load_ground_truth(config.TRAIN_GROUND_TRUTH)
    return s1, s2, s3, gt


def load_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all test data files."""
    print("Loading test data...")
    s1 = load_source(config.TEST_SOURCE1)
    s2 = load_source(config.TEST_SOURCE2)
    s3 = load_source(config.TEST_SOURCE3)
    return s1, s2, s3


# ============================================================
# 2. Text Normalization
# ============================================================

def unicode_normalize(text: str) -> str:
    """Apply Unicode NFKD normalization — decomposes accented characters."""
    return unicodedata.normalize("NFKD", text)


def strip_accents(text: str) -> str:
    """Remove accent marks while keeping base characters. é → e, ñ → n."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_punctuation(text: str) -> str:
    """
    Normalize punctuation:
    - Replace & with 'and'
    - Replace hyphens/dashes with space
    - Remove dots, commas, quotes, parentheses etc.
    - Collapse multiple spaces
    """
    # & → and
    text = re.sub(r"\s*&\s*", " and ", text)
    # Hyphens and dashes → space
    text = re.sub(r"[-–—]+", " ", text)
    # Remove periods, commas, quotes, parens, brackets, slashes
    text = re.sub(r"[.,;:!?\"'`()\[\]{}/\\#@*^~|]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_abbreviations(text: str, abbrev_map: Dict[str, str]) -> str:
    """
    Expand known abbreviations in text using word-boundary matching.
    Only replaces whole words, not substrings.
    """
    tokens = text.split()
    expanded = []
    for token in tokens:
        # Strip trailing dots for abbreviation matching (e.g., "pvt." → "pvt")
        clean_token = token.rstrip(".")
        if clean_token in abbrev_map:
            expanded.append(abbrev_map[clean_token])
        else:
            expanded.append(token)
    return " ".join(expanded)


def transliterate_to_latin(text: str) -> str:
    """
    Convert non-Latin scripts (Devanagari, Tamil, Kannada, etc.) to 
    approximate Latin/ASCII representation using unidecode.
    
    Examples:
        "राम मार्केटिंग" → "ram marketimg"
        "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்" → "raj investmentss"
    """
    if any(ord(c) > 127 for c in text):
        res = text
        if sanscript:
            try:
                # Try Indic transliteration for supported scripts
                res = transliterate(res, sanscript.DEVANAGARI, sanscript.ITRANS)
                res = transliterate(res, sanscript.TAMIL, sanscript.ITRANS)
                res = transliterate(res, sanscript.KANNADA, sanscript.ITRANS)
            except:
                pass
        # Unidecode handles any remaining non-ASCII (e.g. French accents, or other scripts)
        return unidecode(res)
    return text


def remove_url_suffix(name: str) -> str:
    """
    Strip .com, .in, .org etc. from URL-style business names.
    Example: 'maurewilliamscolombier.com' → 'maurewilliamscolombier'
    """
    return re.sub(r"\.(com|org|net|in|co|io|biz|info|us|fr)$", "", name, flags=re.IGNORECASE)


def remove_leading_noise(name: str) -> str:
    """Remove leading dashes, dots, and whitespace from business names."""
    return re.sub(r"^[\s\-\.]+", "", name)


def normalize_name(name: str) -> str:
    """
    Full normalization pipeline for business names.
    
    Steps:
        1. Lowercase
        2. Transliterate non-Latin to ASCII
        3. Strip accents
        4. Remove URL suffixes
        5. Remove leading noise (-- prefixes)
        6. Clean punctuation
        7. Expand abbreviations
    
    Returns:
        Normalized name string
    """
    if not name or pd.isna(name):
        return ""
    
    text = str(name).lower().strip()
    text = transliterate_to_latin(text)
    text = strip_accents(text)
    text = remove_url_suffix(text)
    text = remove_leading_noise(text)
    text = clean_punctuation(text)
    text = expand_abbreviations(text, config.NAME_ABBREVIATIONS)
    # Final whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name_for_key(name: str) -> str:
    """
    Aggressive normalization for blocking keys — removes legal suffixes
    to maximize recall during candidate generation.
    
    'Raj Investments Private Limited' → 'raj investments'
    """
    text = normalize_name(name)
    # Remove legal suffixes
    tokens = text.split()
    cleaned = [t for t in tokens if t not in config.LEGAL_SUFFIXES]
    return " ".join(cleaned) if cleaned else text


def normalize_address(address: str) -> str:
    """
    Full normalization pipeline for business addresses.
    
    Steps:
        1. Lowercase
        2. Transliterate non-Latin to ASCII
        3. Strip accents
        4. Clean punctuation
        5. Expand address abbreviations
        6. Normalize state names
    
    Returns:
        Normalized address string
    """
    if not address or pd.isna(address):
        return ""
    
    text = str(address).lower().strip()
    text = transliterate_to_latin(text)
    text = strip_accents(text)
    text = clean_punctuation(text)
    text = expand_abbreviations(text, config.ADDRESS_ABBREVIATIONS)
    
    # Normalize US state abbreviations in the address
    tokens = text.split()
    normalized_tokens = []
    for token in tokens:
        clean = token.rstrip(",").strip()
        if clean in config.US_STATE_MAP:
            normalized_tokens.append(config.US_STATE_MAP[clean])
        elif clean in config.INDIA_STATE_MAP:
            normalized_tokens.append(config.INDIA_STATE_MAP[clean])
        else:
            normalized_tokens.append(token)
    text = " ".join(normalized_tokens)
    
    # Remove 'null' placeholder that appears in some addresses
    text = re.sub(r"\bnull\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# 3. Address Component Extraction
# ============================================================

# Regex patterns for ZIP/PIN codes
ZIP_PATTERN = re.compile(r"\b(\d{5})(?:-\d{4})?\b")   # US: 5 digits, optional +4
PIN_PATTERN = re.compile(r"\b(\d{6})\b")                # India: 6 digits
FR_ZIP_PATTERN = re.compile(r"\b(\d{5})\b")             # France: 5 digits


def extract_zip_pin(address: str, country: str = "") -> str:
    """
    Extract postal/ZIP code from address based on country.
    Uses position-aware patterns to avoid matching street numbers.
    
    Returns:
        ZIP/PIN code string, or '' if not found
    """
    if not address or pd.isna(address):
        return ""
    
    country_lower = country.lower().strip()
    
    if country_lower == "us":
        # US ZIP: 5 digits, optionally followed by -4 digits
        matches = re.findall(r'\b(\d{5})(?:-\d{4})?\b', address)
        # Prefer the last match (ZIPs typically at end)
        return matches[-1] if matches else ""
    elif country_lower == "india":
        # India PIN: 6 digits, not first token
        matches = re.findall(r'\b(\d{6})\b', address)
        if matches:
            if len(matches) == 1 and address.strip().startswith(matches[0]):
                return "" # Probably a door number
            return matches[-1]
    elif country_lower == "france":
        # France: 5 digits, typically at start
        matches = re.findall(r'\b(\d{5})\b', address)
        return matches[0] if matches else ""
    else:
        # Unknown country — try PIN first (6 digits), then ZIP (5 digits)
        matches_pin = list(PIN_PATTERN.finditer(address))
        if matches_pin and matches_pin[-1].start() > 0:
            return matches_pin[-1].group(1)
        matches_zip = list(ZIP_PATTERN.finditer(address))
        if matches_zip and matches_zip[-1].start() > 5:
            return matches_zip[-1].group(1)
    
    return ""


def extract_street_number(address: str) -> str:
    """Extract leading street/door number from address."""
    if not address:
        return ""
    match = re.match(r"^(\d+[\w/-]*)", address.strip())
    return match.group(1) if match else ""


def extract_address_tokens(address: str) -> list:
    """
    Extract significant tokens from address (removes common filler words).
    Useful for token-overlap blocking.
    """
    if not address:
        return []
    
    stop_words = {
        "near", "behind", "opposite", "beside", "next", "to", "the",
        "of", "at", "in", "on", "and", "or", "no", "number",
        "floor", "block", "sector", "phase", "plot", "kh", "survey",
        "post", "office", "box", "po", "null",
    }
    
    tokens = normalize_address(address).split()
    # Keep only alphabetic tokens of length >= 3 that aren't stop words
    return [t for t in tokens if len(t) >= 3 and t not in stop_words and not t.isdigit()]


# ============================================================
# 4. Apply Preprocessing to DataFrames
# ============================================================

def preprocess_dataframe(df: pd.DataFrame, desc: str = "") -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to a source DataFrame.
    
    Adds columns:
        - name_clean: normalized business name
        - name_key: aggressive normalization for blocking (no legal suffixes)
        - addr_clean: normalized address
        - zip_pin: extracted ZIP/PIN code
        - name_prefix: first N chars of name_key (for prefix blocking)
        - addr_tokens: significant address tokens as space-joined string
    
    Args:
        df: Source DataFrame with entity_id, business_name, business_address, country
        desc: Description string for progress bar
        
    Returns:
        DataFrame with additional preprocessed columns
    """
    print(f"  Preprocessing {desc} ({len(df):,} records)...")
    
    tqdm.pandas(desc=f"  Normalizing names [{desc}]")
    df["name_clean"] = df["business_name"].progress_apply(normalize_name)
    
    tqdm.pandas(desc=f"  Name keys [{desc}]")
    df["name_key"] = df["business_name"].progress_apply(normalize_name_for_key)
    
    tqdm.pandas(desc=f"  Normalizing addresses [{desc}]")
    df["addr_clean"] = df["business_address"].progress_apply(normalize_address)
    
    # Extract ZIP/PIN codes (vectorized-friendly)
    print(f"  Extracting ZIP/PIN codes [{desc}]...")
    df["zip_pin"] = df.apply(
        lambda row: extract_zip_pin(row["business_address"], row["country"]),
        axis=1
    )
    
    # Name prefix for blocking
    df["name_prefix"] = df["name_key"].str[:config.NAME_PREFIX_LEN]
    
    # Address tokens as space-joined string (for TF-IDF)
    print(f"  Extracting address tokens [{desc}]...")
    df["addr_tokens"] = df["business_address"].apply(
        lambda x: " ".join(extract_address_tokens(x))
    )
    
    # Flag columns for null values
    df["has_name"] = df["business_name"].ne("")
    df["has_address"] = df["business_address"].ne("")
    
    # Summary stats
    zip_found = (df["zip_pin"] != "").sum()
    print(f"  Done [{desc}]: ZIP/PIN found for {zip_found:,}/{len(df):,} "
          f"({zip_found/len(df)*100:.1f}%)")
    
    return df


def preprocess_all_train() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and preprocess all training data."""
    s1, s2, s3, gt = load_train_data()
    
    s1 = preprocess_dataframe(s1, desc="S1-train")
    s2 = preprocess_dataframe(s2, desc="S2-train")
    s3 = preprocess_dataframe(s3, desc="S3-train")
    
    return s1, s2, s3, gt


def preprocess_all_test() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and preprocess all test data."""
    s1, s2, s3 = load_test_data()
    
    s1 = preprocess_dataframe(s1, desc="S1-test")
    s2 = preprocess_dataframe(s2, desc="S2-test")
    s3 = preprocess_dataframe(s3, desc="S3-test")
    
    return s1, s2, s3


# ============================================================
# 5. Ground Truth Utilities
# ============================================================

def parse_ground_truth(gt: pd.DataFrame) -> Dict[str, list]:
    """
    Parse ground truth into a dictionary mapping S1 entity_id → list of matched IDs.
    
    Args:
        gt: Ground truth DataFrame
        
    Returns:
        Dict[str, list]: e.g., {'S1-001': ['S2-047', 'S3-812'], 'S1-002': []}
    """
    gt_dict = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        if pd.isna(row["matched_entity_ids"]) or str(row["matched_entity_ids"]).strip() == "":
            gt_dict[s1_id] = []
        else:
            gt_dict[s1_id] = [x.strip() for x in str(row["matched_entity_ids"]).split(",")]
    return gt_dict


def build_match_set(gt_dict: Dict[str, list]) -> set:
    """
    Build a set of (s1_id, matched_id) tuples for fast lookup.
    
    Args:
        gt_dict: Parsed ground truth dictionary
        
    Returns:
        Set of (s1_entity_id, matched_entity_id) tuples
    """
    pairs = set()
    for s1_id, matched_ids in gt_dict.items():
        for m_id in matched_ids:
            pairs.add((s1_id, m_id))
    return pairs
