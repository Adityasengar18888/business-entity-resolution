"""
Configuration module for Business Entity Resolution pipeline.
Contains all paths, hyperparameters, and constants.
"""
import os

# ============================================================
# Base Paths
# ============================================================
# Resolve paths relative to project root (student_resource/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Training files
TRAIN_SOURCE1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_SOURCE2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_SOURCE3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GROUND_TRUTH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

# Test files
TEST_SOURCE1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_SOURCE2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_SOURCE3 = os.path.join(TEST_DIR, "test_source3.tsv")

# Output files
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# ============================================================
# Preprocessing Constants
# ============================================================

# Legal suffix patterns to strip for matching keys
LEGAL_SUFFIXES = [
    "incorporated", "corporation", "company", "limited",
    "private", "public", "enterprises", "enterprise",
    "associates", "association", "holdings", "group",
    "solutions", "services", "technologies", "technology",
    "industries", "international", "consultants", "consulting",
    "ventures", "partners", "partnership",
    "inc", "corp", "co", "ltd", "llc", "llp", "plc",
    "pvt", "sa", "sas", "sarl", "gmbh", "ag",
    "dba", "pllc", "lp",
]

# Common abbreviation expansions (applied during normalization)
NAME_ABBREVIATIONS = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "co": "company",
    "intl": "international",
    "assoc": "associates",
    "mfg": "manufacturing",
    "svcs": "services",
    "svc": "service",
    "tech": "technology",
    "grp": "group",
    "hldgs": "holdings",
    "mgmt": "management",
    "natl": "national",
    "dept": "department",
    "govt": "government",
    "engg": "engineering",
}

ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "cir": "circle",
    "hwy": "highway",
    "pkwy": "parkway",
    "sq": "square",
    "ter": "terrace",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "rm": "room",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "mt": "mount",
    "ft": "fort",
    "jn": "junction",
    "expy": "expressway",
    "fwy": "freeway",
    "tpke": "turnpike",
    "po": "post office",
}

# US state abbreviation → full name
US_STATE_MAP = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

# Indian state abbreviation → full name
INDIA_STATE_MAP = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam",
    "br": "bihar", "cg": "chhattisgarh", "ga": "goa", "gj": "gujarat",
    "hr": "haryana", "hp": "himachal pradesh", "jk": "jammu and kashmir",
    "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram",
    "nl": "nagaland", "od": "odisha", "pb": "punjab", "rj": "rajasthan",
    "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana", "tr": "tripura",
    "up": "uttar pradesh", "uk": "uttarakhand", "wb": "west bengal",
    "dl": "delhi",
}

# ============================================================
# Blocking Hyperparameters
# ============================================================
TFIDF_NGRAM_RANGE = (2, 4)        # Character n-gram range for TF-IDF
TFIDF_TOP_K = 20                   # Top-K candidates from TF-IDF blocking
NAME_PREFIX_LEN = 5                # First N chars of name for prefix blocking
MAX_CANDIDATES_PER_ENTITY = 200    # Hard cap on candidates per S1 entity
MIN_BLOCK_SIMILARITY = 0.1         # Minimum TF-IDF cosine to consider

# ============================================================
# Model Hyperparameters
# ============================================================
NEGATIVE_RATIO = 4                 # Negative:Positive sampling ratio
VALIDATION_SPLIT = 0.2             # Fraction of S1 entities held out
MATCH_THRESHOLD = 0.5              # Initial threshold (will be tuned)
RANDOM_SEED = 42

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 100,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_SEED,
}

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
