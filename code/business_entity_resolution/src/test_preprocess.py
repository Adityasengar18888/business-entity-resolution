"""
Test script for the preprocessing module.
Validates normalization on real data examples from the training set.

Run from student_resource/:
    python -m code.business_entity_resolution.src.test_preprocess
"""
import sys
import os

# Fix Windows terminal encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# Add project root to path so relative imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from code.business_entity_resolution.src.preprocess import (
    normalize_name,
    normalize_name_for_key,
    normalize_address,
    extract_zip_pin,
    extract_street_number,
    extract_address_tokens,
    transliterate_to_latin,
    strip_accents,
    remove_url_suffix,
)


def test_name_normalization():
    """Test business name normalization on real noise patterns."""
    print("=" * 60)
    print("TEST: Name Normalization")
    print("=" * 60)
    
    test_cases = [
        # (input, expected_contains) — we check key tokens are present
        ("Maure Wilblims Colombier Inc", "maure wilblims colombier"),
        ("Maure Williams Colombier", "maure williams colombier"),
        ("-- Holloway Peak Inc Seafood", "holloway peak"),
        ("maurewilliamscolombier.com", "maurewilliamscolombier"),
        ("Payne Énterprises", "payne enterprises"),
        ("PAYNE-ENRTPRMISES", "payne enrtprmises"),
        ("Raj Investments LLP", "raj investments llp"),
        ("B+ Retail Inc", "b retail"),
        ("Delta Tetlecommunication Inc", "delta tetlecommunication"),
        ("LLC Moncada Léarning Center", "llc moncada learning center"),
        ("Pvt. EFS Print Ventures Ltd.", "private efs print ventures limited"),
        ("Ss Food Private Limited", "ss food private limited"),
    ]
    
    passed = 0
    for raw, expected in test_cases:
        result = normalize_name(raw)
        ok = all(tok in result for tok in expected.split())
        status = "✓" if ok else "✗"
        if not ok:
            print(f"  {status} FAIL: '{raw}'")
            print(f"       Expected tokens: {expected}")
            print(f"       Got:             {result}")
        else:
            print(f"  {status} '{raw}' → '{result}'")
            passed += 1
    
    print(f"\n  Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_name_key_normalization():
    """Test blocking-key name normalization (strips legal suffixes)."""
    print("\n" + "=" * 60)
    print("TEST: Name Key (Blocking) Normalization")
    print("=" * 60)
    
    test_cases = [
        ("Raj Investments Private Limited", "raj investments"),
        ("Payne Enterprises LLC", "payne"),
        ("B+ Retail Inc", "b retail"),
        ("Dahlia Power Reliable Scientific LLC", "dahlia power reliable scientific"),
    ]
    
    passed = 0
    for raw, expected in test_cases:
        result = normalize_name_for_key(raw)
        # Check that legal suffixes are removed
        has_legal = any(s in result.split() for s in ["inc", "llc", "ltd", "pvt", "limited", "private", "corporation"])
        ok = not has_legal
        status = "✓" if ok else "✗"
        print(f"  {status} '{raw}' → '{result}'")
        if ok:
            passed += 1
        else:
            print(f"       Still contains legal suffix!")
    
    print(f"\n  Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_transliteration():
    """Test non-Latin script transliteration."""
    print("\n" + "=" * 60)
    print("TEST: Script Transliteration")
    print("=" * 60)
    
    test_cases = [
        ("राम मार्केटिंग प्राइवेट लिमिटेड", True),   # Hindi
        ("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", True),  # Tamil
        ("Payne Enterprises", False),                     # Already Latin
        ("Payne Énterprises", False),                     # Accented Latin
    ]
    
    passed = 0
    for text, should_change in test_cases:
        result = transliterate_to_latin(text)
        is_ascii = all(ord(c) < 128 for c in result)
        changed = result != text
        ok = is_ascii  # Result should always be ASCII-safe
        status = "✓" if ok else "✗"
        print(f"  {status} '{text[:40]}...' → '{result[:40]}...' [ASCII: {is_ascii}]")
        if ok:
            passed += 1
    
    print(f"\n  Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_address_normalization():
    """Test address normalization on real examples."""
    print("\n" + "=" * 60)
    print("TEST: Address Normalization")
    print("=" * 60)
    
    test_cases = [
        ("3315 FREMONT ST, PEORIA, IL", "3315 fremont street peoria illinois"),
        ("85 Wanye Avenue, Ticonderoga Townshiip, New York", "85 wanye avenue ticonderoga"),
        ("KANSAS CITY, MO, 630 45ND TERRACE, null", "kansas city"),
        ("AF-0684, NANDGRAM NEAR MOTHER INDIA PUBLIC SCHOOL. PH. 989, GHAZIABAD, 9487203, उत्तर प्रदेश", "nandgram"),
        ("1795 Westchester Drive, High Point, NC", "1795 westchester drive"),
    ]
    
    passed = 0
    for raw, expected_substr in test_cases:
        result = normalize_address(raw)
        ok = expected_substr in result
        status = "✓" if ok else "✗"
        print(f"  {status} '{raw[:50]}...'")
        print(f"       → '{result[:70]}'")
        if not ok:
            print(f"       Expected to contain: '{expected_substr}'")
        else:
            passed += 1
    
    print(f"\n  Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_zip_extraction():
    """Test ZIP/PIN code extraction."""
    print("\n" + "=" * 60)
    print("TEST: ZIP/PIN Extraction")
    print("=" * 60)
    
    test_cases = [
        ("1795 Westchester Drive, High Point, NC 27265", "US", "27265"),
        ("6(29), C.I.T. COLONY, CHENNAI, Tamil Nadu 600004", "India", "600004"),
        ("3315 FREMONT ST, PEORIA, IL", "US", ""),
        ("", "US", ""),
        ("797, Lake Town Block A, Kolkata, West Bengal", "India", ""),
    ]
    
    passed = 0
    for addr, country, expected in test_cases:
        result = extract_zip_pin(addr, country)
        ok = result == expected
        status = "✓" if ok else "✗"
        print(f"  {status} ({country}) '{addr[:40]}' → ZIP: '{result}' (expected: '{expected}')")
        if ok:
            passed += 1
    
    print(f"\n  Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_url_removal():
    """Test URL suffix removal."""
    print("\n" + "=" * 60)
    print("TEST: URL Suffix Removal")
    print("=" * 60)
    
    test_cases = [
        ("maurewilliamscolombier.com", "maurewilliamscolombier"),
        ("wilfordhancock.com", "wilfordhancock"),
        ("normal business name", "normal business name"),
        ("domain.org", "domain"),
    ]
    
    passed = 0
    for raw, expected in test_cases:
        result = remove_url_suffix(raw)
        ok = result == expected
        status = "✓" if ok else "✗"
        print(f"  {status} '{raw}' → '{result}'")
        if ok:
            passed += 1
    
    print(f"\n  Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


if __name__ == "__main__":
    print("Running preprocessing unit tests...\n")
    
    results = []
    results.append(("Name Normalization", test_name_normalization()))
    results.append(("Name Key (Blocking)", test_name_key_normalization()))
    results.append(("Transliteration", test_transliteration()))
    results.append(("Address Normalization", test_address_normalization()))
    results.append(("ZIP/PIN Extraction", test_zip_extraction()))
    results.append(("URL Removal", test_url_removal()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n  🎉 All tests passed!")
    else:
        print("\n  ⚠️  Some tests failed — review output above.")
    
    sys.exit(0 if all_pass else 1)
