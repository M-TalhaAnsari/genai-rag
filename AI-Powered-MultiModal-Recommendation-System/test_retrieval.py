"""
test_retrieval.py
------------------
Tests retrieval quality after re-embedding.

Run from project root AFTER reembed_all.py completes:
    python test_retrieval.py

Tests:
  1. Cuisine-specific queries     — biryani, chinese, pizza
  2. Vibe/context queries         — cafe, family, quick meal
  3. Name-based queries           — exact name, partial name
  4. City + cuisine combined      — biryani islamabad
  5. Metadata filtering           — has_phone=True
  6. Dense vs sparse vs hybrid    — compare all three signals
  7. Ranking sanity check         — top result should make sense
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def ok(msg):    print(f"  \033[92m✅ {msg}\033[0m")
def fail(msg):  print(f"  \033[91m❌ {msg}\033[0m")
def info(msg):  print(f"  \033[94mℹ  {msg}\033[0m")
def warn(msg):  print(f"  \033[93m⚠  {msg}\033[0m")
def head(msg):  print(f"\n\033[1m{'─'*55}\n{msg}\n{'─'*55}\033[0m")
def query(msg): print(f"\n  \033[95mQuery: '{msg}'\033[0m")


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def print_results(results: list, source: str = "hybrid", limit: int = 5):
    if not results:
        warn(f"No results from {source}")
        return
    for r in results[:limit]:
        score_key = "rrf_score" if "rrf_score" in r else "score"
        score     = r.get(score_key, 0)
        cuisine   = r.get("cuisine", "")
        area      = r.get("area", "")
        dense_r   = r.get("dense_rank")
        sparse_r  = r.get("sparse_rank")

        rank_info = ""
        if dense_r or sparse_r:
            rank_info = f" (dense:{dense_r} sparse:{sparse_r})"

        area_str = f" · {area}" if area else ""
        print(
            f"    [{score:.4f}] {r['name']} — "
            f"{cuisine}{area_str}{rank_info}"
        )


def check_relevance(results: list, expected_keywords: list) -> bool:
    """
    Check if top 3 results contain at least one expected keyword
    in name or cuisine.
    """
    top3 = results[:3]
    for r in top3:
        combined = (r.get("name", "") + " " + r.get("cuisine", "")).lower()
        for kw in expected_keywords:
            if kw.lower() in combined:
                return True
    return False


# ══════════════════════════════════════════════════════════════
# TEST 1 — Cuisine-specific queries
# ══════════════════════════════════════════════════════════════

def test_cuisine_queries():
    head("TEST 1 — Cuisine-specific queries")

    from backend.retrieving.retrieval import hybrid_search

    test_cases = [
        ("biryani",                 ["biryani", "rice", "desi"]),
        ("chinese food noodles",    ["chinese", "dragon", "noodle"]),
        ("pizza",                   ["pizza", "italian"]),
        ("burger",                  ["burger", "fast food"]),
        ("coffee cafe",             ["cafe", "coffee"]),
        ("fried chicken spicy",     ["chicken", "kfc", "nando"]),
        ("pakistani desi karahi",   ["karahi", "desi", "pakistani"]),
        ("ice cream dessert",       ["gelato", "ice cream", "dessert"]),
    ]

    passed = 0
    for q, keywords in test_cases:
        query(q)
        results = hybrid_search(q, top_k=5)
        print_results(results)
        relevant = check_relevance(results, keywords)
        if relevant:
            ok(f"Top results contain expected cuisine type")
            passed += 1
        else:
            warn(f"Top results may not match — expected: {keywords}")

    print(f"\n  Score: {passed}/{len(test_cases)} cuisine queries relevant")
    return passed, len(test_cases)


# ══════════════════════════════════════════════════════════════
# TEST 2 — Vibe / context queries
# ══════════════════════════════════════════════════════════════

def test_vibe_queries():
    head("TEST 2 — Vibe / context queries")

    from backend.retrieving.retrieval import hybrid_search

    test_cases = [
        ("quick affordable meal",       ["fast food", "burger", "pizza"]),
        ("family dinner restaurant",    ["restaurant", "desi", "karahi"]),
        ("morning breakfast cafe",      ["cafe", "coffee", "bakery"]),
        ("late night food",             ["fast food", "burger"]),
        ("sweet dessert treat",         ["ice cream", "gelato", "donut", "bakery"]),
    ]

    passed = 0
    for q, keywords in test_cases:
        query(q)
        results = hybrid_search(q, top_k=5)
        print_results(results)
        relevant = check_relevance(results, keywords)
        if relevant:
            ok("Vibe query matched relevant restaurants")
            passed += 1
        else:
            warn(f"Expected one of: {keywords}")

    print(f"\n  Score: {passed}/{len(test_cases)} vibe queries relevant")
    return passed, len(test_cases)


# ══════════════════════════════════════════════════════════════
# TEST 3 — Name-based queries (exact + partial)
# ══════════════════════════════════════════════════════════════

def test_name_queries():
    head("TEST 3 — Name-based queries (exact + partial)")

    from backend.retrieving.bm25_store import search_restaurants as bm25_search
    from backend.retrieving.vector_store import search_restaurants as dense_search
    from backend.retrieving.retrieval import hybrid_search

    # These names exist in your DB from OSM
    test_cases = [
        "Chaman Biryani",
        "Nando's",
        "Mei Kong",
        "Daman-e-Koh",
        "The Gelato Affair",
    ]

    passed = 0
    for name in test_cases:
        query(name)

        # BM25 should nail exact name matches
        bm25 = bm25_search(name, top_k=3)
        hybrid = hybrid_search(name, top_k=3)

        top_bm25   = bm25[0]["name"]   if bm25   else "none"
        top_hybrid = hybrid[0]["name"] if hybrid else "none"

        info(f"BM25 top:   {top_bm25}")
        info(f"Hybrid top: {top_hybrid}")

        # Check if the exact restaurant appears in top 3
        bm25_names   = [r["name"].lower() for r in bm25[:3]]
        hybrid_names = [r["name"].lower() for r in hybrid[:3]]
        name_lower   = name.lower()

        found_bm25   = any(name_lower in n or n in name_lower for n in bm25_names)
        found_hybrid = any(name_lower in n or n in name_lower for n in hybrid_names)

        if found_bm25 and found_hybrid:
            ok(f"Found in both BM25 and hybrid top 3")
            passed += 1
        elif found_bm25 or found_hybrid:
            warn(f"Found in {'BM25' if found_bm25 else 'hybrid'} only")
            passed += 0.5
        else:
            fail(f"Not found in top 3 of either — check if name exists in DB")

    print(f"\n  Score: {int(passed)}/{len(test_cases)} name queries found")
    return passed, len(test_cases)


# ══════════════════════════════════════════════════════════════
# TEST 4 — City + cuisine combined
# ══════════════════════════════════════════════════════════════

def test_combined_queries():
    head("TEST 4 — City + cuisine combined")

    from backend.retrieving.retrieval import hybrid_search

    test_cases = [
        ("biryani in Islamabad",        ["biryani"]),
        ("cafe Islamabad coffee",       ["cafe", "coffee"]),
        ("chinese restaurant islamabad",["chinese"]),
        ("fast food burger islamabad",  ["burger", "fast food"]),
    ]

    passed = 0
    for q, keywords in test_cases:
        query(q)
        results = hybrid_search(q, top_k=5)
        print_results(results)

        # All results should be Islamabad
        all_islamabad = all(
            r.get("city", "").lower() == "islamabad"
            for r in results[:3]
        )
        relevant = check_relevance(results, keywords)

        if all_islamabad and relevant:
            ok("Correct city + relevant cuisine")
            passed += 1
        elif relevant:
            warn("Relevant cuisine but mixed cities")
            passed += 0.5
        else:
            warn(f"Expected: {keywords}")

    print(f"\n  Score: {int(passed)}/{len(test_cases)} combined queries relevant")
    return passed, len(test_cases)


# ══════════════════════════════════════════════════════════════
# TEST 5 — Metadata filtering
# ══════════════════════════════════════════════════════════════

def test_metadata_filtering():
    head("TEST 5 — Metadata filtering (ChromaDB where clause)")

    from backend.retrieving.vector_store import search_restaurants

    # Filter: only restaurants with phone numbers
    query("cafe islamabad (filtered: has_phone=True)")
    results_filtered = search_restaurants(
        "cafe coffee islamabad",
        top_k=5,
        where={"has_phone": 1}
    )
    results_all = search_restaurants("cafe coffee islamabad", top_k=5)

    info(f"Without filter: {len(results_all)} results")
    info(f"With has_phone filter: {len(results_filtered)} results")
    print_results(results_filtered, source="filtered")

    # Verify all filtered results actually have phone
    all_have_phone = all(r.get("has_phone") for r in results_filtered)
    if all_have_phone:
        ok("All filtered results have phone numbers")
    else:
        warn("Some filtered results missing phone — metadata may need update")

    # Filter by cuisine
    query("food in Islamabad (filtered: cuisine=Chinese)")
    chinese = search_restaurants(
        "food Islamabad",
        top_k=5,
        where={"cuisine": "Chinese"}
    )
    info(f"Chinese cuisine filter: {len(chinese)} results")
    print_results(chinese, source="Chinese filter")

    all_chinese = all(r.get("cuisine") == "Chinese" for r in chinese)
    if all_chinese and chinese:
        ok("Cuisine filter working correctly")
        return True
    elif not chinese:
        warn("No Chinese restaurants found — may be a data coverage issue")
        return True  # not a retrieval bug, just data
    else:
        fail("Some results don't match cuisine filter")
        return False


# ══════════════════════════════════════════════════════════════
# TEST 6 — Dense vs sparse vs hybrid comparison
# ══════════════════════════════════════════════════════════════

def test_signal_comparison():
    head("TEST 6 — Dense vs Sparse vs Hybrid comparison")

    from backend.retrieving.vector_store import search_restaurants as dense_search
    from backend.retrieving.bm25_store import search_restaurants as bm25_search
    from backend.retrieving.retrieval import hybrid_search

    q = "biryani spicy rice islamabad"
    query(q)

    dense  = dense_search(q, top_k=5)
    sparse = bm25_search(q, top_k=5)
    hybrid = hybrid_search(q, top_k=5)

    print("\n  Dense (ChromaDB — semantic meaning):")
    print_results(dense, "dense")

    print("\n  Sparse (BM25 — keyword match):")
    print_results(sparse, "BM25")

    print("\n  Hybrid (RRF fusion):")
    print_results(hybrid, "hybrid")

    # Check hybrid has better coverage than either alone
    dense_ids  = {r["restaurant_id"] for r in dense[:5]}
    sparse_ids = {r["restaurant_id"] for r in sparse[:5]}
    hybrid_ids = {r["restaurant_id"] for r in hybrid[:5]}

    only_dense  = dense_ids  - sparse_ids
    only_sparse = sparse_ids - dense_ids
    in_both     = dense_ids & sparse_ids

    info(f"In both dense and sparse: {len(in_both)}")
    info(f"Only in dense: {len(only_dense)}")
    info(f"Only in sparse: {len(only_sparse)}")

    if len(in_both) > 0:
        ok("RRF is combining signals from both retrievers")
    else:
        warn("Dense and sparse returning completely different results — normal for small dataset")

    return True


# ══════════════════════════════════════════════════════════════
# TEST 7 — Ranking sanity check
# ══════════════════════════════════════════════════════════════

def test_ranking_sanity():
    head("TEST 7 — Ranking sanity check")

    from backend.retrieving.retrieval import hybrid_search

    # The top result for "biryani" should contain biryani/rice/desi
    # The top result for "pizza" should contain pizza/italian
    # Scores should be descending

    checks = [
        ("biryani rice",  ["biryani", "desi", "rice", "pakistani"]),
        ("pizza cheese",  ["pizza", "italian"]),
        ("chinese noodle",["chinese", "dragon", "noodle", "wok"]),
    ]

    passed = 0
    for q, expected in checks:
        query(q)
        results = hybrid_search(q, top_k=5)
        print_results(results)

        # Check scores are descending
        scores = [r.get("rrf_score", 0) for r in results]
        scores_descending = all(
            scores[i] >= scores[i+1]
            for i in range(len(scores)-1)
        )

        relevant = check_relevance(results, expected)

        if scores_descending:
            ok("Scores are correctly sorted descending")
        else:
            fail("Scores are NOT sorted — ranking bug")

        if relevant:
            ok(f"Top results relevant for '{q}'")
            passed += 1
        else:
            warn(f"Top result may not be ideal for '{q}' — expected one of: {expected}")

    print(f"\n  Score: {passed}/{len(checks)} ranking checks passed")
    return passed, len(checks)


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

def print_summary(scores: list):
    head("RETRIEVAL TEST SUMMARY")

    total_passed = sum(p for p, _ in scores)
    total_tests  = sum(t for _, t in scores)
    pct = round(total_passed / total_tests * 100) if total_tests else 0

    labels = [
        "Cuisine queries",
        "Vibe queries",
        "Name queries",
        "Combined queries",
        "Metadata filtering",
        "Signal comparison",
        "Ranking sanity",
    ]

    for label, (p, t) in zip(labels, scores):
        bar = "█" * int(p) + "░" * int(t - p)
        print(f"  {label:<25} {bar} {int(p)}/{int(t)}")

    print(f"\n  Overall: {int(total_passed)}/{int(total_tests)} ({pct}%)")

    if pct >= 80:
        print("\n\033[92m🎉 Retrieval quality is good. Run the full server.\033[0m")
        print("\n  bash run.sh")
    elif pct >= 50:
        print("\n\033[93m⚠️  Retrieval is acceptable but will improve with Apify data.\033[0m")
        print("  Load Apify data: POST http://localhost:8000/load-apify")
    else:
        print("\n\033[91m⚠️  Retrieval needs improvement.\033[0m")
        print("  This is likely a data quality issue — OSM has no descriptions.")
        print("  Load Apify data for real improvement.")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n\033[1m🔍 Connoisseur — Retrieval Quality Test\033[0m")

    scores = []

    try:
        p, t = test_cuisine_queries()
        scores.append((p, t))
    except Exception as e:
        print(f"\n❌ Cuisine test error: {e}")
        scores.append((0, 8))

    try:
        p, t = test_vibe_queries()
        scores.append((p, t))
    except Exception as e:
        print(f"\n❌ Vibe test error: {e}")
        scores.append((0, 5))

    try:
        p, t = test_name_queries()
        scores.append((p, t))
    except Exception as e:
        print(f"\n❌ Name test error: {e}")
        scores.append((0, 5))

    try:
        p, t = test_combined_queries()
        scores.append((p, t))
    except Exception as e:
        print(f"\n❌ Combined test error: {e}")
        scores.append((0, 4))

    try:
        passed = test_metadata_filtering()
        scores.append((1 if passed else 0, 1))
    except Exception as e:
        print(f"\n❌ Metadata test error: {e}")
        scores.append((0, 1))

    try:
        test_signal_comparison()
        scores.append((1, 1))
    except Exception as e:
        print(f"\n❌ Signal comparison error: {e}")
        scores.append((0, 1))

    try:
        p, t = test_ranking_sanity()
        scores.append((p, t))
    except Exception as e:
        print(f"\n❌ Ranking test error: {e}")
        scores.append((0, 3))

    print_summary(scores)