"""
Introduction to Information Retrieval — Lecture 6 Exercise
==========================================================

Runs in a low-resource environment (no GPU, a few GB of memory).
Uses the standard library only (no `pip install` required).

How to run:
    python exercise.py

Implement each task's function and make the tests pass.
"""

import math
import re
from collections import Counter, defaultdict

# =============================================================================
# Corpus (a small English document collection, pre-tokenized)
# =============================================================================

CORPUS = {
    "d01": "Kyoto foliage is the highlight of autumn . Arashiyama and Tofukuji are famous .",
    "d02": "Tokyo cherry blossoms are a spring tradition . Ueno Park and Meguro River are popular .",
    "d03": "Kyoto cherry blossoms are also beautiful . Visit Kyoto temples and Kyoto gardens such as the Philosopher Path .",
    "d04": "Nara foliage is admired for its beauty . Tanzan Shrine is a hidden gem .",
    "d05": "Osaka street food is fun . Dotonbori and Shinsekai are famous .",
    "d06": "Kyoto sightseeing centers on temple and shrine tours . Kinkakuji and Kiyomizudera are popular .",
    "d07": "Famous foliage spots are abundant nationwide . Nikko and Hakone are great in autumn .",
    "d08": "In autumn Kyoto is crowded during foliage season . An early morning stroll is recommended .",
    "d09": "A search engine uses an inverted index to retrieve documents quickly .",
    "d10": "Information search evaluation relies on relevance and ranking metrics .",
    "d11": "Kamakura temples have deep history . In hydrangea season many sightseeing visitors arrive .",
    "d12": "Mount Fuji is a landmark representing Japan . Climbing is best in summer .",
    "d13": "The Okinawa sea has high transparency . Diving and snorkeling are popular .",
    "d14": "Hokkaido lavender fields are a summer tradition . Furano is famous .",
    "d15": "The Hiroshima Peace Memorial Park is a World Heritage site . Many foreigners visit .",
}

# Relevance judgments (qrels)
QRELS = {
    "q1": {"d01": 2, "d04": 1, "d07": 1, "d08": 2, "d03": 0},  # "Kyoto foliage"
    "q2": {"d06": 2, "d01": 1, "d08": 1, "d03": 1},             # "Kyoto sightseeing"
    "q3": {"d09": 2, "d10": 1},                                   # "search engine"
}

QUERIES = {
    "q1": "Kyoto foliage",
    "q2": "Kyoto sightseeing",
    "q3": "search engine",
}


def tokenize(text: str) -> list[str]:
    """Tokenize on whitespace, dropping punctuation tokens."""
    return [t for t in text.split() if t not in {".", ",", ";", ":", "。", "、", "，", "．"}]


# =============================================================================
# Task 1: Build an inverted index (15 points)
# =============================================================================

def build_inverted_index(corpus: dict[str, str]) -> dict[str, list[str]]:
    """
    Build an inverted index.

    Parameters
    ----------
    corpus : dict[str, str]
        A dictionary mapping document ID -> document text.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping term -> [list of document IDs] (the posting list).
        Document IDs must be sorted in ascending order.

    Example
    -------
    >>> idx = build_inverted_index({"d01": "A B C", "d02": "B C D"})
    >>> idx["B"]
    ['d01', 'd02']
    >>> "A" in idx and idx["A"] == ["d01"]
    True
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 1")


# =============================================================================
# Task 2: Boolean retrieval (10 points)
# =============================================================================

def boolean_and(index: dict[str, list[str]], terms: list[str]) -> list[str]:
    """
    Implement AND retrieval.

    Parameters
    ----------
    index : dict[str, list[str]]
        The inverted index.
    terms : list[str]
        The list of query terms.

    Returns
    -------
    list[str]
        Sorted list of document IDs that contain all of the query terms.

    Example
    -------
    >>> idx = build_inverted_index(CORPUS)
    >>> boolean_and(idx, ["Kyoto", "foliage"])
    ['d01', 'd08']
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 2")


def boolean_or(index: dict[str, list[str]], terms: list[str]) -> list[str]:
    """
    Implement OR retrieval.

    Returns
    -------
    list[str]
        Sorted list of document IDs that contain any of the query terms.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 2")


# =============================================================================
# Task 3: TF-IDF and cosine similarity (15 points)
# =============================================================================

def compute_tf(term: str, doc_tokens: list[str]) -> float:
    """
    Compute the logarithmic TF.

    tf_weight = 1 + log10(tf)  if tf > 0
                0              otherwise

    Parameters
    ----------
    term : str
        The term.
    doc_tokens : list[str]
        The token list of the document.

    Returns
    -------
    float
        The logarithmic TF value.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 3")


def compute_idf(term: str, corpus: dict[str, str]) -> float:
    """
    Compute the IDF.

    idf = log10(N / df)

    Parameters
    ----------
    term : str
        The term.
    corpus : dict[str, str]
        The corpus.

    Returns
    -------
    float
        The IDF value.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 3")


def tfidf_cosine_search(
    query: str, corpus: dict[str, str]
) -> list[tuple[str, float]]:
    """
    Retrieve with TF-IDF + cosine similarity and return a ranking in
    descending score order.

    Parameters
    ----------
    query : str
        The query string.
    corpus : dict[str, str]
        The corpus.

    Returns
    -------
    list[tuple[str, float]]
        A list of (document ID, cosine similarity score) in descending
        score order. Documents with a score of 0 are excluded.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 3")


# =============================================================================
# Task 4: Implement BM25 (15 points)
# =============================================================================

def bm25_search(
    query: str,
    corpus: dict[str, str],
    k1: float = 1.2,
    b: float = 0.75,
) -> list[tuple[str, float]]:
    """
    Retrieve with BM25 scoring and return a ranking in descending score order.

    BM25(q, d) = Σ_{t∈q∩d} IDF(t) × tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·ld/L))

    IDF(t) = log10((N - df(t) + 0.5) / (df(t) + 0.5))

    Parameters
    ----------
    query : str
        The query string.
    corpus : dict[str, str]
        The corpus.
    k1 : float
        TF saturation parameter (default: 1.2).
    b : float
        Document-length normalization parameter (default: 0.75).

    Returns
    -------
    list[tuple[str, float]]
        A list of (document ID, BM25 score) in descending score order.
        Documents with a score of 0 are excluded.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 4")


# =============================================================================
# Task 5: Compute evaluation metrics (15 points)
# =============================================================================

def precision_at_k(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    """
    Compute Precision@k.

    Parameters
    ----------
    ranking : list[str]
        The ranking of document IDs (from the top).
    qrels : dict[str, int]
        A dictionary mapping document ID -> relevance (>= 1 means relevant).
    k : int
        Evaluate the top-k results.

    Returns
    -------
    float
        Precision@k.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 5")


def average_precision(ranking: list[str], qrels: dict[str, int]) -> float:
    """
    Compute Average Precision (AP).

    Documents with relevance >= 1 are treated as relevant.
    Relevant documents not present in the ranking are still counted in the
    denominator.

    Parameters
    ----------
    ranking : list[str]
        The ranking of document IDs (from the top).
    qrels : dict[str, int]
        A dictionary mapping document ID -> relevance.

    Returns
    -------
    float
        The AP value.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 5")


def ndcg_at_k(ranking: list[str], qrels: dict[str, int], k: int) -> float:
    """
    Compute nDCG@k.

    DCG@k = Σ_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
    nDCG@k = DCG@k / IDCG@k

    Parameters
    ----------
    ranking : list[str]
        The ranking of document IDs (from the top).
    qrels : dict[str, int]
        A dictionary mapping document ID -> relevance.
    k : int
        Evaluate the top-k results.

    Returns
    -------
    float
        The nDCG@k value.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 5")


# =============================================================================
# Advanced tasks: analysis and discussion (10 points each, 30 points total)
# =============================================================================

"""
For each question below, experiment using the implementations above and write
down your results and discussion.

Q1 (10 points): Compare the TF-IDF and BM25 rankings for the query
    "Kyoto foliage". If any documents are ranked differently, explain why in
    terms of TF saturation and document-length normalization.

Q2 (10 points): For the query "Kyoto foliage", evaluate BOTH the TF-IDF
    ranking and the BM25 ranking with P@3, AP, and nDCG@5 (use QRELS["q1"],
    whose relevance grades are 0/1/2). Discuss (a) why P@3 may fail to tell
    the two systems apart while AP and nDCG can, and (b) how binary-relevance
    metrics (P@k, AP) differ from a graded-relevance metric (nDCG).

Q3 (10 points): As you vary the BM25 parameter k1 over 0, 0.5, 1.2, 5.0,
    100.0, how do the ranking and the per-document scores change for the query
    "Kyoto sightseeing"? Confirm that k1=0 degenerates to BIM (the binary
    independence model).
"""


# =============================================================================
# Tests and execution
# =============================================================================

def run_tests():
    """Run the tests for all tasks."""
    print("=" * 60)
    print("Introduction to Information Retrieval — Lecture 6 Exercise")
    print("=" * 60)

    errors = 0

    # --- Task 1 ---
    print("\n[Task 1] Build an inverted index")
    try:
        idx = build_inverted_index(CORPUS)
        assert isinstance(idx, dict), "the return value must be a dict"
        assert "Kyoto" in idx, "'Kyoto' must exist in the index"
        assert sorted(idx["Kyoto"]) == ["d01", "d03", "d06", "d08"], \
            f"posting list for 'Kyoto': expected=['d01','d03','d06','d08'], actual={sorted(idx['Kyoto'])}"
        assert sorted(idx["foliage"]) == ["d01", "d04", "d07", "d08"], \
            f"posting list for 'foliage': expected=['d01','d04','d07','d08'], actual={sorted(idx['foliage'])}"
        n_terms = len(idx)
        print(f"  OK  inverted index built (vocabulary size: {n_terms} terms)")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 2 ---
    print("\n[Task 2] Boolean retrieval")
    try:
        idx = build_inverted_index(CORPUS)
        and_result = boolean_and(idx, ["Kyoto", "foliage"])
        assert and_result == ["d01", "d08"], \
            f"AND(Kyoto, foliage): expected=['d01','d08'], actual={and_result}"
        or_result = boolean_or(idx, ["Kyoto", "foliage"])
        assert sorted(or_result) == ["d01", "d03", "d04", "d06", "d07", "d08"], \
            f"OR(Kyoto, foliage): expected=['d01','d03','d04','d06','d07','d08'], actual={sorted(or_result)}"
        and3 = boolean_and(idx, ["Kyoto", "autumn", "foliage"])
        assert and3 == ["d01", "d08"], \
            f"AND(Kyoto, autumn, foliage): expected=['d01','d08'], actual={and3}"
        print("  OK  AND/OR retrieval works")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 3 ---
    print("\n[Task 3] TF-IDF + cosine similarity")
    try:
        tf_val = compute_tf("Kyoto", tokenize(CORPUS["d01"]))
        assert abs(tf_val - 1.0) < 0.01, \
            f"tf('Kyoto', d1): expected=1.0, actual={tf_val}"
        idf_val = compute_idf("Kyoto", CORPUS)
        expected_idf = math.log10(15 / 4)
        assert abs(idf_val - expected_idf) < 0.01, \
            f"idf('Kyoto'): expected={expected_idf:.4f}, actual={idf_val:.4f}"

        results = tfidf_cosine_search("Kyoto foliage", CORPUS)
        assert len(results) > 0, "results are empty"
        assert results[0][0] in ["d01", "d08"], \
            f"TF-IDF top-1: {results[0][0]} (expected d01 or d08)"
        print(f"  OK  TF-IDF retrieval works (top 3: {[r[0] for r in results[:3]]})")
        for doc_id, score in results[:5]:
            print(f"    {doc_id}: {score:.4f}")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 4 ---
    print("\n[Task 4] BM25")
    try:
        results = bm25_search("Kyoto foliage", CORPUS)
        assert len(results) > 0, "results are empty"
        top3 = [r[0] for r in results[:3]]
        assert "d01" in top3 or "d08" in top3, f"d01 or d08 should be in the BM25 top 3: {top3}"
        print(f"  OK  BM25 retrieval works (top 3: {top3})")
        for doc_id, score in results[:5]:
            print(f"    {doc_id}: {score:.4f}")

        # Confirm degeneration to the binary model at k1=0
        results_k0 = bm25_search("Kyoto foliage", CORPUS, k1=0.0)
        scores_k0 = {r[0]: r[1] for r in results_k0}
        # At k1=0 the score does not depend on TF, so documents with the same
        # term set receive the same score.
        print(f"  OK  k1=0 test: top={[r[0] for r in results_k0[:3]]}")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 5 ---
    print("\n[Task 5] Evaluation metrics")
    try:
        # Test ranking
        ranking = ["d01", "d07", "d03", "d04", "d08"]
        qrels = QRELS["q1"]

        p3 = precision_at_k(ranking, qrels, 3)
        # d01: rel=2 (relevant), d07: rel=1 (relevant), d03: rel=0 (non-relevant) -> P@3 = 2/3
        assert abs(p3 - 2/3) < 0.01, f"P@3: expected={2/3:.4f}, actual={p3:.4f}"

        ap = average_precision(ranking, qrels)
        # Relevant docs: d01(rank1,rel=2), d07(rank2,rel=1), d04(rank4,rel=1), d08(rank5,rel=2) -> 4 relevant
        # P@1=1/1, P@2=2/2, P@4=3/4, P@5=4/5
        # AP = (1 + 1 + 3/4 + 4/5) / 4 = 0.8875
        assert abs(ap - 0.8875) < 0.01, f"AP: expected=0.8875, actual={ap:.4f}"

        ndcg = ndcg_at_k(ranking, qrels, 5)
        assert 0 <= ndcg <= 1, f"nDCG@5: {ndcg:.4f} (must be in [0, 1])"

        print(f"  OK  P@3={p3:.4f}, AP={ap:.4f}, nDCG@5={ndcg:.4f}")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Result ---
    print("\n" + "=" * 60)
    if errors == 0:
        print("All task tests passed!")
    else:
        print(f"{errors} task(s) still incomplete.")
    print("=" * 60)

    # Guide to the advanced tasks
    if errors == 0:
        print("\n[Advanced tasks]")
        print("Work on questions Q1-Q3 in the 'Advanced tasks' section at the")
        print("end of exercise.py. Use the functions you implemented to run")
        print("experiments and write down your results and discussion.")


if __name__ == "__main__":
    run_tests()
