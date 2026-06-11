"""
Information Retrieval — Lecture 12 Exercise (covers Lectures 7-10)
==================================================================

Topics: Retrieve-and-Rerank, long document ranking (Lecture 7),
pairwise / listwise reranking (Lecture 8), query and document expansion
(Lecture 9), and bi-encoders / dense retrieval (Lecture 10).

Runs in a low-resource environment (no GPU, a few GB of memory).
Uses the standard library only (no `pip install` required).

Real neural models (BERT, T5, LLMs) cannot run here, so they are
*simulated* with small hand-crafted word vectors (`WORD_VECTORS`) and a
provided `cross_encoder_score` function. The algorithms you implement
around them are exactly the ones from the lectures.

How to run:
    python lecture12_exercise.py

Implement each task's function and make the tests pass.
"""

import math

# =============================================================================
# Corpus (a small English document collection, pre-tokenized)
# =============================================================================

CORPUS = {
    "d01": "Kyoto autumn foliage is spectacular . Arashiyama and Tofukuji are famous foliage spots .",
    "d02": "In the old capital the maple leaves turn red in fall . Every temple garden glows at dusk .",
    "d03": "Tokyo cherry blossoms bloom in spring . Ueno Park and Meguro River attract many visitors .",
    "d04": "This car achieves excellent fuel efficiency . A hybrid engine lowers fuel cost on long drives .",
    "d05": "The new automobile reduces gasoline consumption and improves mileage . A lighter body also helps .",
    "d06": "Osaka street food is fun . Dotonbori is famous for takoyaki and neon lights .",
    "d07": "A search engine ranks documents with BM25 . An inverted index makes retrieval fast .",
    "d08": "Neural rankers use BERT . A cross encoder reads the query and the document together .",
    "d09": "Okinawa beaches have clear water . Diving in the ocean is popular in summer .",
    # d10 is a "long document": only one part of it talks about foliage
    "d10": ("Kyoto is a city with many faces . Spring brings cherry blossoms to the Philosopher Path . "
            "Summer festivals fill the streets with music . In autumn the foliage of Arashiyama turns brilliant red . "
            "Winter snow covers the golden pavilion . Every season rewards a visit ."),
    # d11 is a "decoy": shares query terms with q1 but is off-topic (fashion)
    "d11": "Fake plastic foliage is on sale . The autumn foliage garlands and autumn wreaths brighten the shop windows .",
}

QUERIES = {
    "q1": "Kyoto autumn foliage",
    "q2": "car fuel efficiency",
    "q3": "search engine",
}

# Note the vocabulary mismatch built into the corpus:
#   q1 "Kyoto autumn foliage"  <-> d02 "old capital ... maple leaves ... fall"
#   q2 "car fuel efficiency"   <-> d05 "automobile ... gasoline consumption ... mileage"
# BM25 gives d02 and d05 a score of 0 for these queries.

# doc2query output (simulated): queries generated from each document
GENERATED_QUERIES = {
    "d02": ["Kyoto autumn foliage temples", "best fall colors in Kyoto"],
    "d05": ["car fuel efficiency tips", "how to improve car fuel economy"],
    "d09": ["Okinawa summer diving spots", "best beaches in Japan"],
}

# DeepCT/DeepImpact output (simulated): contextual term importance in [0, 1]
TERM_WEIGHTS = {
    "d07": {"search": 0.92, "engine": 0.85, "BM25": 0.70, "index": 0.65,
            "inverted": 0.55, "retrieval": 0.60, "documents": 0.30, "ranks": 0.25,
            "fast": 0.10, "makes": 0.0, "with": 0.0, "A": 0.0, "An": 0.0},
    "d08": {"BERT": 0.88, "encoder": 0.75, "cross": 0.70, "query": 0.55,
            "document": 0.50, "rankers": 0.45, "Neural": 0.40, "reads": 0.05,
            "use": 0.0, "the": 0.0, "and": 0.0, "together": 0.0, "A": 0.0},
}

# Hand-crafted 8-dimensional word vectors (a tiny stand-in for BERT
# embeddings). Synonyms point in nearly the same direction:
#   car ~ automobile, fuel ~ gasoline, autumn ~ fall, foliage ~ leaves, ...
# Keys are lowercase; look tokens up with `token_vector` below.
WORD_VECTORS = {
    # places
    "kyoto":      (1.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
    "capital":    (0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "tokyo":      (0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "osaka":      (0.2, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "okinawa":    (0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8),
    "temple":     (0.5, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
    "arashiyama": (0.9, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0),
    "tofukuji":   (0.9, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0),
    # autumn / foliage
    "autumn":     (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "fall":       (0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0),
    "foliage":    (0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0),
    "leaves":     (0.0, 0.0, 0.9, 0.2, 0.0, 0.0, 0.0, 0.0),
    "maple":      (0.0, 0.0, 0.85, 0.2, 0.0, 0.0, 0.0, 0.0),
    "red":        (0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0),
    # spring
    "spring":     (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "cherry":     (0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
    "blossoms":   (0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0),
    # vehicles / fuel
    "car":        (0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0),
    "automobile": (0.0, 0.0, 0.0, 0.0, 0.95, 0.1, 0.0, 0.0),
    "fuel":       (0.0, 0.0, 0.0, 0.0, 0.2, 1.0, 0.0, 0.0),
    "gasoline":   (0.0, 0.0, 0.0, 0.0, 0.2, 0.95, 0.0, 0.0),
    "efficiency": (0.0, 0.0, 0.0, 0.0, 0.3, 0.8, 0.1, 0.0),
    "mileage":    (0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 0.0, 0.0),
    "consumption": (0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0),
    "hybrid":     (0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.1, 0.0),
    "engine":     (0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.85, 0.0),
    # search tech
    "search":     (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "index":      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0),
    "bm25":       (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0),
    "retrieval":  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.0),
    "documents":  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0),
    "document":   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0),
    "bert":       (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0),
    "encoder":    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.0),
    "rankers":    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0),
    "query":      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.0),
    # sea / summer
    "summer":     (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "ocean":      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95),
    "beaches":    (0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9),
    "diving":     (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85),
}

VECTOR_DIM = 8


def tokenize(text: str) -> list[str]:
    """Tokenize on whitespace, dropping punctuation tokens."""
    return [t for t in text.split() if t not in {".", ",", ";", ":", "。", "、", "，", "．"}]


def token_vector(token: str) -> tuple[float, ...] | None:
    """Return the word vector for a token (case-insensitive), or None."""
    return WORD_VECTORS.get(token.lower())


def cosine(u: tuple[float, ...], v: tuple[float, ...]) -> float:
    """Cosine similarity between two vectors (0.0 if either is a zero vector)."""
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


# =============================================================================
# Provided: first-stage retrieval and simulated neural scorers
# =============================================================================

def bm25_search(
    query: str,
    corpus: dict[str, str],
    k1: float = 1.2,
    b: float = 0.75,
) -> list[tuple[str, float]]:
    """
    BM25 retrieval (you implemented this in the Lecture 6 exercise;
    here it is provided as Stage-1 infrastructure).

    Returns a list of (document ID, BM25 score) in descending score
    order (ties broken by ascending document ID). Documents with a
    score of 0 are excluded.
    """
    q_terms = tokenize(query)
    doc_tokens = {d: tokenize(t) for d, t in corpus.items()}
    n_docs = len(corpus)
    avg_len = sum(len(t) for t in doc_tokens.values()) / n_docs
    df = {t: sum(1 for toks in doc_tokens.values() if t in toks) for t in set(q_terms)}
    results = []
    for doc_id, toks in doc_tokens.items():
        score = 0.0
        for t in q_terms:
            tf = toks.count(t)
            if tf == 0 or df[t] == 0:
                continue
            idf = math.log10((n_docs - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(toks) / avg_len))
        if score > 0:
            results.append((doc_id, score))
    results.sort(key=lambda x: (-x[1], x[0]))
    return results


def cross_encoder_score(query: str, doc_text: str) -> float:
    """
    Simulated cross-encoder (stand-in for monoBERT / monoT5).

    Combines soft semantic matching over WORD_VECTORS with an exact-match
    bonus. Treat it as a black box: in the lectures this would be one
    BERT/T5 forward pass over "[CLS] q [SEP] d [SEP]".
    Score range is roughly [0, 1]; higher means more relevant.
    """
    q_toks = [t for t in tokenize(query) if token_vector(t) is not None]
    d_toks = [t for t in tokenize(doc_text) if token_vector(t) is not None]
    if not q_toks or not d_toks:
        return 0.0
    sem = sum(
        max(cosine(token_vector(q), token_vector(d)) for d in d_toks)
        for q in q_toks
    ) / len(q_toks)
    q_set = {t.lower() for t in tokenize(query)}
    d_set = {t.lower() for t in tokenize(doc_text)}
    exact = len(q_set & d_set) / len(q_set)
    return round(0.8 * sem + 0.2 * exact, 6)


def pairwise_pref(query: str, doc_i_text: str, doc_j_text: str) -> float:
    """
    Simulated pairwise model (stand-in for duoT5).

    Returns P(d_i is more relevant than d_j | query) in [0, 1].
    A value > 0.5 means the model prefers d_i over d_j.
    """
    s_i = cross_encoder_score(query, doc_i_text)
    s_j = cross_encoder_score(query, doc_j_text)
    return 1.0 / (1.0 + math.exp(-10.0 * (s_i - s_j)))


def rank_window(query: str, doc_ids: list[str], corpus: dict[str, str]) -> list[str]:
    """
    Simulated listwise model (stand-in for one RankGPT window call).

    Takes the documents in one window and returns their IDs reordered by
    decreasing relevance (stable for ties).
    """
    return sorted(doc_ids, key=lambda d: -cross_encoder_score(query, corpus[d]))


# =============================================================================
# Task 1: Passage splitting and score aggregation (10 points) — Lecture 7
# =============================================================================

def split_passages(tokens: list[str], window_size: int, stride: int) -> list[list[str]]:
    """
    Split a token list into overlapping passages with a sliding window
    (as in BERT-MaxP, which uses a 150-word window with stride 75).

    Passages start at offsets 0, stride, 2*stride, ... and each passage
    is `tokens[start:start + window_size]`. Stop after the first passage
    that reaches the end of the document (i.e., whose
    `start + window_size >= len(tokens)`); the last passage may be
    shorter than `window_size`.

    Parameters
    ----------
    tokens : list[str]
        The token list of the document.
    window_size : int
        Number of tokens per passage.
    stride : int
        Offset between the start positions of consecutive passages.

    Returns
    -------
    list[list[str]]
        The list of passages (each a list of tokens), in order.

    Example
    -------
    >>> split_passages(["A", "B", "C", "D", "E", "F"], 4, 2)
    [['A', 'B', 'C', 'D'], ['C', 'D', 'E', 'F']]
    >>> split_passages(["A", "B"], 4, 2)
    [['A', 'B']]
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 1")


def aggregate_scores(scores: list[float], method: str) -> float:
    """
    Aggregate per-passage scores into one document score (Lecture 7).

    method = "max"   -> MaxP   (score of the best passage)
    method = "first" -> FirstP (score of the first passage)
    method = "sum"   -> SumP   (sum of all passage scores)

    Parameters
    ----------
    scores : list[float]
        Per-passage scores, in passage order (non-empty).
    method : str
        One of "max", "first", "sum".

    Returns
    -------
    float
        The aggregated document score.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 1")


# =============================================================================
# Task 2: Retrieve-and-Rerank (15 points) — Lecture 7
# =============================================================================

def retrieve_and_rerank(query: str, corpus: dict[str, str], k: int) -> list[tuple[str, float]]:
    """
    Two-stage ranking (Lecture 7).

    Stage 1: Retrieve the top-k candidates with `bm25_search`.
    Stage 2: Re-score ONLY those candidates with `cross_encoder_score`
             and sort by the new score.

    Documents that BM25 did not retrieve must never appear in the
    output, no matter how high the cross-encoder would score them
    (this is the "recall ceiling" of reranking).

    Parameters
    ----------
    query : str
        The query string.
    corpus : dict[str, str]
        The corpus.
    k : int
        The number of Stage-1 candidates to rerank.

    Returns
    -------
    list[tuple[str, float]]
        A list of (document ID, cross-encoder score) in descending
        score order. Ties are broken by ascending document ID.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 2")


# =============================================================================
# Task 3: Pairwise and listwise reranking (15 points) — Lecture 8
# =============================================================================

def pairwise_rerank(query: str, doc_ids: list[str], corpus: dict[str, str]) -> list[str]:
    """
    duoT5-style pairwise reranking with win counting (Lecture 8).

    For every ordered pair (d_i, d_j) with i != j, call
    `pairwise_pref(query, corpus[d_i], corpus[d_j])`; d_i "wins" the
    comparison if the preference is > 0.5. Each document's score is its
    number of wins:

        s(d_i) = sum over j != i of 1[ P(d_i > d_j | q) > 0.5 ]

    Parameters
    ----------
    query : str
        The query string.
    doc_ids : list[str]
        The candidate document IDs (e.g., the top results of Stage 1).
    corpus : dict[str, str]
        The corpus.

    Returns
    -------
    list[str]
        The document IDs sorted by number of wins in descending order.
        For ties, keep the original input order (stable sort).
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 3")


def sliding_window_rerank(
    query: str,
    ranking: list[str],
    corpus: dict[str, str],
    window_size: int,
    stride: int,
) -> list[str]:
    """
    RankGPT-style listwise reranking with a sliding window (Lecture 8).

    Because the listwise model can only see `window_size` documents at a
    time, process the ranking from the BOTTOM to the TOP:

    1. If len(ranking) <= window_size, reorder the whole list with one
       `rank_window` call and return it.
    2. Otherwise, set start = len(ranking) - window_size. Reorder the
       slice ranking[start:start + window_size] with `rank_window`
       (in place in a copy of the list).
    3. Move the window up: start = max(0, start - stride), reorder that
       slice, and repeat. The final window always starts at 0.

    Overlapping windows let highly relevant documents near the bottom
    "bubble up" toward the top, like one pass of bubble sort.

    Parameters
    ----------
    query : str
        The query string.
    ranking : list[str]
        The initial ranking of document IDs (do not modify it in place).
    corpus : dict[str, str]
        The corpus.
    window_size : int
        Number of documents the listwise model sees at once.
    stride : int
        How far the window moves up each step (stride < window_size).

    Returns
    -------
    list[str]
        The final ranking of document IDs.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 3")


# =============================================================================
# Task 4: Document expansion and impact indexing (15 points) — Lecture 9
# =============================================================================

def expand_corpus(corpus: dict[str, str], generated_queries: dict[str, list[str]]) -> dict[str, str]:
    """
    doc2query document expansion (Lecture 9).

    For each document that has generated queries, append all of them to
    the document text:

        expanded_text = original_text + " " + " ".join(queries)

    Documents without generated queries are kept unchanged. The input
    corpus must not be modified.

    Parameters
    ----------
    corpus : dict[str, str]
        The corpus.
    generated_queries : dict[str, list[str]]
        A dictionary mapping document ID -> list of generated query
        strings (e.g., GENERATED_QUERIES).

    Returns
    -------
    dict[str, str]
        A new corpus with expanded document texts.

    Example
    -------
    >>> exp = expand_corpus({"d1": "A B ."}, {"d1": ["C D", "E"]})
    >>> exp["d1"]
    'A B . C D E'
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 4")


def build_impact_index(
    term_weights: dict[str, dict[str, float]],
    scale: int = 100,
) -> dict[str, dict[str, int]]:
    """
    Quantize DeepCT/DeepImpact term weights into an impact index
    (Lecture 9).

    Each real-valued weight w in [0, 1] is quantized to an integer
    impact = round(w * scale). Terms with impact <= 0 are NOT stored
    (just like terms with TF = 0 are absent from an inverted index).

    Parameters
    ----------
    term_weights : dict[str, dict[str, float]]
        A dictionary mapping document ID -> {term: weight}
        (e.g., TERM_WEIGHTS).
    scale : int
        The quantization scaling factor (default: 100).

    Returns
    -------
    dict[str, dict[str, int]]
        The impact index: term -> {document ID: integer impact}.

    Example
    -------
    >>> idx = build_impact_index({"d1": {"apple": 0.9, "the": 0.0}}, scale=100)
    >>> idx["apple"]
    {'d1': 90}
    >>> "the" in idx
    False
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 4")


def impact_search(query: str, index: dict[str, dict[str, int]]) -> list[tuple[str, int]]:
    """
    Retrieve with the impact index (Lecture 9).

    The score of a document is the sum of the impacts of the query
    terms it contains (query terms are obtained with `tokenize`; terms
    absent from the index contribute nothing):

        score(q, d) = sum over t in q of impact(t, d)

    Parameters
    ----------
    query : str
        The query string.
    index : dict[str, dict[str, int]]
        The impact index built by `build_impact_index`.

    Returns
    -------
    list[tuple[str, int]]
        A list of (document ID, score) in descending score order
        (ties broken by ascending document ID). Documents with no
        matching term are excluded.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 4")


# =============================================================================
# Task 5: Bi-encoder dense retrieval and late interaction (15 points) — Lecture 10
# =============================================================================

def encode_text(text: str) -> tuple[float, ...]:
    """
    Bi-encoder text encoding (Lecture 10).

    Encode a text into ONE vector by averaging the word vectors of its
    tokens (mean pooling). Tokens without a vector (i.e.,
    `token_vector(t)` is None) are skipped. If no token has a vector,
    return the zero vector of dimension VECTOR_DIM.

    Parameters
    ----------
    text : str
        The text (query or document).

    Returns
    -------
    tuple[float, ...]
        The mean-pooled vector (length VECTOR_DIM).
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 5")


def dense_search(query: str, corpus: dict[str, str]) -> list[tuple[str, float]]:
    """
    Single-vector dense retrieval (Lecture 10).

    Encode the query and every document with `encode_text`, then rank
    documents by cosine similarity of the vectors. Unlike reranking,
    this scores the WHOLE corpus — in a real system the document
    vectors are pre-computed and searched with ANN indexes.

    Parameters
    ----------
    query : str
        The query string.
    corpus : dict[str, str]
        The corpus.

    Returns
    -------
    list[tuple[str, float]]
        A list of (document ID, cosine similarity) in descending score
        order (ties broken by ascending document ID). Documents with a
        score <= 0 are excluded.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 5")


def colbert_score(query: str, doc_text: str) -> float:
    """
    ColBERT-style late interaction (MaxSim) scoring (Lecture 10).

    Instead of pooling everything into one vector, keep one vector per
    token. For each query token, find the most similar document token,
    then sum these maxima:

        score(q, d) = sum over q_i in q of  max over d_j in d of
                      cos(v(q_i), v(d_j))

    Only tokens that have a word vector participate. If the query or
    the document has no such token, return 0.0.

    Parameters
    ----------
    query : str
        The query string.
    doc_text : str
        The document text.

    Returns
    -------
    float
        The late-interaction score.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 5")


# =============================================================================
# Advanced tasks: analysis and discussion (10 points each, 30 points total)
# =============================================================================

"""
For each question below, experiment using the implementations above and write
down your results and discussion.

Q1 (10 points): The recall ceiling of reranking. For the query
    "car fuel efficiency", run `retrieve_and_rerank` with k = 5 and check
    whether d05 (the "automobile ... gasoline ... mileage" document) appears.
    Then build an expanded corpus with
    `expand_corpus(CORPUS, GENERATED_QUERIES)` and run the same two-stage
    pipeline on it. Explain (a) why reranking alone can never retrieve d05,
    and (b) how document expansion changes Stage-1 recall. Relate your answer
    to the statement "BM25's Recall determines the performance ceiling of
    reranking" from Lecture 7.

Q2 (10 points): Long document aggregation. For the long document d10 and the
    query "Kyoto autumn foliage", split d10 into passages with
    `split_passages(tokenize(CORPUS["d10"]), 16, 8)`, score each passage with
    `cross_encoder_score(query, " ".join(passage))`, and compare the document
    scores given by MaxP, FirstP, and SumP (`aggregate_scores`). Which passage
    wins under MaxP, and why does FirstP underestimate this document? Discuss
    when SumP is biased by document length, and relate your findings to the
    Lecture 7 insight that "a document's relevance is determined by its most
    relevant portion".

Q3 (10 points): Efficiency vs effectiveness of reranking modes. Suppose
    Stage 1 returns n = 100 candidates. Compute the number of model calls
    needed by (a) pointwise reranking (monoT5), (b) all-pairs pairwise
    reranking (duoT5), and (c) sliding-window listwise reranking with
    window 20 and stride 10 (RankGPT). Then, on this exercise's corpus,
    compare the rankings produced by `retrieve_and_rerank`,
    `pairwise_rerank`, and `sliding_window_rerank` for the query
    "Kyoto autumn foliage" (use the BM25 top results as input). Do the three
    modes agree? Discuss when paying the extra cost of pairwise/listwise is
    worthwhile, and why multi-stage pipelines (Lecture 8) apply the expensive
    methods only to a few top candidates.
"""


# =============================================================================
# Tests and execution
# =============================================================================

def run_tests():
    """Run the tests for all tasks."""
    print("=" * 60)
    print("Information Retrieval — Lecture 12 Exercise (Lectures 7-10)")
    print("=" * 60)

    errors = 0

    # --- Task 1 ---
    print("\n[Task 1] Passage splitting and score aggregation")
    try:
        toks = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        ps = split_passages(toks, 4, 2)
        expected = [["A", "B", "C", "D"], ["C", "D", "E", "F"],
                    ["E", "F", "G", "H"], ["G", "H", "I", "J"]]
        assert ps == expected, f"split_passages(10 tokens, w=4, s=2): expected={expected}, actual={ps}"
        ps_short = split_passages(["A", "B"], 4, 2)
        assert ps_short == [["A", "B"]], \
            f"split_passages(2 tokens, w=4, s=2): expected=[['A','B']], actual={ps_short}"
        d10_ps = split_passages(tokenize(CORPUS["d10"]), 16, 8)
        assert [len(p) for p in d10_ps] == [16, 16, 16, 16, 10], \
            f"d10 passage lengths: expected=[16,16,16,16,10], actual={[len(p) for p in d10_ps]}"

        assert aggregate_scores([0.2, 0.9, 0.5], "max") == 0.9, "MaxP failed"
        assert aggregate_scores([0.2, 0.9, 0.5], "first") == 0.2, "FirstP failed"
        assert abs(aggregate_scores([0.2, 0.9, 0.5], "sum") - 1.6) < 1e-9, "SumP failed"
        print(f"  OK  splitting works (d10: {len(d10_ps)} passages) and MaxP/FirstP/SumP work")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 2 ---
    print("\n[Task 2] Retrieve-and-Rerank")
    try:
        # Stage 1 (BM25) ranks the off-topic decoy d11 above the long
        # relevant document d10; the cross-encoder fixes the order.
        bm25_ids = [d for d, _ in bm25_search(QUERIES["q1"], CORPUS)]
        assert bm25_ids == ["d01", "d11", "d10"], \
            f"BM25(q1): expected=['d01','d11','d10'], actual={bm25_ids} (provided function — corpus changed?)"

        rr = retrieve_and_rerank(QUERIES["q1"], CORPUS, 3)
        rr_ids = [d for d, _ in rr]
        assert rr_ids == ["d01", "d10", "d11"], \
            f"rerank(q1, k=3): expected=['d01','d10','d11'], actual={rr_ids}"

        # Recall ceiling 1: with k=2, d10 is cut at Stage 1 and cannot return
        rr2 = retrieve_and_rerank(QUERIES["q1"], CORPUS, 2)
        assert [d for d, _ in rr2] == ["d01", "d11"], \
            f"rerank(q1, k=2): expected=['d01','d11'], actual={[d for d, _ in rr2]}"

        # Recall ceiling 2: d05 has BM25 score 0 for q2, so reranking
        # can never bring it back, however large k is
        rr_q2 = retrieve_and_rerank(QUERIES["q2"], CORPUS, 5)
        assert [d for d, _ in rr_q2] == ["d04"], \
            f"rerank(q2, k=5): expected=['d04'] only, actual={[d for d, _ in rr_q2]}"
        print(f"  OK  reranking works (q1: BM25 {bm25_ids} -> CE {rr_ids})")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 3 ---
    print("\n[Task 3] Pairwise and listwise reranking")
    try:
        pw = pairwise_rerank(QUERIES["q1"], ["d03", "d11", "d02", "d01"], CORPUS)
        assert pw == ["d01", "d02", "d11", "d03"], \
            f"pairwise(q1): expected=['d01','d02','d11','d03'], actual={pw}"

        init = ["d07", "d08", "d03", "d06", "d05", "d04"]
        sw = sliding_window_rerank(QUERIES["q2"], init, CORPUS, 3, 2)
        assert sw[0] == "d04", \
            f"sliding window: the best document d04 should bubble up to the top, actual top={sw[0]}"
        assert sw == ["d04", "d07", "d08", "d03", "d05", "d06"], \
            f"sliding window(q2, w=3, s=2): expected=['d04','d07','d08','d03','d05','d06'], actual={sw}"
        assert init == ["d07", "d08", "d03", "d06", "d05", "d04"], \
            "sliding_window_rerank must not modify the input list"

        sw_small = sliding_window_rerank(QUERIES["q2"], ["d05", "d04"], CORPUS, 3, 2)
        assert sw_small == ["d04", "d05"], \
            f"sliding window (n <= w): expected=['d04','d05'], actual={sw_small}"
        print(f"  OK  pairwise {pw} / sliding window {sw}")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 4 ---
    print("\n[Task 4] Document expansion and impact indexing")
    try:
        expanded = expand_corpus(CORPUS, GENERATED_QUERIES)
        assert expanded["d01"] == CORPUS["d01"], "documents without generated queries must be unchanged"
        assert expanded["d02"].startswith(CORPUS["d02"]), "expansion must append, not replace"
        assert "Kyoto" in expanded["d02"], "generated query terms must be appended to d02"
        assert "Kyoto" not in CORPUS["d02"], "the input corpus must not be modified"

        # Vocabulary mismatch resolved: d02 and d05 are now retrievable by BM25
        exp_q1 = [d for d, _ in bm25_search(QUERIES["q1"], expanded)]
        assert "d02" in exp_q1[:2], \
            f"BM25(q1) on expanded corpus: d02 should be in the top 2, actual={exp_q1}"
        exp_q2 = [d for d, _ in bm25_search(QUERIES["q2"], expanded)]
        assert exp_q2[:2] == ["d04", "d05"], \
            f"BM25(q2) on expanded corpus: expected top 2=['d04','d05'], actual={exp_q2[:2]}"

        idx = build_impact_index(TERM_WEIGHTS, scale=100)
        assert idx.get("search") == {"d07": 92}, \
            f"impact of 'search': expected={{'d07': 92}}, actual={idx.get('search')}"
        assert idx.get("BERT") == {"d08": 88}, \
            f"impact of 'BERT': expected={{'d08': 88}}, actual={idx.get('BERT')}"
        assert "makes" not in idx and "the" not in idx, "terms with impact 0 must not be stored"

        res = impact_search("search engine", idx)
        assert res == [("d07", 177)], \
            f"impact_search('search engine'): expected=[('d07', 177)], actual={res}"
        res2 = impact_search("cross encoder", idx)
        assert res2 == [("d08", 145)], \
            f"impact_search('cross encoder'): expected=[('d08', 145)], actual={res2}"
        print(f"  OK  doc2query expansion (q2 top 2: {exp_q2[:2]}) and impact index work")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 5 ---
    print("\n[Task 5] Dense retrieval and late interaction")
    try:
        v = encode_text("Kyoto autumn")
        assert len(v) == VECTOR_DIM, f"encode_text must return a vector of length {VECTOR_DIM}"
        assert abs(v[0] - 0.5) < 0.01 and abs(v[2] - 0.55) < 0.01, \
            f"encode_text('Kyoto autumn'): expected (0.5, ..., 0.55, ...), actual={v}"
        zero = encode_text("qwerty xyzzy")
        assert all(abs(x) < 1e-9 for x in zero), "unknown-only text must encode to the zero vector"

        # Dense retrieval resolves the vocabulary mismatch that BM25 cannot:
        # d02 shares no term with q1 but is ranked in the top 2
        bm25_q1 = [d for d, _ in bm25_search(QUERIES["q1"], CORPUS)]
        assert "d02" not in bm25_q1, "BM25 should not retrieve d02 for q1 (no shared terms)"
        dense_q1 = [d for d, _ in dense_search(QUERIES["q1"], CORPUS)]
        assert "d02" in dense_q1[:2], \
            f"dense(q1): d02 should be in the top 2, actual={dense_q1[:5]}"
        dense_q2 = [d for d, _ in dense_search(QUERIES["q2"], CORPUS)]
        assert set(dense_q2[:2]) == {"d04", "d05"}, \
            f"dense(q2): expected top 2={{'d04','d05'}}, actual={dense_q2[:5]}"

        # Late interaction: exact topical documents stay on top
        c_d04 = colbert_score(QUERIES["q2"], CORPUS["d04"])
        c_d05 = colbert_score(QUERIES["q2"], CORPUS["d05"])
        assert abs(c_d04 - 3.0) < 0.01, f"colbert(q2, d04): expected=3.0, actual={c_d04:.4f}"
        assert abs(c_d05 - 2.9877) < 0.01, f"colbert(q2, d05): expected=2.9877, actual={c_d05:.4f}"
        assert c_d04 > c_d05, "MaxSim must prefer the exact-match document d04"
        c_none = colbert_score(QUERIES["q2"], "qwerty xyzzy")
        assert c_none == 0.0, "colbert_score must be 0.0 when no document token has a vector"
        print(f"  OK  dense retrieval (q1 top 2: {dense_q1[:2]}) and MaxSim work")
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
        print("Work on questions Q1-Q3 in the 'Advanced tasks' section of this")
        print("file. Use the functions you implemented to run experiments and")
        print("write down your results and discussion.")


if __name__ == "__main__":
    run_tests()
