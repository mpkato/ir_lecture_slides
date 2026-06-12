"""
Information Retrieval — Lecture 12 Exercise (covers Lectures 7-11)
==================================================================

Topics: Retrieve-and-Rerank, long document ranking (Lecture 7),
pairwise / listwise reranking (Lecture 8), query and document expansion
(Lecture 9), bi-encoders / dense retrieval (Lecture 10), and learned
sparse retrieval (Lecture 11).

Unlike the Lecture 6 exercise, this one uses REAL pre-trained neural
models. The corpus is tiny (11 documents), so everything runs on a CPU
in seconds — no GPU needed. The models (about 870 MB in total) are
downloaded automatically from Hugging Face on the first run and cached
under `~/.cache/huggingface`:

    cross-encoder/ms-marco-MiniLM-L6-v2        (~91 MB)  cross-encoder    (Lec. 7-8)
    doc2query/msmarco-t5-small-v1              (~242 MB) doc2query        (Lec. 9)
    sentence-transformers/all-MiniLM-L6-v2     (~91 MB)  bi-encoder       (Lec. 10)
    naver/splade-cocondenser-ensembledistil    (~440 MB) SPLADE           (Lec. 11)

Setup:
    pip install -r requirements.txt

How to run:
    python lecture12_exercise.py

Implement each task's function and make the tests pass. The provided
model wrappers (`cross_encoder_score`, `generate_queries`,
`token_embeddings`, `splade_term_weights`, ...) are the building
blocks; the algorithms you implement around them are exactly the ones
from the lectures.
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
    # d10 is a "long document": it starts like a generic travel guide, and
    # only one sentence in the middle talks about autumn foliage in Kyoto
    "d10": ("This guide covers transport tickets hotels and seasonal events for visitors . "
            "Buses and trains connect every district and one day passes are cheap . "
            "In autumn the foliage of Kyoto is breathtaking and Arashiyama glows in brilliant red . "
            "Winter snow covers the golden pavilion and spring brings cherry blossoms . "
            "Every season rewards a visit ."),
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
# BM25 gives d02 and d05 a score of 0 for these queries. Whether (and how)
# each neural method fixes this mismatch is a running theme of this exercise.


def tokenize(text: str) -> list[str]:
    """Tokenize on whitespace, dropping punctuation tokens."""
    return [t for t in text.split() if t not in {".", ",", ";", ":", "。", "、", "，", "．"}]


def cosine(u: tuple[float, ...], v: tuple[float, ...]) -> float:
    """Cosine similarity between two vectors (0.0 if either is a zero vector)."""
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


# =============================================================================
# Provided: first-stage retrieval
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


# =============================================================================
# Provided: real neural models (loaded lazily, cached after the first call)
# =============================================================================

_MODELS: dict = {}
_CACHE: dict = {"ce": {}, "tok_emb": {}, "splade": {}, "d2q": {}}


def _cross_encoder():
    if "ce" not in _MODELS:
        from sentence_transformers import CrossEncoder
        print("  (loading cross-encoder/ms-marco-MiniLM-L6-v2 ...)")
        _MODELS["ce"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
    return _MODELS["ce"]


def _bi_encoder():
    if "be" not in _MODELS:
        from sentence_transformers import SentenceTransformer
        print("  (loading sentence-transformers/all-MiniLM-L6-v2 ...)")
        _MODELS["be"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _MODELS["be"]


def _doc2query():
    if "d2q" not in _MODELS:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print("  (loading doc2query/msmarco-t5-small-v1 ...)")
        tok = AutoTokenizer.from_pretrained("doc2query/msmarco-t5-small-v1")
        model = AutoModelForSeq2SeqLM.from_pretrained("doc2query/msmarco-t5-small-v1")
        model.eval()
        _MODELS["d2q"] = (tok, model)
    return _MODELS["d2q"]


def _splade():
    if "splade" not in _MODELS:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        print("  (loading naver/splade-cocondenser-ensembledistil ...)")
        tok = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
        model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
        model.eval()
        _MODELS["splade"] = (tok, model)
    return _MODELS["splade"]


def cross_encoder_score(query: str, doc_text: str) -> float:
    """
    Real cross-encoder relevance scoring (monoBERT-style, Lectures 7-8).

    One transformer forward pass over "[CLS] query [SEP] document [SEP]"
    using cross-encoder/ms-marco-MiniLM-L6-v2 (a 6-layer MiniLM trained
    on MS MARCO). The raw logit is mapped to (0, 1) with a sigmoid;
    higher means more relevant.
    """
    key = (query, doc_text)
    if key not in _CACHE["ce"]:
        logit = float(_cross_encoder().predict([(query, doc_text)])[0])
        _CACHE["ce"][key] = round(1.0 / (1.0 + math.exp(-logit)), 6)
    return _CACHE["ce"][key]


def pairwise_pref(query: str, doc_i_text: str, doc_j_text: str) -> float:
    """
    duoT5-style pairwise preference (Lecture 8).

    Returns P(d_i is more relevant than d_j | query) in (0, 1).
    A value > 0.5 means the model prefers d_i over d_j.

    The real duoT5 is a fine-tuned T5 that reads both documents at once;
    to keep the download small, the preference here is derived from two
    real cross-encoder scores: sigmoid(10 * (s_i - s_j)).
    """
    s_i = cross_encoder_score(query, doc_i_text)
    s_j = cross_encoder_score(query, doc_j_text)
    return 1.0 / (1.0 + math.exp(-10.0 * (s_i - s_j)))


def rank_window(query: str, doc_ids: list[str], corpus: dict[str, str]) -> list[str]:
    """
    Listwise window reordering (stand-in for one RankGPT window call,
    Lecture 8).

    Takes the documents in one window and returns their IDs reordered by
    decreasing relevance (stable for ties). RankGPT prompts an LLM with
    the whole window; here the window is reordered with the real
    cross-encoder, which is equivalent for the purposes of the
    sliding-window algorithm you implement around it.
    """
    return sorted(doc_ids, key=lambda d: -cross_encoder_score(query, corpus[d]))


def generate_queries(doc_text: str, n: int = 3) -> list[str]:
    """
    Real doc2query / docTTTTTquery query generation (Lecture 9).

    Feeds the document to doc2query/msmarco-t5-small-v1 (a T5 fine-tuned
    on MS MARCO to map a passage to a query that the passage answers)
    and returns `n` generated queries. Deterministic beam search is used
    so that everyone gets the same output.
    """
    key = (doc_text, n)
    if key not in _CACHE["d2q"]:
        import torch
        tok, model = _doc2query()
        ids = tok(doc_text, return_tensors="pt", truncation=True, max_length=384).input_ids
        with torch.no_grad():
            out = model.generate(ids, max_length=24, num_beams=max(4, n), num_return_sequences=n)
        _CACHE["d2q"][key] = [tok.decode(o, skip_special_tokens=True) for o in out]
    return _CACHE["d2q"][key]


def token_embeddings(text: str) -> list[tuple[str, tuple[float, ...]]]:
    """
    Real contextualized token embeddings (Lecture 10).

    Encodes the text with sentence-transformers/all-MiniLM-L6-v2 and
    returns one (token, vector) pair per WordPiece token, in order.
    Special tokens ([CLS], [SEP]) are removed. Each vector has
    VECTOR_DIM (= 384) dimensions. Note that rare words are split into
    word pieces (e.g. "foliage" -> "fol", "##iage").

    Returns an empty list if the text contains no tokens.
    """
    if text not in _CACHE["tok_emb"]:
        import torch
        model = _bi_encoder()
        features = model.tokenize([text])
        with torch.no_grad():
            out = model(features)
        vecs = out["token_embeddings"][0]
        tokens = model.tokenizer.convert_ids_to_tokens(features["input_ids"][0])
        special = {model.tokenizer.cls_token, model.tokenizer.sep_token, model.tokenizer.pad_token}
        _CACHE["tok_emb"][text] = [
            (t, tuple(v)) for t, v in zip(tokens, vecs.tolist()) if t not in special
        ]
    return _CACHE["tok_emb"][text]


VECTOR_DIM = 384


def splade_term_weights(text: str) -> dict[str, float]:
    """
    Real SPLADE document encoding (Lecture 11).

    Runs naver/splade-cocondenser-ensembledistil over the text and
    returns the sparse term-weight vector as {term: weight} with
    weight > 0, where terms are WordPiece vocabulary entries. The
    weights are max-pooled log(1 + ReLU(logits)) over the token
    positions, as in the SPLADE paper.

    Because SPLADE predicts weights over the WHOLE vocabulary, the
    result contains expansion terms that do not appear in the text
    (e.g. "car" and "fuel" for d05, which only says "automobile ...
    gasoline"). This is learned sparse retrieval's answer to the
    vocabulary mismatch problem.
    """
    if text not in _CACHE["splade"]:
        import torch
        tok, model = _splade()
        enc = tok(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**enc).logits
        weights, _ = torch.max(
            torch.log1p(torch.relu(logits)) * enc.attention_mask.unsqueeze(-1), dim=1
        )
        weights = weights.squeeze(0)
        nonzero = torch.nonzero(weights).squeeze(-1)
        _CACHE["splade"][text] = {
            tok.convert_ids_to_tokens([i])[0]: round(float(weights[i]), 4)
            for i in nonzero.tolist()
        }
    return _CACHE["splade"][text]


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
# Task 2: Retrieve-and-Rerank (10 points) — Lecture 7
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
# Task 4: Document expansion with doc2query (10 points) — Lecture 9
# =============================================================================

def expand_corpus(corpus: dict[str, str], generated_queries: dict[str, list[str]]) -> dict[str, str]:
    """
    doc2query document expansion (Lecture 9).

    For each document that has generated queries, append all of them to
    the document text:

        expanded_text = original_text + " " + " ".join(queries)

    Documents without generated queries are kept unchanged. The input
    corpus must not be modified.

    (In the tests, `generated_queries` comes from the REAL doc2query
    model via `generate_queries` — look at what it produces!)

    Parameters
    ----------
    corpus : dict[str, str]
        The corpus.
    generated_queries : dict[str, list[str]]
        A dictionary mapping document ID -> list of generated query
        strings.

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


# =============================================================================
# Task 5: Bi-encoder dense retrieval and late interaction (15 points) — Lecture 10
# =============================================================================

def encode_text(text: str) -> tuple[float, ...]:
    """
    Bi-encoder text encoding with mean pooling (Lecture 10).

    Encode a text into ONE vector by averaging the contextualized token
    vectors returned by `token_embeddings(text)` (mean pooling — this is
    exactly the pooling all-MiniLM-L6-v2 uses for its sentence
    embeddings). If the text has no tokens, return the zero vector of
    dimension VECTOR_DIM.

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
    ALL documents by cosine similarity of the vectors. Unlike reranking,
    this scores the whole corpus — in a real system the document
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
        A list of (document ID, cosine similarity) for ALL documents in
        descending score order (ties broken by ascending document ID).
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

    Use the token vectors from `token_embeddings`. If the query or the
    document has no token, return 0.0.

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
# Task 6: Learned sparse retrieval with SPLADE (10 points) — Lecture 11
# =============================================================================

def build_impact_index(
    term_weights: dict[str, dict[str, float]],
    scale: int = 100,
) -> dict[str, dict[str, int]]:
    """
    Quantize learned term weights into an impact index (Lecture 11).

    Each real-valued weight w is quantized to an integer
    impact = round(w * scale). Terms with impact <= 0 are NOT stored
    (just like terms with TF = 0 are absent from an inverted index).

    (In the tests, `term_weights` comes from the REAL SPLADE model via
    `splade_term_weights` — including its expansion terms.)

    Parameters
    ----------
    term_weights : dict[str, dict[str, float]]
        A dictionary mapping document ID -> {term: weight}.
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
    raise NotImplementedError("Please implement Task 6")


def impact_search(query: str, index: dict[str, dict[str, int]]) -> list[tuple[str, int]]:
    """
    Retrieve with the impact index (Lecture 11).

    The score of a document is the sum of the impacts of the query
    terms it contains. Query terms are obtained with `tokenize` and
    lowercased (SPLADE's vocabulary is lowercase); terms absent from
    the index contribute nothing:

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
    raise NotImplementedError("Please implement Task 6")


# =============================================================================
# Advanced tasks: analysis and discussion (10 points each, 30 points total)
# =============================================================================

"""
For each question below, experiment using the implementations above and write
down your results and discussion.

Q1 (10 points): The recall ceiling of reranking — and two ways around it.
    For the query "car fuel efficiency", run `retrieve_and_rerank` with k = 5
    and confirm that d05 (the "automobile ... gasoline ... mileage" document)
    never appears, even though the cross-encoder itself scores it clearly
    above the irrelevant documents. (a) Explain why reranking alone can never
    retrieve d05, relating your answer to the statement "BM25's Recall
    determines the performance ceiling of reranking" from Lecture 7.
    (b) Print `generate_queries(CORPUS["d05"])` and explain which generated
    term makes d05 retrievable by BM25 after `expand_corpus` (Lecture 9).
    (c) Print `splade_term_weights(CORPUS["d05"])` and find the weights of
    "car" and "fuel" — terms that do NOT occur in d05. How does SPLADE fix
    the same mismatch differently from doc2query (Lecture 11)?
    (d) Now try the q1 <-> d02 mismatch ("Kyoto autumn foliage" vs "old
    capital ... maple leaves"): print `generate_queries(CORPUS["d02"])` and
    check whether the expanded d02 becomes retrievable for q1. It does not —
    what knowledge would the model need to generate "Kyoto" from d02, and
    why does a small fine-tuned T5 not have it?

Q2 (10 points): Long document aggregation. For the long document d10 and the
    query "Kyoto autumn foliage", split d10 into passages with
    `split_passages(tokenize(CORPUS["d10"]), 16, 8)`, score each passage with
    `cross_encoder_score(query, " ".join(passage))`, and compare the document
    scores given by MaxP, FirstP, and SumP (`aggregate_scores`). Which passage
    wins under MaxP, and why does FirstP fail badly on this document (look at
    what the first 16 tokens of d10 talk about)? Discuss when SumP is biased
    by document length, and relate your findings to the Lecture 7 insight that
    "a document's relevance is determined by its most relevant portion".

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
    print("Information Retrieval — Lecture 12 Exercise (Lectures 7-11)")
    print("=" * 60)
    print("Note: the first run downloads ~870 MB of models from Hugging")
    print("Face and takes a few minutes; later runs load them from cache.")

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
        assert [len(p) for p in d10_ps] == [16, 16, 16, 16, 16, 13], \
            f"d10 passage lengths: expected=[16,16,16,16,16,13], actual={[len(p) for p in d10_ps]}"

        assert aggregate_scores([0.2, 0.9, 0.5], "max") == 0.9, "MaxP failed"
        assert aggregate_scores([0.2, 0.9, 0.5], "first") == 0.2, "FirstP failed"
        assert abs(aggregate_scores([0.2, 0.9, 0.5], "sum") - 1.6) < 1e-9, "SumP failed"

        # With the real cross-encoder, the travel-guide intro of d10 scores
        # ~0 but the foliage passage in the middle scores ~1: MaxP >> FirstP
        p_scores = [cross_encoder_score(QUERIES["q1"], " ".join(p)) for p in d10_ps]
        maxp = aggregate_scores(p_scores, "max")
        firstp = aggregate_scores(p_scores, "first")
        assert maxp > 0.9, f"MaxP of d10 should be > 0.9, actual={maxp:.4f}"
        assert firstp < 0.1, f"FirstP of d10 should be < 0.1, actual={firstp:.6f}"
        best = p_scores.index(maxp)
        print(f"  OK  splitting and MaxP/FirstP/SumP work "
              f"(d10: MaxP={maxp:.3f} at passage {best}, FirstP={firstp:.5f})")
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
        # relevant document d10; the real cross-encoder fixes the order
        # (it scores d10 ~0.999 and the decoy d11 ~0.05).
        bm25_ids = [d for d, _ in bm25_search(QUERIES["q1"], CORPUS)]
        assert bm25_ids == ["d01", "d11", "d10"], \
            f"BM25(q1): expected=['d01','d11','d10'], actual={bm25_ids} (provided function — corpus changed?)"

        rr = retrieve_and_rerank(QUERIES["q1"], CORPUS, 3)
        rr_ids = [d for d, _ in rr]
        assert rr_ids == ["d01", "d10", "d11"], \
            f"rerank(q1, k=3): expected=['d01','d10','d11'], actual={rr_ids}"
        assert all(0.0 <= s <= 1.0 for _, s in rr), "cross-encoder scores must be in [0, 1]"
        assert rr[0][1] > 0.99 and rr[2][1] < 0.1, \
            f"rerank(q1, k=3): expected scores ~[1.0, 1.0, 0.05], actual={[round(s, 4) for _, s in rr]}"

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
        assert pw == ["d01", "d11", "d03", "d02"], \
            f"pairwise(q1): expected=['d01','d11','d03','d02'], actual={pw}"

        init = ["d07", "d08", "d03", "d06", "d05", "d04"]
        sw = sliding_window_rerank(QUERIES["q2"], init, CORPUS, 3, 2)
        assert sw[0] == "d04", \
            f"sliding window: the best document d04 should bubble up to the top, actual top={sw[0]}"
        assert sw == ["d04", "d03", "d07", "d08", "d05", "d06"], \
            f"sliding window(q2, w=3, s=2): expected=['d04','d03','d07','d08','d05','d06'], actual={sw}"
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
    print("\n[Task 4] Document expansion with doc2query")
    try:
        exp = expand_corpus({"d1": "A B ."}, {"d1": ["C D", "E"]})
        assert exp == {"d1": "A B . C D E"}, \
            f"expand_corpus toy example: expected='A B . C D E', actual={exp}"

        # Generate queries with the REAL doc2query model
        gen = {d: generate_queries(CORPUS[d]) for d in ("d02", "d05", "d09")}
        assert "car" in " ".join(gen["d05"]).lower(), \
            f"doc2query should generate 'car' for d05 (automobile doc), actual={gen['d05']}"

        expanded = expand_corpus(CORPUS, gen)
        assert expanded["d01"] == CORPUS["d01"], "documents without generated queries must be unchanged"
        assert expanded["d05"].startswith(CORPUS["d05"]), "expansion must append, not replace"
        assert "car" not in CORPUS["d05"], "the input corpus must not be modified"

        # Vocabulary mismatch resolved for q2: the generated term "car"
        # makes d05 retrievable by BM25
        exp_q2 = [d for d, _ in bm25_search(QUERIES["q2"], expanded)]
        assert exp_q2 == ["d04", "d05"], \
            f"BM25(q2) on expanded corpus: expected=['d04','d05'], actual={exp_q2}"

        # ... but NOT for q1: doc2query cannot guess that the "old capital"
        # in d02 is Kyoto, so d02 is still not retrievable (see Q1d)
        exp_q1 = [d for d, _ in bm25_search(QUERIES["q1"], expanded)]
        assert "d02" not in exp_q1, \
            f"BM25(q1) on expanded corpus: d02 should still be missing, actual={exp_q1}"
        print(f"  OK  doc2query expansion works (d05 gen: {gen['d05'][0]!r} -> BM25(q2)={exp_q2})")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 5 ---
    print("\n[Task 5] Dense retrieval and late interaction")
    try:
        v = encode_text(QUERIES["q1"])
        assert len(v) == VECTOR_DIM, f"encode_text must return a vector of length {VECTOR_DIM}"
        zero = encode_text("")
        assert all(abs(x) < 1e-9 for x in zero), "empty text must encode to the zero vector"

        # Dense retrieval resolves the q2 vocabulary mismatch that BM25
        # cannot: d05 shares no term with q2 but is ranked 2nd
        bm25_q2 = [d for d, _ in bm25_search(QUERIES["q2"], CORPUS)]
        assert "d05" not in bm25_q2, "BM25 should not retrieve d05 for q2 (no shared terms)"
        dense_q2 = dense_search(QUERIES["q2"], CORPUS)
        assert len(dense_q2) == len(CORPUS), "dense_search must rank ALL documents"
        assert [d for d, _ in dense_q2[:2]] == ["d04", "d05"], \
            f"dense(q2): expected top 2=['d04','d05'], actual={[d for d, _ in dense_q2[:3]]}"
        assert dense_q2[1][1] > 0.4, \
            f"dense(q2): d05 cosine should be > 0.4, actual={dense_q2[1][1]:.4f}"

        dense_q1 = [d for d, _ in dense_search(QUERIES["q1"], CORPUS)]
        assert dense_q1[:2] == ["d01", "d10"], \
            f"dense(q1): expected top 2=['d01','d10'], actual={dense_q1[:3]}"
        # ... but the q1 <-> d02 mismatch ("old capital") is NOT resolved:
        # even a real bi-encoder does not place d02 near "Kyoto autumn foliage"
        assert "d02" not in dense_q1[:3], \
            f"dense(q1): d02 should NOT be in the top 3 (see Q1d), actual={dense_q1[:4]}"

        # Late interaction: MaxSim prefers the exact-topic document and
        # keeps the decoy d11 below the two real foliage documents
        c_q1 = {d: colbert_score(QUERIES["q1"], CORPUS[d]) for d in ("d01", "d10", "d11")}
        assert c_q1["d01"] > c_q1["d10"] > c_q1["d11"], \
            f"colbert(q1): expected d01 > d10 > d11, actual={ {d: round(s, 3) for d, s in c_q1.items()} }"
        c_d04 = colbert_score(QUERIES["q2"], CORPUS["d04"])
        c_d05 = colbert_score(QUERIES["q2"], CORPUS["d05"])
        assert c_d04 > c_d05 > colbert_score(QUERIES["q2"], CORPUS["d06"]), \
            f"colbert(q2): expected d04 > d05 > d06, actual d04={c_d04:.3f}, d05={c_d05:.3f}"
        assert colbert_score(QUERIES["q2"], "") == 0.0, \
            "colbert_score must be 0.0 when the document has no token"
        print(f"  OK  dense retrieval (q2 top 2: {[d for d, _ in dense_q2[:2]]}) and MaxSim work")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 6 ---
    print("\n[Task 6] Learned sparse retrieval with SPLADE")
    try:
        idx_toy = build_impact_index({"d1": {"apple": 0.9, "the": 0.0}}, scale=100)
        assert idx_toy == {"apple": {"d1": 90}}, \
            f"toy impact index: expected={{'apple': {{'d1': 90}}}}, actual={idx_toy}"

        # Build the impact index from the REAL SPLADE model
        weights = {d: splade_term_weights(t) for d, t in CORPUS.items()}
        idx = build_impact_index(weights, scale=100)

        # SPLADE expands d05 with terms it does not contain: "car", "fuel"
        assert "d05" in idx.get("car", {}) and "d05" in idx.get("fuel", {}), \
            "SPLADE should assign 'car' and 'fuel' impacts to d05 (expansion terms)"

        res_q2 = impact_search(QUERIES["q2"], idx)
        ids_q2 = [d for d, _ in res_q2]
        assert ids_q2[0] == "d04" and "d05" in ids_q2[:2], \
            f"impact_search(q2): expected d04 1st and d05 2nd, actual={res_q2[:3]}"

        res_q3 = impact_search(QUERIES["q3"], idx)
        assert res_q3[0][0] == "d07", \
            f"impact_search(q3): expected d07 first, actual={res_q3[:3]}"
        assert res_q3[0][1] == idx["search"]["d07"] + idx["engine"]["d07"], \
            "impact_search must sum the impacts of the query terms"
        print(f"  OK  impact index works (q2: {res_q2[:2]} — d05 found via SPLADE expansion)")
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
