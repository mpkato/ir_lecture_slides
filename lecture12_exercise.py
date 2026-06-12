"""
Information Retrieval — Lecture 12 Exercise (covers Lectures 7-10)
==================================================================

You implement the code that runs three real pre-trained neural IR
models. The models to use are fixed:

    Task 1-2 : cross-encoder/ms-marco-MiniLM-L6-v2        (~91 MB)
    Task 3   : doc2query/msmarco-t5-small-v1              (~242 MB)
    Task 4-5 : sentence-transformers/all-MiniLM-L6-v2     (~91 MB)

All are small enough to run on a CPU; the corpus has only 11 documents,
so a full run takes well under a minute. The models are downloaded
automatically from Hugging Face on first use and cached under
`~/.cache/huggingface`.

Setup:
    pip install -r requirements.txt

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
# d02 and d05 share NO term with their query, so BM25 scores them 0.

# BM25 results on this corpus, precomputed with the BM25 you implemented in
# the Lecture 6 exercise (k1 = 1.2, b = 0.75; documents with score 0 are
# excluded). These are the Stage-1 rankings used in Task 2.
BM25_RESULTS = {
    "q1": [("d01", 1.695), ("d11", 1.072), ("d10", 0.744)],
    "q2": [("d04", 3.018)],
    "q3": [("d07", 1.599), ("d04", 0.619)],
}

# The models to use (do not change — the tests assume exactly these):
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DOC2QUERY_MODEL = "doc2query/msmarco-t5-small-v1"
BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def cosine(u, v) -> float:
    """Cosine similarity between two vectors (0.0 if either is a zero vector)."""
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


# =============================================================================
# Task 1: Cross-encoder scoring (20 points) — Lectures 7-8
# =============================================================================

def cross_encoder_score(query: str, doc_text: str) -> float:
    """
    Score one (query, document) pair with the real cross-encoder
    CROSS_ENCODER_MODEL (monoBERT-style: one transformer forward pass
    over "[CLS] query [SEP] document [SEP]").

    Implementation notes:
    - Use `sentence_transformers.CrossEncoder`. Pass `device="cpu"` so
      that everyone gets the same (deterministic) results.
    - `model.predict([(query, doc_text)])` returns one raw logit; map
      it to (0, 1) with a sigmoid: 1 / (1 + exp(-logit)).
    - Loading the model takes a few seconds — create it ONCE (e.g.,
      store it in a module-level variable) and reuse it across calls.

    Returns
    -------
    float
        Relevance score in (0, 1); higher means more relevant.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 1")


# =============================================================================
# Task 2: Reranking (20 points) — Lecture 7
# =============================================================================

def rerank(query: str, doc_ids: list[str]) -> list[tuple[str, float]]:
    """
    Stage 2 of Retrieve-and-Rerank: re-score the given Stage-1
    candidates with `cross_encoder_score` and sort by the new score.

    Only the given candidates may appear in the output — documents the
    first stage did not retrieve are gone for good ("BM25's recall
    determines the performance ceiling of reranking", Lecture 7).

    Parameters
    ----------
    query : str
        The query string.
    doc_ids : list[str]
        Candidate document IDs (e.g., the BM25 top results from
        BM25_RESULTS); look the texts up in CORPUS.

    Returns
    -------
    list[tuple[str, float]]
        (document ID, cross-encoder score) in descending score order.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 2")


# =============================================================================
# Task 3: doc2query generation (20 points) — Lecture 9
# =============================================================================

def generate_queries(doc_text: str, n: int = 3) -> list[str]:
    """
    Generate `n` queries for a document with the real doc2query model
    DOC2QUERY_MODEL (a T5 fine-tuned on MS MARCO to map a passage to a
    query that the passage answers). Appending the generated queries to
    the document before indexing adds new terms for exact-match
    retrieval (document expansion, Lecture 9).

    Implementation notes:
    - Use `transformers.AutoTokenizer` and
      `transformers.AutoModelForSeq2SeqLM` (load each once, reuse).
    - Tokenize the document and call `model.generate` with
      `max_length=24, num_beams=4, num_return_sequences=n`
      (deterministic beam search — everyone gets the same output;
      do NOT use sampling).
    - Decode each output with `tokenizer.decode(..., skip_special_tokens=True)`.

    Returns
    -------
    list[str]
        The generated queries.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 3")


# =============================================================================
# Task 4: Bi-encoder embedding (20 points) — Lecture 10
# =============================================================================

def embed(text: str) -> list[float]:
    """
    Encode a text into ONE 384-dimensional vector with the real
    bi-encoder BI_ENCODER_MODEL (mean pooling over the token
    embeddings is done inside the model).

    Queries and documents are encoded INDEPENDENTLY — that is the
    whole point of the bi-encoder architecture (Lecture 10).

    Implementation notes:
    - Use `sentence_transformers.SentenceTransformer` with
      `device="cpu"`; create it once and reuse it.
    - `model.encode(text)` returns the vector (a numpy array — return
      it as a list of floats).

    Returns
    -------
    list[float]
        The embedding (length 384).
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 4")


# =============================================================================
# Task 5: Dense retrieval (20 points) — Lecture 10
# =============================================================================

def dense_search(query: str, corpus: dict[str, str]) -> list[tuple[str, float]]:
    """
    Single-vector dense retrieval: embed the query and EVERY document
    with `embed`, then rank ALL documents by cosine similarity
    (use the provided `cosine`).

    Unlike reranking, this scores the whole corpus — in a real system
    the document vectors are precomputed offline and searched with an
    ANN index (Lecture 10).

    Returns
    -------
    list[tuple[str, float]]
        (document ID, cosine similarity) for ALL documents in
        descending score order.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 5")


# =============================================================================
# Tests and execution
# =============================================================================

def run_tests():
    """Run the tests for all tasks."""
    print("=" * 60)
    print("Information Retrieval — Lecture 12 Exercise (Lectures 7-10)")
    print("=" * 60)
    print("Note: the first run downloads ~420 MB of models from Hugging")
    print("Face and may take a few minutes; later runs load them from cache.")

    errors = 0

    # --- Task 1 ---
    print("\n[Task 1] Cross-encoder scoring")
    try:
        s_rel = float(cross_encoder_score(QUERIES["q1"], CORPUS["d01"]))
        s_dec = float(cross_encoder_score(QUERIES["q1"], CORPUS["d11"]))
        s_irr = float(cross_encoder_score(QUERIES["q1"], CORPUS["d06"]))
        assert 0.0 <= s_rel <= 1.0, \
            f"score must be in (0, 1) — did you apply the sigmoid? got {s_rel}"
        assert s_rel > 0.9, f"CE(q1, d01) should be > 0.9, got {s_rel:.4f}"
        assert s_irr < 0.1, f"CE(q1, d06) should be < 0.1, got {s_irr:.6f}"
        # the decoy d11 contains "autumn" and "foliage" but is about fake
        # plastic decorations — the cross-encoder reads the context
        assert s_dec < 0.1, \
            f"CE(q1, d11) should be < 0.1 (d11 is an off-topic decoy), got {s_dec:.4f}"
        print(f"  OK  CE(q1): d01={s_rel:.4f}, d11(decoy)={s_dec:.4f}, d06={s_irr:.6f}")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 2 ---
    print("\n[Task 2] Reranking")
    try:
        # BM25 (Stage 1) ranks the decoy d11 above the long relevant
        # document d10; the cross-encoder must reverse them
        rr = rerank(QUERIES["q1"], [d for d, _ in BM25_RESULTS["q1"]])
        assert [d for d, _ in rr] == ["d01", "d10", "d11"], \
            f"rerank(q1, BM25 top 3): expected order ['d01','d10','d11'], got {[d for d, _ in rr]}"
        # only the given candidates may appear, in the given roles
        rr2 = rerank(QUERIES["q2"], ["d06", "d04"])
        assert [d for d, _ in rr2] == ["d04", "d06"], \
            f"rerank(q2, ['d06','d04']): expected ['d04','d06'], got {[d for d, _ in rr2]}"
        print(f"  OK  BM25 {[d for d, _ in BM25_RESULTS['q1']]} -> CE {[d for d, _ in rr]} (decoy d11 demoted)")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 3 ---
    print("\n[Task 3] doc2query generation")
    try:
        gen = generate_queries(CORPUS["d05"])
        assert isinstance(gen, list) and len(gen) == 3 and all(isinstance(g, str) for g in gen), \
            f"generate_queries must return 3 strings, got {gen}"
        # d05 says "automobile ... gasoline ... mileage"; with beam search the
        # model generates the term "car" — this is what fixes the vocabulary
        # mismatch with q2 "car fuel efficiency" after expansion
        assert "car" in " ".join(gen).lower(), \
            f"doc2query should generate 'car' for d05 — got {gen} (sampling instead of beams?)"
        print(f"  OK  d05 -> {gen}")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 4 ---
    print("\n[Task 4] Bi-encoder embedding")
    try:
        v = embed("car")
        assert len(v) == 384, f"embedding must have 384 dimensions, got {len(v)}"
        sim_syn = float(cosine(embed("car"), embed("automobile")))
        sim_far = float(cosine(embed("car"), embed("temple")))
        assert sim_syn > sim_far, \
            f"'car' should be closer to 'automobile' ({sim_syn:.4f}) than to 'temple' ({sim_far:.4f})"
        print(f"  OK  cos(car, automobile)={sim_syn:.4f} > cos(car, temple)={sim_far:.4f}")
    except NotImplementedError:
        print("  X   not implemented")
        errors += 1
    except Exception as e:
        print(f"  X   error: {e}")
        errors += 1

    # --- Task 5 ---
    print("\n[Task 5] Dense retrieval")
    try:
        res = dense_search(QUERIES["q2"], CORPUS)
        assert len(res) == len(CORPUS), "dense_search must rank ALL documents"
        # d05 shares NO term with q2 (BM25 score 0), yet the bi-encoder
        # ranks it 2nd — dense retrieval resolves the vocabulary mismatch
        assert [d for d, _ in res[:2]] == ["d04", "d05"], \
            f"dense(q2): expected top 2 ['d04','d05'], got {[d for d, _ in res[:3]]}"
        res_q1 = dense_search(QUERIES["q1"], CORPUS)
        assert res_q1[0][0] == "d01", \
            f"dense(q1): expected d01 first, got {res_q1[:3]}"
        top = [(d, round(float(s), 4)) for d, s in res[:3]]
        print(f"  OK  dense(q2) top 3: {top} (d05 found with zero term overlap)")
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


if __name__ == "__main__":
    run_tests()
