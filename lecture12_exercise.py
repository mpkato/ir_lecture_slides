"""
Information Retrieval — Lecture 12 Exercise (covers Lectures 7-10)
==================================================================

In this hands-on you make three real neural IR models work on a tiny
corpus and OBSERVE how they behave:

    Experiment 1: cross-encoder reranking        (Lectures 7-8)
    Experiment 2: doc2query document expansion   (Lecture 9)
    Experiment 3: bi-encoder dense retrieval     (Lecture 10)

YOUR JOB is to implement the five functions marked TODO (Tasks 1-5) —
the code that actually calls the models. The models to use are fixed:

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

The script first checks your implementations (loose sanity checks),
then runs the three experiments with them. Read the output carefully
and write down your answers to the questions Q1-Q4 at the bottom of
this file.
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
# Whether (and how) each neural method fixes this mismatch is the running
# theme of the three experiments.

# BM25 results on this corpus, precomputed with the BM25 you implemented in
# the Lecture 6 exercise (k1 = 1.2, b = 0.75; documents with score 0 are
# excluded). This is the Stage-1 ranking used in Experiment 1.
BM25_RESULTS = {
    "q1": [("d01", 1.695), ("d11", 1.072), ("d10", 0.744)],
    "q2": [("d04", 3.018)],
    "q3": [("d07", 1.599), ("d04", 0.619)],
}

# The models to use (do not change — the checks and the expected
# behavior in the questions assume exactly these):
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DOC2QUERY_MODEL = "doc2query/msmarco-t5-small-v1"
BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text: str) -> list[str]:
    """Tokenize on whitespace, dropping punctuation tokens."""
    return [t for t in text.split() if t not in {".", ",", ";", ":", "。", "、", "，", "．"}]


def cosine(u, v) -> float:
    """Cosine similarity between two vectors (0.0 if either is a zero vector)."""
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def shared_terms(query: str, text: str) -> list[str]:
    """Query terms that also occur in the text (case-insensitive) — the
    terms an exact-match model like BM25 can use."""
    q = {t.lower() for t in tokenize(query)}
    d = {t.lower() for t in tokenize(text)}
    return sorted(q & d)


# =============================================================================
# Task 1: Cross-encoder scoring (Lectures 7-8)
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
# Task 2: Reranking (Lecture 7)
# =============================================================================

def rerank(query: str, doc_ids: list[str]) -> list[tuple[str, float]]:
    """
    Stage 2 of Retrieve-and-Rerank: re-score the given Stage-1
    candidates with `cross_encoder_score` and sort by the new score.

    Only the given candidates may appear in the output — documents the
    first stage did not retrieve are gone for good (the "recall
    ceiling" you will observe in Experiment 1).

    Parameters
    ----------
    query : str
        The query string.
    doc_ids : list[str]
        Candidate document IDs (e.g., the BM25 top results); look the
        texts up in CORPUS.

    Returns
    -------
    list[tuple[str, float]]
        (document ID, cross-encoder score) in descending score order.
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 2")


# =============================================================================
# Task 3: doc2query generation (Lecture 9)
# =============================================================================

def generate_queries(doc_text: str, n: int = 3) -> list[str]:
    """
    Generate `n` queries for a document with the real doc2query model
    DOC2QUERY_MODEL (a T5 fine-tuned on MS MARCO to map a passage to a
    query that the passage answers).

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
# Task 4: Bi-encoder embedding (Lecture 10)
# =============================================================================

def embed(text: str) -> list[float]:
    """
    Encode a text into ONE 384-dimensional vector with the real
    bi-encoder BI_ENCODER_MODEL (mean pooling over the token
    embeddings is done inside the model).

    Implementation notes:
    - Use `sentence_transformers.SentenceTransformer` with
      `device="cpu"`; create it once and reuse it.
    - `model.encode(text)` returns the vector (a numpy array — return
      it as a list of floats).

    Queries and documents are encoded INDEPENDENTLY — that is the
    whole point of the bi-encoder architecture (Lecture 10).

    Returns
    -------
    list[float]
        The embedding (length 384).
    """
    # TODO: implement this
    raise NotImplementedError("Please implement Task 4")


# =============================================================================
# Task 5: Dense retrieval (Lecture 10)
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
# Sanity checks for your implementations
# =============================================================================

def run_checks() -> dict[int, bool]:
    """Loose checks so you know each task basically works."""
    print("=" * 72)
    print("Checking your implementations")
    print("=" * 72)
    ok = {}

    def check(task, fn):
        try:
            fn()
            ok[task] = True
            print(f"  Task {task}: OK")
        except NotImplementedError:
            ok[task] = False
            print(f"  Task {task}: not implemented yet")
        except Exception as e:
            ok[task] = False
            print(f"  Task {task}: ERROR — {e}")

    def t1():
        s_rel = cross_encoder_score(QUERIES["q1"], CORPUS["d01"])
        s_irr = cross_encoder_score(QUERIES["q1"], CORPUS["d06"])
        assert isinstance(s_rel, float) and 0.0 <= s_rel <= 1.0, \
            f"score must be a float in (0, 1) — did you apply the sigmoid? got {s_rel}"
        assert s_rel > 0.9, f"CE(q1, d01) should be > 0.9, got {s_rel:.4f}"
        assert s_irr < 0.1, f"CE(q1, d06) should be < 0.1, got {s_irr:.6f}"

    def t2():
        rr = rerank(QUERIES["q1"], [d for d, _ in BM25_RESULTS["q1"]])
        assert [d for d, _ in rr] == ["d01", "d10", "d11"], \
            f"rerank(q1, BM25 top 3): expected order ['d01','d10','d11'], got {[d for d, _ in rr]}"

    def t3():
        gen = generate_queries(CORPUS["d05"])
        assert isinstance(gen, list) and len(gen) == 3 and all(isinstance(g, str) for g in gen), \
            f"generate_queries must return 3 strings, got {gen}"
        assert "car" in " ".join(gen).lower(), \
            f"with beam search, doc2query generates 'car' for d05 — got {gen} (sampling instead of beams?)"

    def t4():
        v = embed("car")
        assert len(v) == 384, f"embedding must have 384 dimensions, got {len(v)}"
        assert cosine(embed("car"), embed("automobile")) > cosine(embed("car"), embed("temple")), \
            "'car' should be closer to 'automobile' than to 'temple'"

    def t5():
        res = dense_search(QUERIES["q2"], CORPUS)
        assert len(res) == len(CORPUS), "dense_search must rank ALL documents"
        assert [d for d, _ in res[:2]] == ["d04", "d05"], \
            f"dense(q2): expected top 2 ['d04','d05'], got {[d for d, _ in res[:3]]}"

    check(1, t1)
    check(2, t2)
    check(3, t3)
    check(4, t4)
    check(5, t5)
    return ok


# =============================================================================
# Experiment 1: Cross-encoder reranking (Lectures 7-8) — uses Tasks 1-2
# =============================================================================

def experiment1():
    print("\n" + "=" * 72)
    print("Experiment 1: Cross-encoder reranking (Lectures 7-8)")
    print("=" * 72)

    # --- 1a: Retrieve-and-Rerank for q1 ----------------------------------
    q = QUERIES["q1"]
    print(f'\n[1a] Retrieve-and-Rerank for q1 = "{q}"')
    print("\n  Stage 1 — BM25 ranking (your Lecture 6 implementation):")
    for rank, (d, s) in enumerate(BM25_RESULTS["q1"], 1):
        print(f"    {rank}. {d}  BM25={s:<6}  {CORPUS[d][:60]}...")

    print("\n  Stage 2 — your `rerank` rescores the same candidates:")
    for rank, (d, s) in enumerate(rerank(q, [d for d, _ in BM25_RESULTS["q1"]]), 1):
        print(f"    {rank}. {d}  CE={s:<8.6f}  {CORPUS[d][:60]}...")
    print("\n  >> Compare the two rankings: which documents swapped places?")
    print("     Look at d11 (the decoy) — it CONTAINS 'autumn' and 'foliage'.")

    # --- 1b: where in the long document d10 is the relevance? ------------
    print(f'\n[1b] Per-sentence cross-encoder scores of the long document d10 (q1 = "{q}")')
    for i, sent in enumerate(s.strip() for s in CORPUS["d10"].split(" . ")):
        print(f"    sentence {i}: CE={cross_encoder_score(q, sent):<8.6f}  {sent[:64]}")
    print(f"    whole d10 : CE={cross_encoder_score(q, CORPUS['d10']):<8.6f}")
    print("\n  >> Which single sentence carries (almost) all the relevance?")

    # --- 1c: the recall ceiling -------------------------------------------
    q2 = QUERIES["q2"]
    print(f'\n[1c] The recall ceiling: q2 = "{q2}"')
    print(f"\n  Stage 1 — BM25 ranking: {BM25_RESULTS['q2']}")
    print("  (d05 has BM25 score 0 — it shares no term with q2:",
          f"shared_terms = {shared_terms(q2, CORPUS['d05'])})")
    print("\n  But the cross-encoder CAN see that d05 is relevant:")
    for d in ("d04", "d05", "d06"):
        print(f"    CE(q2, {d}) = {cross_encoder_score(q2, CORPUS[d]):<8.6f}  {CORPUS[d][:55]}...")
    print("\n  >> d05 scores far above the irrelevant d06 — yet a")
    print("     Retrieve-and-Rerank pipeline can NEVER return it. Why?")


# =============================================================================
# Experiment 2: Document expansion with doc2query (Lecture 9) — uses Task 3
# =============================================================================

def experiment2():
    print("\n" + "=" * 72)
    print("Experiment 2: Document expansion with doc2query (Lecture 9)")
    print("=" * 72)
    print("\ndoc2query reads a document and generates queries the document")
    print("could answer; the queries are APPENDED to the document before")
    print("indexing, adding new terms for exact-match retrieval (BM25).")

    for d in ("d05", "d02", "d09"):
        print(f"\n[{d}] {CORPUS[d]}")
        queries = generate_queries(CORPUS[d])
        for g in queries:
            print(f"    generated: {g!r}")
        expanded = CORPUS[d] + " " + " ".join(queries)
        for q_id in ("q1", "q2"):
            before = shared_terms(QUERIES[q_id], CORPUS[d])
            after = shared_terms(QUERIES[q_id], expanded)
            if before != after:
                print(f"    -> terms shared with {q_id} \"{QUERIES[q_id]}\": "
                      f"{before} -> {after}  (BM25 score is no longer 0!)")
    print("\n  >> Which generated TERM makes d05 retrievable for q2?")
    print("  >> d02 is about Kyoto ('the old capital'), but does any")
    print("     generated query contain 'Kyoto'? Why (not)?")


# =============================================================================
# Experiment 3: Bi-encoder dense retrieval (Lecture 10) — uses Tasks 4-5
# =============================================================================

def experiment3():
    print("\n" + "=" * 72)
    print("Experiment 3: Bi-encoder dense retrieval (Lecture 10)")
    print("=" * 72)
    print("\nThe bi-encoder embeds queries and documents INDEPENDENTLY into")
    print("384-dim vectors; relevance = cosine similarity. Document vectors")
    print("can be precomputed and searched with an ANN index (Lecture 10).")

    for q_id in ("q1", "q2"):
        q = QUERIES[q_id]
        print(f'\n[{q_id}] "{q}" — dense ranking of ALL 11 documents:')
        for rank, (d, s) in enumerate(dense_search(q, CORPUS), 1):
            mark = " <-- shares NO term with the query" if rank <= 3 and s > 0.3 and not shared_terms(q, CORPUS[d]) else ""
            print(f"    {rank:2}. {d}  cos={s:<7.4f}  {CORPUS[d][:52]}...{mark}")
    print("\n  >> For q2: where does d05 rank, even though BM25 scores it 0?")
    print("  >> For q1: where does d02 (the 'old capital' document) rank?")
    print("     Did ANY method in this exercise manage to connect d02 to q1?")


# =============================================================================
# Try your own query (for Q4)
# =============================================================================

def try_your_own_query(query: str):
    """Run all three methods for an arbitrary query and print the results."""
    print("\n" + "=" * 72)
    print(f'Your query: "{query}"')
    print("=" * 72)
    print("\n  shared terms (what BM25 could match):")
    for d, t in CORPUS.items():
        st = shared_terms(query, t)
        if st:
            print(f"    {d}: {st}")
    print("\n  cross-encoder scores (top 5):")
    for d, s in rerank(query, list(CORPUS))[:5]:
        print(f"    {d}  CE={s:<8.6f}  {CORPUS[d][:55]}...")
    print("\n  bi-encoder cosine (top 5):")
    dense = dense_search(query, CORPUS)
    for d, s in dense[:5]:
        print(f"    {d}  cos={s:<7.4f}  {CORPUS[d][:55]}...")
    print("\n  doc2query for the top dense document:")
    for g in generate_queries(CORPUS[dense[0][0]]):
        print(f"    {dense[0][0]} generated: {g!r}")


# =============================================================================
# Questions (write down your answers and observations)
# =============================================================================

QUESTIONS = """
Q1 (Experiment 1 — cross-encoder):
  (a) For q1, BM25 ranks the decoy d11 above the long relevant document
      d10, but the cross-encoder reverses them. d11 literally contains
      "autumn" and "foliage" — what can the cross-encoder see that BM25
      cannot?
  (b) Looking at the per-sentence scores of d10 [1b]: which sentence
      carries the relevance, and what does this suggest about how long
      documents should be scored (recall MaxP from Lecture 7)?
  (c) In [1c] the cross-encoder clearly recognizes d05 as relevant to
      q2, yet Retrieve-and-Rerank can never return it. Explain why,
      using the phrase "BM25's recall determines the performance
      ceiling of reranking" (Lecture 7).

Q2 (Experiment 2 — doc2query):
  (a) Which generated term makes d05 retrievable for q2 after
      expansion?
  (b) d02 is about Kyoto ("the old capital ... maple leaves ... fall"),
      but no generated query contains "Kyoto". What would the model
      need to know to generate it? What does this tell you about the
      limits of document expansion?
  (c) Look at the queries generated for d09. doc2query appends its
      output into the index — what kinds of errors or noise could this
      introduce in a larger collection?

Q3 (Experiment 3 — bi-encoder):
  (a) For q2, where does d05 rank in the dense ranking, and why can the
      bi-encoder find it when BM25 cannot?
  (b) For q1, where does d02 rank? None of the three methods connects
      d02 to "Kyoto autumn foliage" — what makes this mismatch harder
      than the d05 <-> q2 one?
  (c) The bi-encoder embeds each document once, offline; the
      cross-encoder needs one forward pass per (query, document) pair
      at query time. Using this, explain why real systems use the
      bi-encoder (or BM25) for Stage 1 and the cross-encoder for
      Stage 2 (Lectures 7, 10).

Q4 (free exploration):
  Call `try_your_own_query("...")` (see the bottom of this file) with
  at least one query of your own. Report the query, the outputs, and
  one behavior that surprised you — e.g., a disagreement between the
  cross-encoder and the bi-encoder, or a doc2query output that would
  help or hurt retrieval.
"""


if __name__ == "__main__":
    print("Information Retrieval — Lecture 12 Exercise (Lectures 7-10)")
    print("Note: the first run downloads ~420 MB of models from Hugging")
    print("Face and may take a few minutes; later runs load them from cache.\n")
    ok = run_checks()
    if ok[1] and ok[2]:
        experiment1()
    else:
        print("\n(Experiment 1 skipped — finish Tasks 1-2 first)")
    if ok[3]:
        experiment2()
    else:
        print("\n(Experiment 2 skipped — finish Task 3 first)")
    if ok[4] and ok[5]:
        experiment3()
    else:
        print("\n(Experiment 3 skipped — finish Tasks 4-5 first)")
    if all(ok.values()):
        # Q4: uncomment and put your own query here
        # try_your_own_query("ramen restaurants in Fukuoka")
        print("\n" + "=" * 72)
        print("All tasks done. Now answer the questions Q1-Q4 at the bottom")
        print("of this file (and try your own query for Q4).")
        print("=" * 72)
