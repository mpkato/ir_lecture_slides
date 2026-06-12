"""
Information Retrieval — Lecture 12 Exercise (covers Lectures 7-10)
==================================================================

In this hands-on you TRY OUT three real neural IR models on a tiny
corpus and OBSERVE how they behave:

    Experiment 1: cross-encoder reranking        (Lectures 7-8)
    Experiment 2: doc2query document expansion   (Lecture 9)
    Experiment 3: bi-encoder dense retrieval     (Lecture 10)

There is no code to implement. Run the script, read the output
carefully, and write down your answers to the questions Q1-Q4 at the
bottom of this file.

Everything runs on a CPU in well under a minute — the corpus has only
11 documents. The models (about 420 MB in total) are downloaded
automatically from Hugging Face on the first run and cached under
`~/.cache/huggingface`:

    cross-encoder/ms-marco-MiniLM-L6-v2        (~91 MB)   cross-encoder
    doc2query/msmarco-t5-small-v1              (~242 MB)  doc2query
    sentence-transformers/all-MiniLM-L6-v2     (~91 MB)   bi-encoder

Setup:
    pip install -r requirements.txt

How to run:
    python lecture12_exercise.py
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


def shared_terms(query: str, text: str) -> list[str]:
    """Query terms that also occur in the text (case-insensitive) — the
    terms an exact-match model like BM25 can use."""
    q = {t.lower() for t in tokenize(query)}
    d = {t.lower() for t in tokenize(text)}
    return sorted(q & d)


# =============================================================================
# Real models (loaded lazily on first use, then kept in memory)
# =============================================================================

_MODELS: dict = {}


def cross_encoder_score(query: str, doc_text: str) -> float:
    """
    Real cross-encoder relevance scoring (monoBERT-style, Lectures 7-8).

    One transformer forward pass over "[CLS] query [SEP] document [SEP]"
    using cross-encoder/ms-marco-MiniLM-L6-v2 (a 6-layer MiniLM trained
    on MS MARCO). The raw logit is mapped to (0, 1) with a sigmoid;
    higher means more relevant.
    """
    if "ce" not in _MODELS:
        from sentence_transformers import CrossEncoder
        print("  (loading cross-encoder/ms-marco-MiniLM-L6-v2 ...)")
        _MODELS["ce"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
    logit = float(_MODELS["ce"].predict([(query, doc_text)])[0])
    return round(1.0 / (1.0 + math.exp(-logit)), 6)


def generate_queries(doc_text: str, n: int = 3) -> list[str]:
    """
    Real doc2query / docTTTTTquery query generation (Lecture 9).

    Feeds the document to doc2query/msmarco-t5-small-v1 (a T5 fine-tuned
    on MS MARCO to map a passage to a query that the passage answers)
    and returns `n` generated queries. Deterministic beam search is used
    so that everyone gets the same output.
    """
    import torch
    if "d2q" not in _MODELS:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print("  (loading doc2query/msmarco-t5-small-v1 ...)")
        tok = AutoTokenizer.from_pretrained("doc2query/msmarco-t5-small-v1")
        model = AutoModelForSeq2SeqLM.from_pretrained("doc2query/msmarco-t5-small-v1")
        model.eval()
        _MODELS["d2q"] = (tok, model)
    tok, model = _MODELS["d2q"]
    ids = tok(doc_text, return_tensors="pt", truncation=True, max_length=384).input_ids
    with torch.no_grad():
        out = model.generate(ids, max_length=24, num_beams=max(4, n), num_return_sequences=n)
    return [tok.decode(o, skip_special_tokens=True) for o in out]


def embed(text: str) -> tuple[float, ...]:
    """
    Real bi-encoder text embedding (Lecture 10).

    Encodes the text into ONE 384-dimensional vector with
    sentence-transformers/all-MiniLM-L6-v2 (mean pooling over the token
    embeddings). Queries and documents are encoded INDEPENDENTLY —
    compare vectors with `cosine`.
    """
    if "be" not in _MODELS:
        from sentence_transformers import SentenceTransformer
        print("  (loading sentence-transformers/all-MiniLM-L6-v2 ...)")
        _MODELS["be"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return tuple(float(x) for x in _MODELS["be"].encode(text))


# =============================================================================
# Experiment 1: Cross-encoder reranking (Lectures 7-8)
# =============================================================================

def experiment1():
    print("=" * 72)
    print("Experiment 1: Cross-encoder reranking (Lectures 7-8)")
    print("=" * 72)

    # --- 1a: Retrieve-and-Rerank for q1 ----------------------------------
    q = QUERIES["q1"]
    print(f'\n[1a] Retrieve-and-Rerank for q1 = "{q}"')
    print("\n  Stage 1 — BM25 ranking (your Lecture 6 implementation):")
    for rank, (d, s) in enumerate(BM25_RESULTS["q1"], 1):
        print(f"    {rank}. {d}  BM25={s:<6}  {CORPUS[d][:60]}...")

    print("\n  Stage 2 — the cross-encoder rescores the same candidates:")
    rescored = sorted(
        ((d, cross_encoder_score(q, CORPUS[d])) for d, _ in BM25_RESULTS["q1"]),
        key=lambda x: -x[1],
    )
    for rank, (d, s) in enumerate(rescored, 1):
        print(f"    {rank}. {d}  CE={s:<8}  {CORPUS[d][:60]}...")
    print("\n  >> Compare the two rankings: which documents swapped places?")
    print("     Look at d11 (the decoy) — it CONTAINS 'autumn' and 'foliage'.")

    # --- 1b: where in the long document d10 is the relevance? ------------
    print(f'\n[1b] Per-sentence cross-encoder scores of the long document d10 (q1 = "{q}")')
    for i, sent in enumerate(s.strip() for s in CORPUS["d10"].split(" . ")):
        print(f"    sentence {i}: CE={cross_encoder_score(q, sent):<8}  {sent[:64]}")
    print(f"    whole d10 : CE={cross_encoder_score(q, CORPUS['d10']):<8}")
    print("\n  >> Which single sentence carries (almost) all the relevance?")

    # --- 1c: the recall ceiling -------------------------------------------
    q2 = QUERIES["q2"]
    print(f'\n[1c] The recall ceiling: q2 = "{q2}"')
    print(f"\n  Stage 1 — BM25 ranking: {BM25_RESULTS['q2']}")
    print("  (d05 has BM25 score 0 — it shares no term with q2:",
          f"shared_terms = {shared_terms(q2, CORPUS['d05'])})")
    print("\n  But the cross-encoder CAN see that d05 is relevant:")
    for d in ("d04", "d05", "d06"):
        print(f"    CE(q2, {d}) = {cross_encoder_score(q2, CORPUS[d]):<8}  {CORPUS[d][:55]}...")
    print("\n  >> d05 scores far above the irrelevant d06 — yet a")
    print("     Retrieve-and-Rerank pipeline can NEVER return it. Why?")


# =============================================================================
# Experiment 2: Document expansion with doc2query (Lecture 9)
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
# Experiment 3: Bi-encoder dense retrieval (Lecture 10)
# =============================================================================

def experiment3():
    print("\n" + "=" * 72)
    print("Experiment 3: Bi-encoder dense retrieval (Lecture 10)")
    print("=" * 72)
    print("\nThe bi-encoder embeds queries and documents INDEPENDENTLY into")
    print("384-dim vectors; relevance = cosine similarity. Document vectors")
    print("can be precomputed and searched with an ANN index (Lecture 10).")

    doc_vecs = {d: embed(t) for d, t in CORPUS.items()}
    for q_id in ("q1", "q2"):
        q = QUERIES[q_id]
        ranking = sorted(
            ((d, round(cosine(embed(q), doc_vecs[d]), 4)) for d in CORPUS),
            key=lambda x: -x[1],
        )
        print(f'\n[{q_id}] "{q}" — dense ranking of ALL 11 documents:')
        for rank, (d, s) in enumerate(ranking, 1):
            mark = " <-- shares NO term with the query" if rank <= 3 and not shared_terms(q, CORPUS[d]) else ""
            print(f"    {rank:2}. {d}  cos={s:<7}  {CORPUS[d][:52]}...{mark}")
    print("\n  >> For q2: where does d05 rank, even though BM25 scores it 0?")
    print("  >> For q1: where does d02 (the 'old capital' document) rank?")
    print("     Did ANY method in this exercise manage to connect d02 to q1?")


# =============================================================================
# Try your own query
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
    ce = sorted(((d, cross_encoder_score(query, t)) for d, t in CORPUS.items()), key=lambda x: -x[1])
    for d, s in ce[:5]:
        print(f"    {d}  CE={s:<8}  {CORPUS[d][:55]}...")
    print("\n  bi-encoder cosine (top 5):")
    qv = embed(query)
    dense = sorted(((d, round(cosine(qv, embed(t)), 4)) for d, t in CORPUS.items()), key=lambda x: -x[1])
    for d, s in dense[:5]:
        print(f"    {d}  cos={s:<7}  {CORPUS[d][:55]}...")
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
    experiment1()
    experiment2()
    experiment3()
    # Q4: uncomment and put your own query here
    # try_your_own_query("ramen restaurants in Fukuoka")
    print("\n" + "=" * 72)
    print("Done. Now answer the questions Q1-Q4 at the bottom of this file.")
    print("=" * 72)
