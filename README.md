# Information Retrieval - Lecture Slides

Lecture slides for the Information Retrieval course.

**Instructor:** Makoto P. Kato (National Institute of Informatics / University of Tsukuba)

## Lectures

| # | Topic | Slides |
|---|-------|--------|
| 1 | Document Ranking and Keyword Search | [PDF](lecture1.pdf) |
| 2 | Machine Learning and Information Retrieval | [PDF](lecture2.pdf) |
| 3 | Relevance and Search Evaluation | [PDF](lecture3.pdf) |
| 4 | Traditional Retrieval Models | [PDF](lecture4.pdf) |
| 5 | Transformer and BERT for Ranking | [PDF](lecture5.pdf) |
| 6 | Exercise (hands-on) | [PDF](lecture6.pdf) · [exercise.py](lecture6_exercise.py) |
| 7 | Neural Ranking Models: Long Document Ranking | [PDF](lecture7.pdf) |
| 8 | Cross-Encoder Reranking | [PDF](lecture8.pdf) |
| 9 | Query and Document Expansion | [PDF](lecture9.pdf) |
| 10 | Bi-Encoders and Dense Retrieval | |
| 11 | Learned Sparse Retrieval | |
| 12 | Exercise (hands-on) | [exercise.py](lecture12_exercise.py) |

## Lecture 12 exercise

In the Lecture 12 exercise ([lecture12_exercise.py](lecture12_exercise.py),
covering Lectures 7-10) you run three real pre-trained models — a
cross-encoder, doc2query, and a bi-encoder — on a tiny corpus and write down
your observations. Everything runs on a CPU; about 420 MB of models are
downloaded from Hugging Face on the first run.

```bash
pip install -r requirements.txt
python lecture12_exercise.py
```
