# Figures (SVG)

Redrawn vector figures for Lectures 1–5 and 7–9. Each file is a self-contained
SVG (no external fonts or scripts required) that follows the slide color scheme
(blue `#2B87B8` / orange `#E87E2E`). They can be embedded directly in Marp
slides, e.g.:

```markdown
![w:560](figures/lecture7/monobert-architecture.svg)
```

## Lecture 1 — Document Ranking and Keyword Search

| File | Slide | Contents |
|------|-------|----------|
| `lecture1/document-ranking-overview.svg` | p.4 | Query + corpus → ranking model → scored ranked list |
| `lecture1/ctr-by-rank.svg` | p.5 | Click-through rate by search result position (conceptual) |
| `lecture1/ir-timeline.svg` | p.20 | IR history timeline 1945 (Bush) → 2018 (BERT) with era bands |
| `lecture1/vocabulary-mismatch.svg` | p.23 | Query vs document vocabulary with broken matches, BM25 = 0 |
| `lecture1/bm25-structure.svg` | p.27 | BM25 decomposition: IDF × TF saturation + parameters |
| `lecture1/tf-saturation.svg` | p.29 | Raw TF / log TF / BM25 TF weighting curves (also used in L4 p.17) |
| `lecture1/inverted-index.svg` | p.37 | Dictionary and posting lists structure |

## Lecture 2 — Machine Learning and Information Retrieval

| File | Slide | Contents |
|------|-------|----------|
| `lecture2/ltr-pipeline.svg` | p.6 | Learning-to-Rank flow: features → model → score |
| `lecture2/ltr-three-categories.svg` | p.8 | Pointwise / Pairwise / Listwise three-column comparison |
| `lecture2/ranknet-sigmoid.svg` | p.11 | RankNet ordering probability curve |
| `lecture2/ranknet-to-lambdarank.svg` | p.18 | Gradient weighting by &#124;ΔnDCG&#124; comparison |
| `lecture2/representation-vs-interaction.svg` | p.25 | Two pre-BERT neural model families |
| `lecture2/dssm-architecture.svg` | p.27 | DSSM two-tower architecture with word hashing |
| `lecture2/drmm-architecture.svg` | p.30 | DRMM similarity matrix → histogram → FFN |
| `lecture2/msmarco-bert-impact.svg` | p.38 | MRR@10 bar chart: BM25 / IRNet / BERT |

## Lecture 3 — Relevance and Search Evaluation

| File | Slide | Contents |
|------|-------|----------|
| `lecture3/taylor-four-levels.svg` | p.8 | Taylor's four levels from information need to query |
| `lecture3/query-need-gap.svg` | p.9 | Gap between the query and the true information need |
| `lecture3/ndcg-gain-discount.svg` | p.31 | nDCG gain (2^rel − 1) and discount (1/log₂(i+1)) |
| `lecture3/evaluation-workflow.svg` | p.37 | Test collection components and trec_eval workflow |
| `lecture3/pooling-method.svg` | p.40 | Top-k pooling: collect runs → merge → judge |

## Lecture 4 — Traditional Retrieval Models

| File | Slide | Contents |
|------|-------|----------|
| `lecture4/four-models-evolution.svg` | p.5 / p.40 | Boolean → VSM → Probabilistic → LM evolution |
| `lecture4/boolean-venn.svg` | p.9 | Set-theoretic interpretation of Boolean operators |
| `lecture4/cosine-similarity.svg` | p.13 | Geometric interpretation with q=(2,3), d₁=(4,1), d₂=(1,3) |
| `lecture4/bim-to-bm25.svg` | p.28 | Theoretical bridge: BIM → 2-Poisson → BM25 |
| `lecture4/query-likelihood.svg` | p.33 | Language-model ranking via P(q&#124;M_d) with worked example |
| `lecture4/smoothing-effect.svg` | p.36 | MLE vs Dirichlet smoothing bar chart |

## Lecture 5 — Transformer and BERT for Ranking

| File | Slide | Contents |
|------|-------|----------|
| `lecture5/mlp.svg` | p.5 | Multi-layer perceptron with weighted connections |
| `lecture5/activation-functions.svg` | p.6 | ReLU / Sigmoid / GELU curves |
| `lecture5/one-hot-vs-distributed.svg` | p.7 | One-hot vs distributed word representations |
| `lecture5/qkv-projection.svg` | p.12 | Self-Attention Q/K/V projections and intuition |
| `lecture5/transformer-encoder.svg` | p.19 | Encoder layer: MHSA + FFN + Add&LayerNorm + residuals |
| `lecture5/bert-input-embeddings.svg` | p.26 | Token + Segment + Position embedding sum |
| `lecture5/pretrain-finetune.svg` | p.37 | Two-phase pre-training → fine-tuning pipeline |
| `lecture5/pretrained-model-evolution.svg` | p.23 | word2vec/GloVe → ELMo → GPT/BERT |

## Lecture 7 — Neural Ranking Models: Long Document Ranking

| File | Slide | Contents |
|------|-------|----------|
| `lecture7/monobert-architecture.svg` | p.5 | monoBERT input format and scoring from the `[CLS]` representation |
| `lecture7/cross-vs-bi-encoder.svg` | p.11 | Cross-encoder vs Bi-encoder side-by-side comparison |
| `lecture7/retrieve-and-rerank-flow.svg` | p.13 | Two-stage Retrieve-and-Rerank funnel (BM25 → monoBERT) |
| `lecture7/birch-pipeline.svg` | p.17 | Birch sentence-level scoring and BM25 interpolation |
| `lecture7/bert-maxp-pipeline.svg` | p.18 | BERT-MaxP sliding-window splitting and max aggregation |
| `lecture7/birch-maxp-cedr-comparison.svg` | p.19 | Three-column comparison of Birch / BERT-MaxP / CEDR-PARADE |
| `lecture7/multistage-ranking.svg` | p.21 | Full multi-stage ranking architecture |

## Lecture 8 — Cross-Encoder Reranking

| File | Slide | Contents |
|------|-------|----------|
| `lecture8/monot5-architecture.svg` | p.9 | monoT5 input template, encoder–decoder, true/false scoring |
| `lecture8/duot5-pipeline.svg` | p.12 | duoT5 pairwise comparison and win/loss aggregation |
| `lecture8/multistage-telescoping.svg` | p.16 | Three-stage telescoping pipeline with candidate-reduction funnel |
| `lecture8/knowledge-distillation.svg` | p.18 | Teacher–student distillation with hard/soft labels and loss |
| `lecture8/rankgpt-sliding-window.svg` | p.21 | RankGPT listwise reranking via sliding window |
| `lecture8/efficiency-accuracy-tradeoff.svg` | p.24 | Latency vs nDCG@10 scatter plot with Pareto frontier |

## Lecture 9 — Query and Document Expansion

| File | Slide | Contents |
|------|-------|----------|
| `lecture9/rm3-flow.svg` | p.8 | RM3 pseudo-relevance feedback processing flow |
| `lecture9/doc2query-concept.svg` | p.11 | doc2query document expansion by query prediction |
| `lecture9/deepct-architecture.svg` | p.15 | DeepCT per-token weight prediction with BERT |
| `lecture9/deepimpact-pipeline.svg` | p.20 | DeepImpact expansion + impact-score learning pipeline |
| `lecture9/mrr-comparison.svg` | p.21 | MRR@10 bar chart: BM25 / DeepCT / DeepImpact |
| `lecture9/expansion-design-space.svg` | p.24 | 2×2 design space: vocabulary expansion × weight learning |
