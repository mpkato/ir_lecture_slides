# Figures (SVG)

Redrawn vector figures for Lectures 7–9. Each file is a self-contained SVG
(no external fonts or scripts required) that follows the slide color scheme
(blue `#2B87B8` / orange `#E87E2E`). They can be embedded directly in Marp
slides, e.g.:

```markdown
![w:560](figures/lecture7/monobert-architecture.svg)
```

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
