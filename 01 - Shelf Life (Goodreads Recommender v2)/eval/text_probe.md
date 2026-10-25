# Description embeddings vs co-rating neighbours (top 20)

Model: BAAI/bge-small-en-v1.5 reduced to 128 dims. 140,040 of 144,112 books have a description (97.2%).

## Agreement

- Of each book's top 20 text neighbours, **10.9%** are also among its top 20 co-rating neighbours (median 5%; 139,806 books with both).
- Over co-rating neighbour pairs, text cosine vs co-rating similarity: Spearman **0.356** (245,206 pairs). Mean text cosine of co-rating neighbours 0.239 vs -0.000 for random pairs.

## Where co-rating evidence is thin

| Raters | Books | Fewer than 20 co-rating neighbours | Has a description | Median co-rating neighbours |
|---|---|---|---|---|
| 0–99 | 43,903 | 0.6% | 96.1% | 50 |
| 100–299 | 59,305 | 0.0% | 96.6% | 50 |
| 300–999 | 27,758 | 0.0% | 98.8% | 50 |
| 1,000+ | 13,146 | 0.0% | 99.7% | 50 |

## What the neighbours are

| Share of top-k neighbours | Text | Co-rating |
|---|---|---|
| same parent genre | 53.6% | 69.5% |
| same author | 12.5% | 30.1% |
| same series | 13.7% | 21.1% |
| median raters of a neighbour | 161 | 336 |
