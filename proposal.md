# Binary DNA Research Plan

This document outlines a phased, reproducible research plan for the Binary DNA toolkit, organized as "Do this / Achieve this" for each phase.

## Phase 0 – Environment & Reproducibility

- **Do**
  - Set up a clean Python environment and install `requirements.txt`.
  - Standardize a results layout, e.g. `results/{system,github}/…` plus a single `config.yaml`.
  - Add a small “smoke corpus” config (10–20 binaries) and a script to run the full pipeline end-to-end.
- **Achieve**
  - One-command execution (e.g. `python binary_dna.py system --output-dir results/system`).
  - Fixed folder structure used by all experiments.

## Phase 1 – Corpus Design & Extraction

- **Do**
  - Define 2–3 corpora: `coreutils_system`, `mixed_system` (bash/gcc/python), `github_compiled`.
  - Run extraction (`extract` / `system` / `github`) and persist `Binary` objects and vocab.
- **Achieve**
  - Stable opcode sequences for each binary with metadata (size, category, compiler where known).
  - Shared `vocab.json` and corpus summaries (counts, lengths) reused in later phases.

## Phase 2 – Frequency, Zipf & N-gram Structure

- **Do**
  - Run frequency and Zipf analysis on each corpus.
  - Compute n-gram distributions (n=1–5), entropy, and entropy rate; build random/shuffled baselines.
- **Achieve**
  - Plots and JSON: Zipf curves, α estimates, entropy-rate vs. n, top n-grams.
  - Testable statements about “natural-language-like” behavior and predictability of opcode sequences.

## Phase 3 – Redundancy, Compression & Motifs

- **Do**
  - Compute LZ complexity and compression ratios (zlib/lzma) per binary.
  - Run motif discovery (k-mers, k=4–12) and positional analysis of function prologues/epilogues.
- **Achieve**
  - Evidence of high redundancy (strong compression) and recurrent compiler idioms (motifs).
  - Positional heatmaps showing conserved structure at function boundaries.

## Phase 4 – Similarity, Clustering & Categories

- **Do**
  - Build NCD distance matrices and n-gram TF-IDF cosine similarity matrices.
  - Perform hierarchical clustering and PCA/UMAP; label binaries by tool category, size, compiler.
- **Achieve**
  - Dendrograms and embeddings where similar tools cluster together.
  - Quantitative cluster quality metrics (e.g. category purity, silhouette scores).

## Phase 5 – Information Geometry & Manifold Thickness

- **Do**
  - Run sliding-window entropy and mutual information decay analyses.
  - Estimate “space coverage” and intrinsic dimensionality from n-gram vectors and PCA.
- **Achieve**
  - MI decay curves and dimensionality estimates showing how “thin” the program manifold is.
  - Comparisons to random baselines supporting the “tiny structured manifold” hypothesis.

## Phase 6 – LLM Connection & Reporting

- **Do**
  - (Optional) Train a simple opcode language model and measure perplexity vs. entropy/MI metrics.
  - Use `visualization/report.py` to generate a unified HTML report per corpus and a concise written summary of hypotheses vs. results.
- **Achieve**
  - Cohesive narrative linking empirical results to why LLMs perform well on code.
  - Paper-ready figures, tables, and a single reproducible pipeline that others can run.

