# Remora Reference Bundle

This directory contains local copies of the Remora language references needed to implement the MLIR prototype. Use the `.txt` files for `rg` searches and the `.pdf` files when equations, figures, or notation are hard to read in extracted text.

## Files

| Local file | Source URL | Use for |
|---|---|---|
| `remora-tutorial-draft.pdf` / `.txt` | https://www.ccs.neu.edu/home/shivers/papers/remora-tutorial-draft.pdf | Best first read. Covers the user-facing rank-polymorphic programming model, arrays, shapes, frames, cells, lifting, reranking, and static typing motivation. |
| `intro-rank-polymorphic-programming-remora.pdf` / `.txt` | https://arxiv.org/abs/1912.13451 | Published arXiv entry for the tutorial. Use for citation metadata and the same conceptual model as the tutorial draft. |
| `semantics-of-rank-polymorphism.pdf` / `.txt` | https://arxiv.org/abs/1907.00509 | Formal dynamic/static semantics. Use for precise frame/cell typing rules, rank-polymorphic application, shape soundness, and type-driven execution. |
| `slepak-dissertation.pdf` / `.txt` | https://ccs.neu.edu/~jrslepak/Dissertation.pdf | Deep reference for Remora's type system, implicit frame polymorphism, shape inference, bidirectional typing, constraint solving, and type erasure toward explicit iteration. |

## Fast Lookup Guide

| Implementation task | Start here | Search terms |
|---|---|---|
| Parser syntax and examples | `remora-tutorial-draft.txt` | `array form`, `frame form`, `define`, `lambda`, `rank`, `shape` |
| Array literals and shape consistency | `remora-tutorial-draft.txt` | `square-bracket`, `different shapes`, `well-defined shape`, `array literal` |
| Frame/cell decomposition | `remora-tutorial-draft.txt`, then `semantics-of-rank-polymorphism.txt` | `Functions operate on "cells"`, `frame of cells`, `cell suffix`, `frame prefix` |
| Rank-polymorphic application typing | `semantics-of-rank-polymorphism.txt` | `typing rule`, `frame shape`, `iteration space`, `cells` |
| Prototype type checker design | `slepak-dissertation.txt` | `bidirectional typing`, `implicitly typed Remora`, `base environment`, `shape inference` |
| Lowering to explicit iteration | `slepak-dissertation.txt` | `type erasure`, `explicit iteration`, `rank-monomorphic`, `control structure` |
| Built-ins such as `iota` | `slepak-dissertation.txt`, `remora-tutorial-draft.txt` | `iota`, `primitive operations`, `base environment` |

## Notes for Agents

- These references describe the full Remora language. The implementation plan intentionally starts with a smaller vertical slice: `iota`, unary `map`, and scalar `fold`.
- Prefer the tutorial for user-facing semantics and examples.
- Prefer the semantics paper when deciding whether a type or shape rule is correct.
- Prefer the dissertation when implementing type inference or translating rank-polymorphic behavior into explicit lower-level iteration.
- Ignore unrelated projects named Remora, including the ocean model, C++ linear algebra library, nanopore package, and R package.
