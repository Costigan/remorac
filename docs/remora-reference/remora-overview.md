# Remora: A Comprehensive Overview

Remora is a higher-order, rank-polymorphic array-processing programming language in the same class as APL and J. Designed by Olin Shivers, Justin Slepak, and Panagiotis Manolios at Northeastern University, it aims to combine Iverson's rank-polymorphic computational model with a static, dependent type system suitable for compilation to parallel hardware.

---

## Table of Contents

1. [Core Philosophy](#1-core-philosophy)
2. [Arrays, Shapes, and Ranks](#2-arrays-shapes-and-ranks)
3. [Concrete Syntax](#3-concrete-syntax)
4. [Functions and Cell Ranks](#4-functions-and-cell-ranks)
5. [The Frame/Cell Decomposition](#5-the-framecell-decomposition)
6. [Principal Frame and Cell Replication](#6-principal-frame-and-cell-replication)
7. [Reranking](#7-reranking)
8. [Primitive Operators](#8-primitive-operators)
9. [Conditional Code and Filtering](#9-conditional-code-and-filtering)
10. [Boxes and Ragged Arrays](#10-boxes-and-ragged-arrays)
11. [Static Type System](#11-static-type-system)
12. [Dynamic Semantics](#12-dynamic-semantics)
13. [Type Inference](#13-type-inference)
14. [Type Erasure and Translation to Explicit Iteration](#14-type-erasure-and-translation-to-explicit-iteration)
15. [Complete Examples](#15-complete-examples)
16. [Parallelism Model](#16-parallelism-model)
17. [References](#17-references)

---

## 1. Core Philosophy

Remora embodies three principles that together form its control story:

1. **Frame polymorphism** — any function defined for arguments of a given rank is automatically lifted to operate on arrays of any higher rank.
2. **Principal-frame cell replication** — when arguments have different frame shapes, the shorter frames are replicated to match the longest (principal) frame.
3. **Reranking** — programmers can adjust the frame/cell split to control which axes are iterated over.

The key insight: **the iteration space of a program is reified in the shape of its aggregate data structures**. No explicit loops or recursion constructs are needed — function application itself is the elimination form for arrays.

---

## 2. Arrays, Shapes, and Ranks

### 2.1 Fundamental Concepts

| Concept | Definition |
|---------|-----------|
| **Array** | A collection of data arranged in a hyper-rectangle of some dimensionality. *Every value in Remora is an array.* |
| **Atom** | The elements of an array: numbers, characters, booleans, or **functions** (making Remora higher-order). |
| **Rank** | The number of dimensions (axes) an array has. |
| **Shape** | A sequence (vector) of natural numbers giving the size of each dimension. |
| **Scalar** | An array of rank 0 with shape `[]`. |

### 2.2 Examples

```
Scalar:         17              rank 0, shape []
Vector:         [10 20 30]      rank 1, shape [3]
Matrix:         [[7 1 2]        rank 2, shape [2 3]
                 [2 0 5]]
3D Array:       [[[0 1]         rank 3, shape [2 2 2]
                  [1 0]]
                 [[1 0]
                  [0 1]]]
```

### 2.3 Key Properties

- The rank of an array equals the length of its shape vector.
- The product of all dimensions in the shape equals the number of atoms contained.
- A scalar (shape `[]`) contains exactly `1` atom (empty product = 1).

---

## 3. Concrete Syntax

Remora uses a Lisp-like s-expression syntax with both parentheses and square brackets.

### 3.1 Primitive Forms

**Array form** — for literal array constants:
```scheme
(array [d1 d2 ...] atom1 atom2 ...)
```
The shape is followed by atoms in row-major order. The number of atoms must equal the product of the dimensions.

```scheme
(array [2 3] 7 1 2 2 0 5)   ; 2x3 matrix
(array [] 17)                 ; scalar
```

**Frame form** — for constructing arrays from evaluated sub-expressions:
```scheme
(frame [d1 d2 ...] expr1 expr2 ...)
```
Each sub-expression evaluates to an array. All result arrays must have identical shape `[s1 ... sm]`. The final array has shape `[d1 ... dn s1 ... sm]`.

```scheme
(define v (array [3] 8 1 7))    ; v has shape [3]
(define m (frame [2] v v))       ; m has shape [2 3]
```

### 3.2 Syntactic Sugar

1. **Bare atoms** are scalar arrays: `17` is sugar for `(array [] 17)`.
2. **Square brackets** are vector frames: `[e1 e2 ... en]` desugars to `(frame [n] e1 e2 ... en)`.
3. **Nested square brackets** nest frames: `[[1 2] [3 4]]` is a matrix.

Thus the natural way to write arrays:
```scheme
[1 2 3]                       ; vector
[[7 1 2] [2 0 5]]             ; matrix
[[[0 1] [1 0]] [[1 0] [0 1]]] ; 3D array
```

All sibling elements in square-bracket notation must have matching shapes — ragged arrays are illegal (use boxes for those).

### 3.3 Other Syntax

```scheme
; Variable definition
(define name expr)

; Function definition with rank annotations
(define (fname [param1 rank1] [param2 rank2] ...)
  body-expr)

; Lambda with rank annotations
(λ ([param1 rank1] [param2 rank2] ...)
  body-expr)

; Function application
(func arg1 arg2 ...)

; Reranking shorthand
~(r1 r2 ... rn) func
; Desugars to: (λ ([v1 r1] ... [vn rn]) (func v1 ... vn))
```

---

## 4. Functions and Cell Ranks

Every function is defined to operate on arguments of a specific rank — its **cell rank**. The cell rank is annotated on each parameter.

```scheme
; + operates on two scalar (rank 0) arguments
(+ 3 4)  →  7

; A function on scalars:
(define (diff-square [x 0] [y 0])
  (- (* x x) (* y y)))

(diff-square 5 3)  →  16

; A function on vectors:
(define (vmag [v 1])
  (square-root (reduce/zero + 0 (square v))))

(vmag [3 4])  →  5
```

### 4.1 The `all` Keyword

A parameter can be tagged with `all` instead of a rank number, meaning the function consumes its **entire** argument as a single cell (scalar frame):

```scheme
(define (append [a all] [b all]) ...)
```

Functions like `append`, `length`, `reduce` (its data argument), and `rotate` (its data argument) use `all`.

---

## 5. The Frame/Cell Decomposition

This is the fundamental iteration mechanism of Remora.

### 5.1 The Basic Idea

An array of rank `r` and shape `[d1, ..., dr]` can be viewed as a **frame of cells** in `r+1` ways, by splitting the shape into a **frame prefix** and a **cell suffix**.

For a 2×3 matrix `[[7 1 2] [2 0 5]]` (shape `[2, 3]`):

| View | Frame shape | Cell shape | Description |
|------|-------------|------------|-------------|
| scalar frame | `[]` | `[2, 3]` | One 2×3 matrix |
| vector frame | `[2]` | `[3]` | Two 3-vectors (the rows) |
| matrix frame | `[2, 3]` | `[]` | Six scalars |

### 5.2 Automatic Lifting

When a function expecting rank-`r` cells is applied to a rank-`r'` array (where `r' ≥ r`):

1. The last `r` dimensions of the argument's shape become the **cell shape**.
2. The remaining `r' − r` dimensions become the **frame shape**.
3. The function is applied independently to each cell in the frame.
4. All results (which must have identical shape) are collected back into the frame.

```
(define (vmag [v 1]) ...)    ; expects rank-1 cells (vectors)

(vmag [[1 2 2]               ; matrix → vector frame of vector cells
       [2 3 6]])
→ [3 7]                       ; two scalar results in a vector frame
```

Applying `vmag` to a 6D array treats it as a 5D frame of 1D cells, producing a 5D scalar result.

---

## 6. Principal Frame and Cell Replication

When a function has multiple arguments, their frames must be **compatible**.

### 6.1 Frame Agreement

The cell shapes are stripped from the suffix of each argument's shape. The remaining prefixes are the **frames**. For a well-formed application, the frames must be prefix-orderable: one must be a prefix of all others. The longest frame is the **principal frame**.

### 6.2 Cell Replication

If an argument's frame is shorter than the principal frame, the argument is **replicated** into the missing dimensions.

```
(+ [10 20] [[8 1 3]            ; vector (shape [2]) + matrix (shape [2, 3])
            [5 0 9]])
→ [[18 11 13]                   ; [2] replicated to [2, 3]
   [25 20 29]]
```

Here, `+` expects scalar cells (rank 0). Frame shapes: `[2]` and `[2, 3]`. Principal frame = `[2, 3]`. The vector's frame `[2]` is replicated to `[2, 3]`: for position `[i, j]`, the index `j` is dropped when fetching from the vector.

**Adding a scalar to any array** adds it element-wise (scalar frame `[]` is replicated to match):

```
(+ 10 [7 1 4])  →  [17 11 14]
```

### 6.3 Functions in Function Position

The function position can itself be an array of functions (with scalar cells). It participates in principal-frame determination:

```scheme
(define m [[square   square-root]
           [add1     sub1]])
(m 9)  →  [[81 3]
           [10 8]]
```

This gives Remora MIMD capability on top of its SIMD model.

---

## 7. Reranking

Remora's default frame/cell split sometimes doesn't match the programmer's intent. **Reranking** adjusts how arguments are cut into cells.

### 7.1 The Problem

Adding a vector to a matrix adds the vector element-wise to each row:
```
(+ [10 100] [[1 2] [3 4]])  →  [[11 102] [13 104]]
```
What if we want to add to each **column** instead?

### 7.2 The Solution: η-expand with Different Cell Ranks

```scheme
((λ ([x 1] [y 1]) (+ x y))    ; + lifted to rank-1 cells
 [10 100]
 [[1 2] [3 4]])
→ [[11 102]
   [13 104]]
```

By wrapping `+` in a lambda with rank-1 parameters, the matrix is viewed as a frame of vector rows, and the vector is replicated across those rows. Execution steps:

```
[(+ [10 100] [1 2])            ; each row gets the full vector
 (+ [10 100] [3 4])]
→ [[(+ 10 1) (+ 100 2)]
   [(+ 10 3) (+ 100 4)]]
→ [[11 102]
   [13 104]]
```

### 7.3 Reranking Shorthand

The tilde syntax `~()` provides syntactic sugar:
```
~(r1 r2 ... rn) func
```
For example:
```scheme
(~(1 1) + v m)    ; same as the η-expanded version above
```

### 7.4 Reranking for Axis Control

**Append side-by-side** instead of top-to-bottom:
```scheme
(append m1 m2)           ; stacks vertically
(~(1 1) append m1 m2)    ; appends side-by-side (row-wise)
```

**Reduce across columns** instead of rows:
```scheme
(reduce + [[0 1 2] [0 10 100]])     ; → [0 11 102]  (sum rows)
(~(0 1) reduce + [[0 1 2] [0 10 100]])  ; → [3 110]  (sum each row)
```

---

## 8. Primitive Operators

### 8.1 Reduce

Maps an associative binary operator over the initial dimension:

```scheme
; Sum elements of a vector
(reduce + [1 4 9 16])  →  30

; Sum columns of a matrix
(reduce + [[1 2 3] [10 20 30] [100 200 300]])
→ [111 222 333]

; Product with zero base
(reduce/zero * 1 [1 2 3 4 5])  →  120  ; 5!
```

The combining operator must be associative (`α × α → α`). Reduce is **parallel**.

### 8.2 Fold

Uses a general folding operator (`α × β → β`) for serial accumulation:

```scheme
(fold (λ ([v 1] [sum 0]) (+ sum (vmag v)))
      0
      [[1 2 2] [2 3 6]])
→ 10
```

Fold is **serial** (has a loop-carried dependency).

### 8.3 Scan (Prefix Sums)

```scheme
; Interior scan (includes current element)
(iscan + [2 10 5])  →  [2 12 17]

; Exterior scan (excludes current element; one element longer)
(scan/zero + 0 [2 10 5])  →  [0 2 12 17]
```

**Reduce/Scan/Fold/Trace family**: Remora provides 3 reduces, 8 scans, 2 folds, and 2 traces. The variants differ in whether they require a non-empty initial dimension, include/exclude the current element, and operate left-to-right or right-to-left.

### 8.4 iota

Generates counting arrays:

```scheme
(iota [5])        →  [0 1 2 3 4]
(iota [2 3])      →  [[0 1 2] [3 4 5]]

; Factorial using iota
(define (fact [n 0])
  (reduce/zero * 1 (+ 1 (iota [n]))))

(fact [0 3 5 10])  →  [1 6 120 3628800]
```

Variants: `iota/v` (produces vector), `iota/s` (shape given as type index), `iota/w` (shape witness from another array).

### 8.5 append, length, rotate

```scheme
; append along leading axis
(append [[0 1] [2 3]] [[10 20] [30 40]])
→ [[0 1] [2 3] [10 20] [30 40]]

; length of leading dimension
(length [[1 2 3] [4 5 6]])  →  2

; rotate — array and vector of per-axis rotation amounts
(rotate [2 3 4 5 11] [2])  →  [4 5 11 2 3]

; Lifted rotation: rotate by each rotation vector
(rotate [2 3 5 7] [[0] [1] [2]])
→ [[2 3 5 7] [3 5 7 2] [5 7 2 3]]
```

### 8.6 indices-of

Produces coordinates for each position in an array (reifies the iteration space):

```scheme
(indices-of [[0 1 2] [3 4 5]])
→ [[[0 0] [0 1] [0 2]]
   [[1 0] [1 1] [1 2]]]
```

### 8.7 shape, ravel, with-shape, reshape

```scheme
(shape [[1 2] [3 4]])  →  [2 2]        ; returns shape as vector
(ravel [[1 2] [3 4]])  →  [1 2 3 4]    ; flatten to vector
(with-shape [0 0 0 0] 5)  →  [5 5 5 5] ; replicate to match shape
```

### 8.8 Arithmetic and Comparison

All arithmetic operators (`+`, `-`, `*`, `/`, `expt`) operate on scalar cells and lift via rank polymorphism. Comparison operators (`>`, `<`, `=`, `zero?`) similarly lift.

---

## 9. Conditional Code and Filtering

### 9.1 Traditional Conditionals

Remora provides `if` and `cond` forms (from Lisp), but these are barriers to parallelism:

```scheme
(define (fact [n 0])
  (if (zero? n) 1
      (* n (fact (- n 1)))))
```

### 9.2 Data-Parallel Filtering

**filter** selects sub-arrays from the leading dimension:

```scheme
; Filter positive numbers from a vector
(filter (> nums 0) nums)          ; nums = [0 5 -7 -22 91 100]
→ [5 91 100]

; Filter rows from a matrix
(filter [#t #f #f #t #t]
        [[0 1 2] [16 17 18] [9 10 11] [22 23 24] [96 97 98]])
→ [[0 1 2] [22 23 24] [96 97 98]]
```

**select** picks between two values based on booleans:

```scheme
(select #t 3 4)  →  3
(select [#t #f #f #t #t]
        [0 1 2 3 4]
        [20 21 22 23 24])
→ [0 21 22 3 4]
```

**partition** splits data into two collections based on a boolean vector.

---

## 10. Boxes and Ragged Arrays

A **box** wraps an arbitrary array as an atom, enabling arrays of ragged (irregular) data.

```scheme
(box [4 5 6])                           ; box a vector
(unbox contents (box [4 5 6])           ; extract and use
  (sum contents))
→ 15
```

Boxes type as **dependent sums** (`Sigma` types), existentially hiding dimensions. This enables:

- Vectors of strings (each string is a character vector of unknown length)
- Results of operations like `filter` and `iota` where the result shape depends on run-time data
- Ragged arrays where sibling elements may have different shapes

```scheme
; Type of a list of 20 strings:
(Arr (Sigma ((len Dim)) (Arr Char (Shp len))) (Shp 20))

; Type of filter result — hides the unknown leading dimension:
(-> ((Arr Bool d) (Arr t (++ (Shp d) s)))
    (Arr (Sigma ((k Dim)) (Arr t (++ (Shp k) s))) (Shp)))
```

---

## 11. Static Type System

Remora has a formal static type system using dependent types in the style of Dependent ML (restricted to Presburger arithmetic).

### 11.1 Type Structure

**Kinds**: `Atom` (scalar types) or `Array` (aggregate types).

| Type Form | Syntax | Kind | Description |
|-----------|--------|------|-------------|
| Base type | `B` | Atom | e.g., `Num`, `Bool`, `Char` |
| Array | `(Arr τ ι)` | Array | Array with atom type `τ` and shape `ι` |
| Function | `(-> (τ ...) τ')` | Atom | Function; args/results must be Array-kinded |
| Universal | `(Forall ((x k) ...) τ)` | Atom | Parametric polymorphism over types |
| Dependent product | `(Pi ((x γ) ...) τ)` | Atom | Polymorphism over indices (dimensions/shapes) |
| Dependent sum | `(Sigma ((x γ) ...) τ)` | Atom | Existential: hidden shape info (boxes) |

### 11.2 Type Indices

Indices inhabit sorts: `Dim` (individual natural numbers) or `Shape` (sequences of naturals).

| Index Form | Sort | Description |
|-----------|------|-------------|
| `n` | Dim | Natural number literal |
| `(+ ι ...)` | Dim | Sum of Dims |
| `(Shp ι ...)` | Shape | Shape from Dims (each element must be Dim) |
| `(++ ι ...)` | Shape | Concatenation of Shapes |
| `x` | Dim or Shape | Index variable |

The theory of indices is the free monoid on ℕ: `Shp` (empty shape) is identity for `++`, `0` is identity for `+`, both are associative, with equidivisibility.

### 11.3 Example Types

```
; Scalar addition
+ : (-> ((Arr Num (Shp)) (Arr Num (Shp))) (Arr Num (Shp)))

; Vector magnitude: works on vectors of any length
vmag : (Pi ((n Dim))
        (-> ((Arr Float (Shp n)))
            (Arr Float (Shp))))

; Major-axis mean: works on leading axis of any length
major-mean : (Pi ((c Shape) (n Dim))
              (-> ((Arr Float (++ (Shp n) c)))
                  (Arr Float c)))

; append: any element type, any remainder shape, any leading dims
append : (Pi ((c Shape) (m Dim) (n Dim))
          (Forall ((a Atom))
            (-> ((Arr a (++ (Shp m) c))
                 (Arr a (++ (Shp n) c)))
                (Arr a (++ (Shp (+ m n)) c)))))

; filter: existentially hides result leading dimension
filter : (-> ((Arr Bool d)
             (Arr t (++ (Shp d) s)))
            (Arr (Sigma ((k Dim)) (Arr t (++ (Shp k) s))) (Shp)))

; head: extracts first sub-array; requires non-zero leading dimension
head : (-> ((Arr t (++ (Shp (+ 1 d)) s))) (Arr t s))

; reduce: collapses leading dimension
reduce : (-> ((Arr (-> ((Arr t s) (Arr t s)) (Arr t s)) (Shp))
             (Arr t (++ (Shp (+ 1 d)) s)))
            (Arr t s))
```

### 11.4 Key Typing Rules

**T-APP (Function Application):** The most important rule. Given:
- Function type: `(Arr (-> ((Arr τ ι) ...) (Arr τ' ι')) ιf)`
- Arguments: each of type `(Arr τ (++ ιa ι))` where `ιa` is that argument's frame

Compute the **principal frame** `ιp` = the join (under prefix ordering) of `ιf` and all `ιa`. Result type: `(Arr τ' (++ ιp ι'))`.

**T-BOX / T-UNBOX:** Boxes use dependent sums. Unboxing is a let-like form that opens the existential, with the constraint that hidden index information cannot leak into the result type.

**T-TAPP / T-IAPP:** Type and index application substitute arguments and prefix the function frame unchanged.

### 11.5 Type Equivalence

Type equivalence (`τ ∼ τ'`) is α-equivalence augmented with index equivalence under the algebraic theory of naturals and shapes. Two arrays are equivalent if their atom types are equivalent and their shapes are provably equal under the free monoid theory.

### 11.6 Type Soundness

The type system satisfies **Progress** and **Preservation**:

- **Progress**: A well-typed non-value can always take a step.
- **Preservation**: If a well-typed term steps, its type is preserved.

This means "array shape" errors cannot occur at run time in a well-typed program.

---

## 12. Dynamic Semantics

The dynamic semantics is **type-driven** — the type information determines how arrays are decomposed into frames and cells during function application.

### 12.1 Values

Values are arrays of atomic values (including closures, primitive operators, boxes). An array value is written `(array (n ...) v ...)`.

### 12.2 Key Evaluation Rules

**Function application** `(ef ea ...)`:
1. Evaluate `ef` to an array of functions.
2. Evaluate each `ea` to an array.
3. Compute the cell shapes from the function's type.
4. Split each argument into a frame of cells.
5. Replicate cells as needed to the principal frame.
6. Apply the function atom to corresponding cells from each argument.
7. Collect results into the principal frame.

**Box construction** `(box ι ... e τ)`: Evaluate `e` to a value `v`; wrap as box atom.

**Unbox** `(unbox (xi ... xe es) eb)`: Evaluate `es` to a box value; bind index variables `xi` to the box's indices and `xe` to its contained array; evaluate body `eb`.

### 12.3 Partially Erased Semantics

Remora also has a **partially erased** variant where full type annotations are reduced to only the shape information needed for dynamic dispatch. A bisimulation theorem proves that fully-typed and partially-erased computations stay in lock step.

---

## 13. Type Inference

Explicit Remora types are verbose. Bidirectional typing with a novel constraint solver handles inference.

### 13.1 Bidirectional Typing

Uses two judgments:
- **Synthesis** (`Γ ⊢ e ⇒ τ`): The type is inferred from the term.
- **Checking** (`Γ ⊢ e ⇐ τ`): The term is checked against a given type.

Application synthesis uses a specialized judgment that decomposes argument arrays, identifies the principal frame, and generates/solves constraints on the unknown dimensions.

### 13.2 Constraint Solver

The solver works over string equations modulo theories (mixed-prefix fragment):
- Free monoid on ℕ for shapes (with `++` and `Shp`)
- Presburger arithmetic for dimensions (with `+`)

Constraints arise when the frame/cell split of an argument must be determined — the solver must find how to partition each argument shape into a frame prefix and cell suffix satisfying the function's type.

### 13.3 Elaboration

Bidirectional typing elaborates implicitly-typed surface programs to fully-typed Core Remora. Elaboration soundness ensures the elaborated program has the same control structure as the surface program intended.

---

## 14. Type Erasure and Translation to Explicit Iteration

### 14.1 Type Erasure

Removes detailed type annotations from runtime representations (closures, arrays), moving the description of expected input shapes from dynamic closures to static call sites. The residual types characterise precisely the information needed by the dynamic semantics.

### 14.2 Explicit Iteration

Translates rank-polymorphic function application into explicit iteration:
- Frame/cell decomposition becomes nested loops.
- Cell replication becomes broadcasting.
- The principal frame becomes the loop nest structure.
- Reductions become parallel reduce operations.
- Scans become prefix-sum operations.

This two-phase translation (erase types → make iteration explicit) connects the high-level rank-polymorphic semantics to conventional rank-monomorphic target languages.

---

## 15. Complete Examples

### 15.1 Vector Magnitude

```scheme
(define (vmag [v 1])
  (square-root (reduce/zero + 0 (square v))))

(vmag [3 4])              →  5
(vmag [[1 2 2] [2 3 6]])  →  [3 7]
```

### 15.2 Mean, Variance, Covariance

```scheme
(define (mean [xs 1])
  (/ (reduce + xs) (length xs)))

(define (variance [xs 1])
  (mean (square (- xs (mean xs)))))

(define (covariance [xs 1] [ys 1])
  (mean (* (- xs (mean xs)) (- ys (mean ys)))))
```

### 15.3 1D Convolution

```scheme
(define (vector-convolve [v 1] [w 1])
  (reduce + (* (rotate v (indices-of w)) w)))
```

This works because:
1. `indices-of w` gives `[[0] [1] ... [n-1]]`.
2. `rotate v [[0] [1] ... [n-1]]` produces a matrix of rotated copies of `v`.
3. `* w` multiplies each row by the corresponding weight.
4. `reduce +` sums each column, producing the convolution.

### 15.4 Matrix Multiplication

```scheme
; Vector × Matrix
(define (v*m [v 1] [m 2])
  (reduce/zero + 0 (* v m)))

; Matrix × Matrix
(define (m*m [a 2] [b 2])
  (v*m a b))
; equivalently: (define m*m ~(2 2) v*m)

; Single-line version
(define (m*m [a 2] [b 2])
  (~(0 0 2) reduce/zero + 0 (~(1 2) * a b)))
```

How `v*m a b` works:
1. `a` (shape `[n, p]`) → viewed as vector frame `[n]` of vector cells `[p]`.
2. `b` (shape `[p, q]`) → `v*m` expects matrix cells; `b` is a scalar frame containing one matrix.
3. `b` is replicated `n` times; each row of `a` is multiplied by `b`.
4. Inside `v*m`, `(* v m)` lifts `*` (scalars) with principal frame `(++ [n] [p q])`; `v` replicated across columns of `m`.
5. `reduce/zero + 0` sums across columns.

### 15.5 Matrix × Vector (via reranking)

Adding a vector to each column of a matrix:
```scheme
(~(1 1) + v m)
; distributes + across rows: [(+ v row1) (+ v row2) ...]
; inside each, + adds element-wise
```

### 15.6 Polynomial Evaluation (Three Ways)

**Simple (quadratic multiplies):**
```scheme
(define (poly-eval [coeffs 1] [x 0])
  (reduce/zero + 0
    (* coeffs (expt x (iota [(length coeffs)])))))
```

**Horner's Rule (serial, linear multiplies):**
```scheme
(define (poly-eval [coeffs 1] [x 0])
  (fold-right (λ ([coeff 0] [acc 0]) (+ coeff (* x acc)))
              0
              coeffs))
```

**Parallel (linear multiplies, parallelisable):**
```scheme
(define (poly-eval [coeffs 1] [x 0])
  (reduce/zero + 0
    (* coeffs
       (open-scan/zero * 1 (with-shape coeffs x)))))
```

---

## 16. Parallelism Model

Remora is a **map/reduce architecture**:

1. **Every function application is an implicit parallel map.** When `(f collection)` is evaluated, `f` is applied independently to each cell — all invocations are parallel by default.

2. **Reductions are explicit parallel combine.** `reduce` with an associative operator can be executed in `O(log n)` parallel time.

3. **Fold/scan expose loop-carried dependencies.** When the programmer needs serial computation, they use `fold`, `trace`, or serial variants — making the dependency explicit.

4. **No heroic compiler analysis needed.** The compiler doesn't need to prove independence between loop iterations; parallelism is explicit in the notation:
   - Default case = parallel (map)
   - Marked case = serial (fold)

5. **Good hardware utilisation.** For `n` data items on `p` processors:
   - If `n < p`: `O(log n)` time (effectively constant for real-world `n`).
   - If `n > p`: near-`p` speedup, which is the best possible.

---

## 17. References

| Reference | Source | Content |
|-----------|--------|---------|
| Remora Tutorial Draft | Shivers, Slepak, Manolios (2019) | Programming introduction and examples |
| arXiv:1912.13451 | Same as above, published version | Citation metadata |
| Semantics of Rank Polymorphism | Slepak, Shivers, Manolios (2019), arXiv:1907.00509 | Formal dynamic/static semantics, type soundness |
| Slepak Dissertation | Justin Slepak (2020) | Full type system, bidirectional inference, constraint solving, type erasure, explicit iteration |
