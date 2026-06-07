Continue implementing the Full Remora plan from docs/FULL_REMORA_PLAN.md.
Read that document first — it has a progress tracker in Section 13 with
checkboxes.

CURRENT STATE:
- Dense Core is complete (530 tests passed, 1 skipped).
- All 30 Dense Core checkboxes in Section 13 are checked.
- An MLIR builder API exists: remora/lowering/_builder_emitter.py (scalar ops),
  remora/lowering/_builder_ops.py (tensor/views), remora/lowering/_gpu_builder.py
  (simple GPU scaffolds).
- The LLVM descriptor-ABI GPU path in gpu_lowering.py still uses text generation
  (ir.Operation.create doesn't round-trip llvm.func/nvvm.* dialect attributes).
- Two syntaxes exist: the default ML-like syntax (.remora files) and a planned
  Lisp syntax (Phase 1, not yet implemented).
- A user guide exists at docs/USER_GUIDE.md with a syntax reference and a
  Dense Core vs. full Remora mapping table.
- The full Remora tutorial draft is at docs/remora-reference/remora-tutorial-draft.txt
  (1882 lines) — this is the specification for what full Remora should do.

IMMEDIATE TASK — Phase 1: Lisp syntax reader

Implement remora/lisp_reader.py — a Lark grammar for s-expressions that
desugars to the existing Dense Core AST. This is purely syntactic: no new
semantics, no type system changes. The reader produces the same AST nodes
(Program, AppExpr, LetExpr, IfExpr, LambdaExpr, etc.) that the existing
remora/parser.py produces, so the compiler, typechecker, and lowering
pipeline work unchanged.

Syntax mapping (Lisp → current ML AST):

  Literals:
    42                                    → 42          (IntLit)
    3.14                                  → 3.14        (FloatLit)
    #t / #f                               → true / false (BoolLit)
    [1 2 3]                               → [1, 2, 3]   (ArrayLit)

  Let and if:
    (:: x 5 (+ x 1))                      → let x = 5 in x + 1
    (if (< 1 2) 10 20)                    → if 1 < 2 then 10 else 20

  Arithmetic / comparison:
    (+ 1 2)                               → 1 + 2
    (< x 5)                               → x < 5
    (&& a b)                              → a && b
    (|| a b)                              → a || b

  Map and fold:
    (map (+ 2) xs)                        → map (+ 2) xs
    (map (lambda (x) (* x 2)) xs)         → map (\x -> x * 2) xs
    (fold + 0 xs)                         → fold (+) 0 xs
    (fold (lambda (acc x) (+ acc x)) 0 xs)→ fold (\acc x -> acc + x) 0 xs

  Iota and views:
    (iota 5)                              → iota 5
    (iota 2 3)                            → iota 2 3
    (reverse xs)                          → reverse xs
    (transpose m)                         → transpose m
    (reshape xs [2 2])                    → reshape xs [2, 2]
    (ravel m)                             → ravel m
    (take 2 xs)                           → take 2 xs
    (drop 2 xs)                           → drop 2 xs

  Indexing:
    (index xs 0)                          → xs[0]
    (index xs 0 1)                        → xs[0, 1]

  Shape/rank:
    (shape xs)                            → shape xs
    (rank xs)                             → rank xs

  Function definitions:
    (define (double [x]) (* x 2))         → def double x = x * 2
    (define (add [x y]) (+ x y))          → def add x y = x + y
    (define xs [1 2 3])                   → def xs = [1, 2, 3]
    (define (mean [xs 1]) (/ (reduce + xs) (length xs)))  -- for Phase 2+

  Lambda:
    (lambda (x) body)                     → \x -> body
    (lambda (x y) body)                   → \x y -> body
    (λ (x) body)                          → same (Unicode lambda)

Implementation notes:
- The existing parser is at remora/parser.py. It uses a Lark grammar
  (remora/grammar.lark, 97 lines) and produces AST nodes defined in
  remora/ast_nodes.py.
- The Lisp reader should be a SEPARATE file (remora/lisp_reader.py) with its
  own Lark grammar. It should NOT modify the existing parser or grammar.
- The Lisp reader produces the same AST nodes. Reuse the AST node classes
  from remora/ast_nodes.py and remora/types.py.
- The Lisp reader must handle operator sections: (+ 2) is a left section (the
  operator is partially applied with 2 on the right). (2 +) is a right section.
  These map to the existing OperatorFunc, LeftSection, RightSection AST
  nodes.
- Square brackets in function definition parameters denote cell ranks:
  (define (f [x 0]) body) means parameter x has cell rank 0. These ranks
  are currently ignored by Dense Core but should be stored in the AST for
  future use (Phase 2). Store them as annotations on Param nodes.
- Comments in Lisp syntax use ; to end of line.
- The existing grammar uses names like NAME, INT, FLOAT, BOOL for tokens.
  The Lisp grammar can define its own token rules or reuse the existing ones.

Integration:
- Add a parse_lisp(source: str) -> Program function
- Add --syntax lisp flag to CLI (remora/cli.py) and REPL (remora/repl.py)
- The :syntax lisp REPL command switches the REPL to Lisp syntax
- Default syntax remains ML (backward compatible)
- Both syntaxes can coexist; the :load REPL command should auto-detect
  syntax from file extension (.remora vs future .lisp) or a flag

Tests to add:
- tests/test_lisp_reader.py with parametrized tests for every form above
- Verify that parse_lisp(source) produces identical AST to parse_program
  for semantically equivalent programs
- Test operator sections: (+ 2), (2 +), (< 5), (&& true)
- Test nested expressions: (fold + (:: x 5 (+ x 1)) (iota 10))
- Test multi-definition programs: (define ...) (define ...) expr
- Test error cases: mismatched parens, unknown keywords, invalid syntax
- Test Unicode lambda: (λ (x) (* x 2))

FILES TO READ FIRST:
- docs/FULL_REMORA_PLAN.md (section 13 progress tracker, Phase 1 checklist)
- remora/parser.py (existing parser, see how parse_program works)
- remora/grammar.lark (existing grammar, 97 lines)
- remora/ast_nodes.py (AST node classes)
- remora/types.py (types, StaticDim, ArrayType, ScalarType)
- remora/cli.py (CLI entry point, see how --target etc. are handled)
- remora/repl.py (REPL, see how :target, :load etc. are handled)
- docs/USER_GUIDE.md (Dense Core vs full Remora mapping table)

COMMAND TO VERIFY WORK:
  cd /e/projects/remorac && uv run pytest -q
