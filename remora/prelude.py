"""Prelude source helpers for Remora Dense Core."""

from __future__ import annotations

from pathlib import Path


PRELUDE_PATH = Path(__file__).parent.parent / "stdlib" / "prelude.rem"


def prelude_source() -> str:
    return "\n".join(prelude_definition_sources())


def prelude_definition_sources() -> list[str]:
    return [
        line.strip()
        for line in PRELUDE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("def ")
    ]


def lisp_prelude_definition_sources() -> list[str]:
    return [
        "(define (add [x y]) (+ x y))",
        "(define (sub [x y]) (- x y))",
        "(define (mul [x y]) (* x y))",
        "(define (div [x y]) (/ x y))",
        "(define (neg [x]) (- 0 x))",
        "(define (id [x]) x)",
        "(define (const [a b]) a)",
        "(define (sum [xs]) (fold + 0.0 xs))",
        "(define (product [xs]) (fold * 1.0 xs))",
        "(define (scale [s xs]) (map (* s) xs))",
        "(define (dot [a b]) (sum (map * a b)))",
        "(define (max [a b]) (if (< a b) b a))",
        "(define (min [a b]) (if (< a b) a b))",
        "(define (abs [x]) (if (< x 0) (- 0 x) x))",
        "(define (any [xs]) (fold (lambda (a x) (|| a x)) #f xs))",
        "(define (all [xs]) (fold (lambda (a x) (&& a x)) #t xs))",
        "(define pi 3.141592653589793)",
        "(define (signum [x]) (select (> x 0) 1 (select (< x 0) -1 0)))",
        "(define (positive? [x]) (> x 0))",
        "(define (negative? [x]) (< x 0))",
        "(define (zero? [x]) (== x 0))",
        "(define (even? [n]) (== (modulo n 2) 0))",
        "(define (odd? [n]) (!= (modulo n 2) 0))",
    ]


def prelude_definition_sources_for_syntax(syntax: str) -> list[str]:
    if syntax == "lisp":
        return lisp_prelude_definition_sources()
    return prelude_definition_sources()


def with_prelude(source: str) -> str:
    source = _strip_leading_ignored_lines(source)
    prelude = prelude_source()
    return f"{prelude}\n{source}" if source else prelude


def _strip_leading_ignored_lines(source: str) -> str:
    lines = source.splitlines()
    while lines and (not lines[0].strip() or lines[0].strip().startswith("--")):
        lines.pop(0)
    return "\n".join(lines).strip()
