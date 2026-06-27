"""Generated differential test program generator for a narrow dense subset.

Produces valid Remora source programs exercising:
- Element types: f32, i32
- Shapes: rank-1 at small fixed sizes
- Constructs: scalar arithmetic, map, fold (via prelude sum).
- Deterministic seeding, bounded program size, reproducible corpus.
"""

from __future__ import annotations

import random
from typing import Iterator

SEED = 42

SCALAR_INT_EXPRS = [
    "2 + 3",
    "7 - 1",
    "3 * 4",
    "42 + 0",
    "1 + 7",
]

SCALAR_F32_EXPRS = [
    "2.0 + 3.0",
    "7.0 - 1.0",
    "3.14 * 2.0",
    "42.0 + 0.0",
    "2.0 * 3.14",
]

SECTION_OPS = ["(* 2)", "(* 3)", "(+ 1)", "(- 1)"]


def _iota(size: int) -> str:
    return f"(iota {size})"


def _gen(rng: random.Random) -> str:
    kind = rng.choice(["scalar_int", "scalar_f32", "sum_f32", "sum_int",
                         "map_section_f32", "map_section_int"])
    n = rng.choice([1, 3, 5, 10])

    if kind == "scalar_int":
        return rng.choice(SCALAR_INT_EXPRS)
    if kind == "scalar_f32":
        return rng.choice(SCALAR_F32_EXPRS)
    if kind == "sum_int":
        return f"sum {_iota(n)}"
    if kind == "sum_f32":
        return f"sum {_iota(n)}"
    if kind == "map_section_int":
        section = rng.choice(SECTION_OPS)
        return f"map {section} {_iota(n)}"
    # map_section_f32
    section = rng.choice(SECTION_OPS)
    return f"map {section} {_iota(n)}"


def generate_programs(seed: int = SEED, count: int = 30) -> Iterator[str]:
    rng = random.Random(seed)
    for _ in range(count):
        yield _gen(rng)


def is_well_typed(program: str) -> bool:
    try:
        from remora.parser import parse_program
        from remora.prelude import with_prelude
        from remora.typechecker import TypeChecker
        ast = parse_program(with_prelude(program))
        TypeChecker().check_program(ast)
        return True
    except Exception:
        return False
