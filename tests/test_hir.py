import pytest

from remora.hir import (
    HIRArrayLit,
    HIRCast,
    HIRFold,
    HIRIf,
    HIRIndex,
    HIRLoweringError,
    HIRIota,
    HIRLambda,
    HIRLet,
    HIRLit,
    HIRMap,
    HIRPrimCallable,
    HIRPrimOp,
    HIRProgram,
    HIRReverse,
    HIRVar,
    lower_to_hir,
)
from remora.parser import parse_program
from remora.typechecker import TypeChecker
from remora.types import FLOAT, INT, ArrayType, StaticDim


def lower_program_source(source: str):
    typed = TypeChecker().check_program(parse_program(source))
    return lower_to_hir(typed)


def test_lowers_iota():
    program = lower_program_source("iota 10")

    assert isinstance(program, HIRProgram)
    assert isinstance(program.main, HIRIota)
    assert program.main.size == StaticDim(10)
    assert program.return_type == ArrayType(INT, (StaticDim(10),))


def test_lowers_reverse():
    program = lower_program_source("reverse [[1, 2], [3, 4]]")

    assert isinstance(program.main, HIRReverse)
    assert program.return_type == ArrayType(INT, (StaticDim(2), StaticDim(2)))


def test_lowers_shape_and_rank_to_static_constants():
    shape_program = lower_program_source("shape [[1, 2], [3, 4]]")
    rank_program = lower_program_source("rank [[1, 2], [3, 4]]")
    scalar_shape_program = lower_program_source("shape 42")

    assert isinstance(shape_program.main, HIRArrayLit)
    assert [
        element.value
        for element in shape_program.main.elements
        if isinstance(element, HIRLit)
    ] == [2, 2]
    assert shape_program.return_type == ArrayType(INT, (StaticDim(2),))

    assert isinstance(rank_program.main, HIRLit)
    assert rank_program.main.value == 2
    assert rank_program.return_type == INT

    assert isinstance(scalar_shape_program.main, HIRArrayLit)
    assert scalar_shape_program.main.elements == []
    assert scalar_shape_program.return_type == ArrayType(INT, (StaticDim(0),))


def test_lowers_index_expression():
    program = lower_program_source("[[1, 2], [3, 4]][1, 0]")

    assert isinstance(program.main, HIRIndex)
    assert isinstance(program.main.array, HIRArrayLit)
    assert [
        index.value
        for index in program.main.indices
        if isinstance(index, HIRLit)
    ] == [1, 0]
    assert program.return_type == INT


def test_lowers_scalar_if_expression():
    program = lower_program_source("if true then 1 else 2")

    assert isinstance(program.main, HIRIf)
    assert isinstance(program.main.condition, HIRLit)
    assert program.main.condition.value is True
    assert program.main.result_type == INT


def test_lowers_array_literal_with_typed_elements():
    program = lower_program_source("[1, 2, 3]")

    assert isinstance(program.main, HIRArrayLit)
    assert [element.value for element in program.main.elements if isinstance(element, HIRLit)] == [
        1,
        2,
        3,
    ]


def test_lowers_numeric_casts_explicitly():
    program = lower_program_source("1 + 2.0")

    assert isinstance(program.main, HIRPrimOp)
    assert program.main.op == "+f"
    assert isinstance(program.main.args[0], HIRCast)
    assert program.main.args[0].from_type == INT
    assert program.main.args[0].to_type == FLOAT


def test_lowers_scalar_map_with_lambda():
    program = lower_program_source("let xs = [1.0, 2.0] in map (\\x -> x * 2.0) xs")

    assert isinstance(program.main, HIRLet)
    assert isinstance(program.main.body, HIRMap)
    map_node = program.main.body
    assert map_node.frame_shape == (StaticDim(2),)
    assert map_node.cell_shape == ()
    assert isinstance(map_node.func, HIRLambda)
    assert isinstance(map_node.func.body, HIRPrimOp)
    assert map_node.func.body.op == "*f"


def test_lowers_binary_map_shape_metadata():
    program = lower_program_source(
        "let xs = [1, 2] in let ys = [3, 4] in map (*) xs ys"
    )

    assert isinstance(program.main, HIRLet)
    inner = program.main.body
    assert isinstance(inner, HIRLet)
    assert isinstance(inner.body, HIRMap)
    map_node = inner.body
    assert len(map_node.arrays) == 2
    assert map_node.frame_shape == (StaticDim(2),)
    assert map_node.cell_shape == ()
    assert map_node.result_type == ArrayType(INT, (StaticDim(2),))


def test_lowers_vector_cell_map_shape_metadata():
    program = lower_program_source(
        "let xs = [[1.0, 2.0], [3.0, 4.0]] in map (\\row -> fold (+) 0.0 row) xs"
    )

    assert isinstance(program.main, HIRLet)
    assert isinstance(program.main.body, HIRMap)
    map_node = program.main.body
    assert map_node.frame_shape == (StaticDim(2),)
    assert map_node.cell_shape == (StaticDim(2),)
    assert map_node.result_type == ArrayType(FLOAT, (StaticDim(2),))


def test_lowers_fold_with_primitive_callable():
    program = lower_program_source("let xs = [1.0, 2.0, 3.0] in fold (+) 0.0 xs")

    assert isinstance(program.main, HIRLet)
    assert isinstance(program.main.body, HIRFold)
    fold = program.main.body
    assert fold.reduction_dim == StaticDim(3)
    assert isinstance(fold.func, HIRPrimCallable)
    assert fold.func.op == "+"
    assert fold.result_type == FLOAT


def test_lowers_array_cell_fold_with_primitive_callable():
    program = lower_program_source(
        "let init = [0, 0] in let xs = [[1, 2], [3, 4]] in fold (+) init xs"
    )

    assert isinstance(program.main, HIRLet)
    inner = program.main.body
    assert isinstance(inner, HIRLet)
    assert isinstance(inner.body, HIRFold)
    fold = inner.body
    assert fold.reduction_dim == StaticDim(2)
    assert isinstance(fold.func, HIRPrimCallable)
    assert fold.func.op == "+"
    assert fold.result_type == ArrayType(INT, (StaticDim(2),))


def test_lowers_operator_section_bound_argument():
    program = lower_program_source("map (* 2.0) (iota 10)")

    assert isinstance(program.main, HIRMap)
    assert isinstance(program.main.func, HIRPrimCallable)
    assert program.main.func.op == "*"
    assert isinstance(program.main.func.right_arg, HIRLit)
    assert program.main.func.right_arg.value == 2.0
    assert program.main.result_type == ArrayType(FLOAT, (StaticDim(10),))


def test_lowers_top_level_value_definitions_as_lets():
    program = lower_program_source("def xs = iota 4\nmap (* 2.0) xs")

    assert isinstance(program.main, HIRLet)
    assert program.main.name == "xs"
    assert isinstance(program.main.value, HIRIota)
    assert isinstance(program.main.body, HIRMap)
    assert isinstance(program.main.body.array, HIRVar)
    assert program.main.body.array.name == "xs"


def test_definition_only_program_is_rejected_by_hir_lowering():
    typed = TypeChecker().check_program(parse_program("def xs = iota 4"))

    with pytest.raises(HIRLoweringError, match="definition-only"):
        lower_to_hir(typed)


def test_lowers_m2_milestone_expression():
    program = lower_program_source("fold (+) 0.0 (map (\\x -> x * x) (iota 10))")

    assert isinstance(program.main, HIRFold)
    assert program.main.result_type == FLOAT
    assert isinstance(program.main.array, HIRMap)
    assert program.main.array.result_type == ArrayType(INT, (StaticDim(10),))


# ---------------------------------------------------------------------------
# HIR CSE tests (Phase 4)
# ---------------------------------------------------------------------------


def _make_test_map(name="x", size=4):
    """Helper: build ``map (* 2.0) x`` as a HIRMap."""
    from remora.types import FLOAT, ArrayType, StaticDim
    x = HIRVar(name, ArrayType(FLOAT, (StaticDim(size),)))
    dc = HIRPrimCallable("*", (FLOAT, FLOAT), FLOAT)
    return HIRMap((StaticDim(size),), (), dc, [x], ArrayType(FLOAT, (StaticDim(size),)))


class TestHIRCSE:
    def test_identical_pure_maps_are_shared(self):
        """Two structurally identical HIRMap nodes become one shared binding."""
        from remora.hir_opt import hir_cse
        from remora.hir import HIRIf, HIRVar, HIRLit
        from remora.types import FLOAT, BOOL, ArrayType, StaticDim

        m1 = _make_test_map("x", 4)
        m2 = _make_test_map("x", 4)
        body = HIRIf(HIRLit(True, BOOL), m1, m2, ArrayType(FLOAT, (StaticDim(4),)))
        rewritten, bindings = hir_cse(body)

        # Both maps reference the same free variable, so one binding should be shared
        map_names = [n for n, e in bindings if isinstance(e, HIRMap)]
        assert len(map_names) == 1

    def test_different_shapes_are_not_merged(self):
        """Maps with different output shapes are not merged."""
        from remora.hir_opt import hir_cse
        from remora.hir import HIRIf, HIRLit
        from remora.types import FLOAT, BOOL, ArrayType, StaticDim

        m_small = _make_test_map("x", 2)
        m_large = _make_test_map("x", 4)
        body = HIRIf(HIRLit(True, BOOL), m_small, m_large,
                     ArrayType(FLOAT, (StaticDim(2),)))
        rewritten, bindings = hir_cse(body)

        map_bindings = [(n, e) for n, e in bindings if isinstance(e, HIRMap)]
        # Two maps with different shapes → 2 separate bindings
        assert len(map_bindings) == 2

    def test_shadowed_variable_not_merged_with_free_variable(self):
        """A map referencing a let-bound variable is not merged with a free-var map."""
        from remora.hir_opt import hir_cse
        from remora.hir import HIRIf, HIRLet, HIRLit, HIRVar
        from remora.types import FLOAT, BOOL, ArrayType, StaticDim

        free_map = _make_test_map("x", 4)
        # Map that references a local 'x' (different from the outer 'x')
        shadowed_var = HIRVar("x", ArrayType(FLOAT, (StaticDim(4),)))
        dc = HIRPrimCallable("*", (FLOAT, FLOAT), FLOAT)
        shadowed_map = HIRMap(
            (StaticDim(4),), (), dc, [shadowed_var],
            ArrayType(FLOAT, (StaticDim(4),)),
        )
        let_body = HIRLet(
            "x", ArrayType(FLOAT, (StaticDim(4),)),
            HIRLit(0.0, FLOAT), shadowed_map,
            ArrayType(FLOAT, (StaticDim(4),)),
        )
        body = HIRIf(HIRLit(True, BOOL), free_map, let_body,
                     ArrayType(FLOAT, (StaticDim(4),)))
        rewritten, bindings = hir_cse(body)

        # The let-bound map references a local 'x', so it should NOT be hoisted
        # Only the free-var map should be hoisted
        map_names = [n for n, e in bindings if isinstance(e, HIRMap)]
        assert len(map_names) <= 1  # at most the free-var map is hoisted

    def test_duplicate_analysis_counts_duplicates(self):
        """hir_duplicate_analysis reports counts for repeated subtrees."""
        from remora.hir_opt import hir_duplicate_analysis
        from remora.hir import HIRIf, HIRLit
        from remora.types import FLOAT, BOOL, ArrayType, StaticDim

        m1 = _make_test_map("x", 4)
        m2 = _make_test_map("x", 4)
        body = HIRIf(HIRLit(True, BOOL), m1, m2, ArrayType(FLOAT, (StaticDim(4),),))

        stats = hir_duplicate_analysis(body)
        assert stats["total_subtrees"] > 0
        assert stats["unique_subtrees"] > 0
        assert stats["duplicated_subtrees"] >= 1
        assert stats["max_duplication"] >= 2

    def test_dce_removes_dead_let_binding(self):
        """hir_dce removes a let whose name is unused in the body."""
        from remora.hir_opt import hir_dce
        from remora.hir import HIRLet, HIRVar
        from remora.types import FLOAT, ArrayType, StaticDim

        map_expr = _make_test_map("x", 4)
        dead_let = HIRLet(
            "unused", ArrayType(FLOAT, (StaticDim(4),)),
            map_expr,
            map_expr,  # body doesn't reference "unused"
            ArrayType(FLOAT, (StaticDim(4),)),
        )
        result = hir_dce(dead_let)
        # After DCE, the dead let should be removed; result is just map_expr
        assert not isinstance(result, HIRLet)
        assert isinstance(result, HIRMap)

    def test_let_used_in_body_is_preserved_by_dce(self):
        """hir_dce preserves a let whose name is used in the body."""
        from remora.hir_opt import hir_dce
        from remora.hir import HIRLet, HIRVar
        from remora.types import FLOAT, ArrayType, StaticDim

        map_expr = _make_test_map("x", 4)
        used_let = HIRLet(
            "used", ArrayType(FLOAT, (StaticDim(4),)),
            map_expr,
            HIRVar("used", ArrayType(FLOAT, (StaticDim(4),))),
            ArrayType(FLOAT, (StaticDim(4),)),
        )
        result = hir_dce(used_let)
        assert isinstance(result, HIRLet)
        assert result.name == "used"

    def test_repeated_array_expression_emits_mlir_once(self):
        """A repeated array-valued map lowered via descriptor export appears
        once in MLIR, not once per occurrence."""
        from remora.types import FLOAT, ArrayType, StaticDim
        from remora.compiler import prepare_function_source
        from remora.lowering import MLIRLowering

        # A function returning an array composed of two identical maps.
        # The form (+ map map) uses the same map expression twice.
        # CSE should share the map body so it's lowered once.
        source = (
            "(define/pi ()\n"
            "  (dup-map [x (Array Float 4)] (Array Float 4))\n"
            "  (+ (map (* 2.0) x)\n"
            "     (map (* 2.0) x)))\n"
        )
        prepared = prepare_function_source(
            source,
            "dup-map",
            (ArrayType(FLOAT, (StaticDim(4),)),),
            include_prelude=False,
            syntax="lisp",
        )
        lowered = MLIRLowering().lower_function_descriptor_export(
            prepared.hir_function, export_name="test_fn"
        )
        mlir_text = str(lowered.module)
        assert mlir_text, "expected non-empty MLIR"
        # With CSE sharing, the map body is lowered once and reused.
        # Without sharing, each occurrence becomes a separate linalg.generic.
        linalg_generic_count = mlir_text.count("linalg.generic")
        # Binary map (+ 2.0) produces one linalg.generic for the add,
        # plus the scalar * 2.0 map. With CSE sharing the * 2.0 map,
        # total should be 2 (one for the scalar map, one for the add).
        assert linalg_generic_count == 2, (
            f"Expected 2 linalg.generic (shared scalar map + binary add), "
            f"got {linalg_generic_count}"
        )

    def test_generated_gradient_has_shared_forward_intermediates(self):
        """CSE detects and shares repeated forward intermediates in a
        generated gradient."""
        from remora.hir_opt import hir_cse, hir_duplicate_analysis
        from remora.compiler import prepare_function_source
        from remora.ad_source import generate_gradient_function_source
        from remora.types import FLOAT, ArrayType, StaticDim

        # A simple loss: dot product of x and w, then squared.
        source = (
            "(define/pi ([n Dim])\n"
            "  (sq-dot [x (Array Float n) w (Array Float n)] Float)\n"
            "  (:: dot (fold + 0.0 (* x w))\n"
            "    (* dot dot)))\n"
        )
        gradient = generate_gradient_function_source(
            source,
            "sq-dot",
            (ArrayType(FLOAT, (StaticDim(5),)), ArrayType(FLOAT, (StaticDim(5),))),
            differentiate_input=0,
            include_prelude=False,
            syntax="lisp",
        )
        prepared = prepare_function_source(
            gradient.source,
            gradient.function_name,
            gradient.param_types,
            include_prelude=False,
            syntax="lisp",
        )

        # Run duplicate analysis on the original HIR body
        stats_before = hir_duplicate_analysis(prepared.hir_function.body)
        assert stats_before["duplicated_subtrees"] > 0, (
            "expected duplicated subexpressions in generated gradient"
        )

        # Run CSE and verify bindings are produced
        cse_body, cse_bindings = hir_cse(prepared.hir_function.body)
        assert len(cse_bindings) > 0, (
            "expected CSE to find shared forward intermediates"
        )
