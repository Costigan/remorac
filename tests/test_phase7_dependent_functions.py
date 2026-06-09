import pytest

from remora.compiler import compile_function_source, compile_source
from remora.core_verify import CoreVerificationError, verify_core_program
from remora.lisp_reader import parse_lisp
from remora.index import DimVar
from remora.elaborate import elaborate_program
from remora.runtime import evaluate_source, evaluate_source_compiled
from remora.index import IndexBinder, IndexSort
from remora.typechecker import (
    TypeChecker,
    TypeEnv,
    TypedApp,
    TypedArray,
    TypedIndexApp,
)
from remora.types import FLOAT, ArrayType, FuncType, PiType, RemoraTypeError, StaticDim


def check_lisp(source: str):
    return TypeChecker().check_program(parse_lisp(source))


def test_type_environment_keeps_index_namespace_separate():
    env = TypeEnv().extend("n", FLOAT).extend_index(
        IndexBinder("n", IndexSort.DIM)
    )

    assert env.lookup("n") == FLOAT
    assert env.lookup_index("n") is IndexSort.DIM


def test_dependent_function_definition_has_pi_type():
    program = check_lisp(
        "(define/pi ([n Dim]) (dot [xs (Array Float n) ys (Array Float n)] Float) "
        "(fold + 0.0 (* xs ys)))"
    )

    definition_type = program.definitions[0].type
    assert isinstance(definition_type, PiType)
    assert len(definition_type.binders) == 1
    assert isinstance(definition_type.body, FuncType)
    assert definition_type.body.params == (
        ArrayType(FLOAT, (DimVar("n"),)),
        ArrayType(FLOAT, (DimVar("n"),)),
    )
    assert definition_type.body.result == FLOAT


def test_dependent_function_call_specializes_dimension():
    program = check_lisp(
        "(define/pi ([n Dim]) (dot [xs (Array Float n) ys (Array Float n)] Float) "
        "(fold + 0.0 (* xs ys))) "
        "(dot [1.0 2.0] [3.0 4.0])"
    )

    assert isinstance(program.body, TypedApp)
    assert program.type == FLOAT
    assert program.body.func.type == FuncType(
        (
            ArrayType(FLOAT, (StaticDim(2),)),
            ArrayType(FLOAT, (StaticDim(2),)),
        ),
        FLOAT,
    )


def test_dependent_function_call_rejects_mismatched_dimensions():
    with pytest.raises(RemoraTypeError, match="binding mismatch"):
        check_lisp(
            "(define/pi ([n Dim]) (dot [xs (Array Float n) ys (Array Float n)] Float) "
            "(fold + 0.0 (* xs ys))) "
            "(dot [1.0 2.0] [3.0])"
        )


def test_dependent_function_erases_for_interpreted_and_compiled_execution():
    source = (
        "(define/pi ([n Dim]) (dot [xs (Array Float n) ys (Array Float n)] Float) "
        "(fold + 0.0 (* xs ys))) "
        "(dot [1.0 2.0] [3.0 4.0])"
    )

    interpreted = evaluate_source(source, include_prelude=False, syntax="lisp")
    compiled = evaluate_source_compiled(source, include_prelude=False, syntax="lisp")

    assert interpreted.value == pytest.approx(11.0)
    assert compiled.value == pytest.approx(11.0)


def test_dependent_function_checks_declared_result_type():
    with pytest.raises(RemoraTypeError, match="expected bool"):
        check_lisp(
            "(define/pi ([n Dim]) (bad [xs (Array Float n)] Bool) "
            "(fold + 0.0 xs))"
        )


def test_dependent_function_rejects_duplicate_index_binders():
    with pytest.raises(RemoraTypeError, match="duplicate index binder"):
        check_lisp(
            "(define/pi ([n Dim] [n Dim]) (bad [xs (Array Float n)] Float) "
            "(fold + 0.0 xs))"
        )


def test_dependent_function_rejects_unbound_dimension_variable():
    with pytest.raises(RemoraTypeError, match="unbound index variable"):
        check_lisp(
            "(define/pi ([n Dim]) (bad [xs (Array Float m)] Float) "
            "(fold + 0.0 xs))"
        )


def test_explicit_index_application_specializes_pi_function():
    program = check_lisp(
        "(define/pi ([n Dim]) (dot [xs (Array Float n) ys (Array Float n)] Float) "
        "(fold + 0.0 (* xs ys))) "
        "((iapp dot 2) [1.0 2.0] [3.0 4.0])"
    )

    assert isinstance(program.body, TypedApp)
    assert isinstance(program.body.func, TypedIndexApp)
    assert program.body.func.index_args == (StaticDim(2),)
    assert program.type == FLOAT


def test_explicit_index_application_is_recorded_in_core():
    typed = check_lisp(
        "(define/pi ([n Dim]) (sum [xs (Array Float n)] Float) "
        "(fold + 0.0 xs)) "
        "((iapp sum 3) [1.0 2.0 3.0])"
    )

    core = elaborate_program(typed)

    assert len(core.index_applications) == 1
    assert core.index_applications[0].function_name == "sum"
    assert core.index_applications[0].index_args == (StaticDim(3),)
    assert core.index_applications[0].type == FuncType(
        (ArrayType(FLOAT, (StaticDim(3),)),),
        FLOAT,
    )


def test_explicit_index_application_executes_after_erasure():
    source = (
        "(define/pi ([n Dim]) (dot [xs (Array Float n) ys (Array Float n)] Float) "
        "(fold + 0.0 (* xs ys))) "
        "((iapp dot 2) [1.0 2.0] [3.0 4.0])"
    )

    interpreted = evaluate_source(source, include_prelude=False, syntax="lisp")
    compiled = evaluate_source_compiled(source, include_prelude=False, syntax="lisp")

    assert interpreted.value == pytest.approx(11.0)
    assert compiled.value == pytest.approx(11.0)


def test_explicit_index_application_rejects_wrong_index_arity():
    with pytest.raises(RemoraTypeError, match="expects 1 index argument"):
        check_lisp(
            "(define/pi ([n Dim]) (sum [xs (Array Float n)] Float) "
            "(fold + 0.0 xs)) "
            "((iapp sum 2 3) [1.0 2.0])"
        )


def test_explicit_index_application_rejects_monomorphic_function():
    with pytest.raises(RemoraTypeError, match="does not have a Pi type"):
        check_lisp("(define (sum [xs]) (fold + 0.0 xs)) ((iapp sum 2) [1.0 2.0])")


def test_explicit_index_application_specializes_dependent_result_shape():
    source = (
        "(define/pi ([n Dim]) (identity [xs (Array Float n)] (Array Float n)) xs) "
        "((iapp identity 3) [1.0 2.0 3.0])"
    )

    typed = check_lisp(source)
    interpreted = evaluate_source(source, include_prelude=False, syntax="lisp")
    compiled = evaluate_source_compiled(source, include_prelude=False, syntax="lisp")

    assert typed.type == ArrayType(FLOAT, (StaticDim(3),))
    assert interpreted.value.tolist() == [1.0, 2.0, 3.0]
    assert compiled.value.tolist() == [1.0, 2.0, 3.0]


def test_explicit_specialization_erases_to_monomorphic_hir_and_mlir():
    dependent = (
        "(define/pi ([n Dim]) (dot [xs (Array Float n) ys (Array Float n)] Float) "
        "(fold + 0.0 (* xs ys))) "
        "((iapp dot 2) [1.0 2.0] [3.0 4.0])"
    )
    monomorphic = (
        "(define (dot [xs ys]) (fold + 0.0 (* xs ys))) "
        "(dot [1.0 2.0] [3.0 4.0])"
    )

    dependent_artifact = compile_source(
        dependent,
        verify=False,
        include_prelude=False,
        syntax="lisp",
    )
    monomorphic_artifact = compile_source(
        monomorphic,
        verify=False,
        include_prelude=False,
        syntax="lisp",
    )

    assert dependent_artifact.hir == monomorphic_artifact.hir
    assert dependent_artifact.mlir_text == monomorphic_artifact.mlir_text


def test_inferred_dependent_call_records_named_core_specialization():
    typed = check_lisp(
        "(define/pi ([m Dim] [n Dim]) "
        "(identity [xs (Array Float m n)] (Array Float m n)) xs) "
        "(identity [[1.0 2.0 3.0] [4.0 5.0 6.0]])"
    )

    core = elaborate_program(typed)

    assert len(core.specializations) == 1
    specialization = core.specializations[0]
    assert specialization.name == "identity__m_2__n_3"
    assert specialization.index_args == (StaticDim(2), StaticDim(3))
    assert specialization.type == FuncType(
        (ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),),
        ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
    )


def test_repeated_explicit_index_application_reuses_specialization():
    typed = check_lisp(
        "(define/pi ([n Dim]) (sum [xs (Array Float n)] Float) "
        "(fold + 0.0 xs)) "
        "[((iapp sum 2) [1.0 2.0]) ((iapp sum 2) [3.0 4.0])]"
    )

    assert isinstance(typed.body, TypedArray)
    first = typed.body.elements[0]
    second = typed.body.elements[1]
    assert isinstance(first, TypedApp)
    assert isinstance(second, TypedApp)
    assert isinstance(first.func, TypedIndexApp)
    assert isinstance(second.func, TypedIndexApp)
    assert first.func.function is second.func.function

    core = elaborate_program(typed)
    assert len(core.index_applications) == 2
    assert len(core.specializations) == 1
    assert core.specializations[0].name == "sum__n_2"


def test_core_verifier_requires_matching_specialization_for_iapp():
    typed = check_lisp(
        "(define/pi ([n Dim]) (sum [xs (Array Float n)] Float) "
        "(fold + 0.0 xs)) "
        "((iapp sum 2) [1.0 2.0])"
    )
    core = elaborate_program(typed)
    broken = type(core)(
        core.definitions,
        core.body,
        core.type,
        core.index_applications,
        (),
    )

    with pytest.raises(CoreVerificationError, match="no matching core specialization"):
        verify_core_program(broken)


def test_dependent_function_body_is_checked_without_application():
    with pytest.raises(RemoraTypeError, match="unbound variable"):
        check_lisp(
            "(define/pi ([n Dim]) (bad [xs (Array Float n)] Float) missing)"
        )


def test_standalone_function_compilation_uses_dependent_specialization():
    source = (
        "(define/pi ([n Dim]) (sum [xs (Array Float n)] Float) "
        "(fold + 0.0 xs))"
    )

    artifact = compile_function_source(
        source,
        "sum",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        verify=False,
        include_prelude=False,
        syntax="lisp",
    )

    assert artifact.specialization_name == "sum__n_4"
    assert artifact.index_args == (StaticDim(4),)
    assert artifact.hir_function.name == "sum__n_4"
    assert artifact.function_type == FuncType(
        (ArrayType(FLOAT, (StaticDim(4),)),),
        FLOAT,
    )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("[1.0 2.0]", "[3.0 4.0]", 11.0),
        ("[1.0 2.0 3.0]", "[4.0 5.0 6.0]", 32.0),
    ],
)
def test_dot_product_compiles_at_multiple_concrete_lengths(left, right, expected):
    source = (
        "(define/pi ([n Dim]) (dot [xs (Array Float n) ys (Array Float n)] Float) "
        "(fold + 0.0 (* xs ys))) "
        f"(dot {left} {right})"
    )

    result = evaluate_source_compiled(
        source,
        include_prelude=False,
        syntax="lisp",
    )

    assert result.value == pytest.approx(expected)


# ── Phase 7.3: Shape variables ────────────────────────────────────────────

def test_shape_preserving_identity_rank1():
    source = (
        "(define/pi ([s Shape]) (id [x (Array Float s)] (Array Float s)) x) "
        "(id [1.0 2.0 3.0])"
    )
    result = evaluate_source(source, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(result.value, [1.0, 2.0, 3.0])


def test_shape_preserving_identity_rank2():
    source = (
        "(define/pi ([s Shape]) (id [x (Array Float s)] (Array Float s)) x) "
        "(id [[1.0 2.0] [3.0 4.0] [5.0 6.0]])"
    )
    result = evaluate_source(source, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(result.value, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_shape_preserving_identity_compiled():
    source = (
        "(define/pi ([s Shape]) (id [x (Array Float s)] (Array Float s)) x) "
        "(id [1.0 2.0 3.0])"
    )
    result = evaluate_source_compiled(source, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(result.value, [1.0, 2.0, 3.0])


def test_shape_preserving_identity_specialization_named():
    source = (
        "(define/pi ([s Shape]) (id [x (Array Float s)] (Array Float s)) x) "
        "(id [1.0 2.0 3.0])"
    )
    from remora.compiler import compile_function_source
    artifact = compile_function_source(
        source,
        "id",
        (ArrayType(FLOAT, (StaticDim(3),)),),
        verify=False,
        include_prelude=False,
        syntax="lisp",
    )
    assert artifact.specialization_name is not None
    assert "id__" in artifact.specialization_name
    assert "s_shape" in artifact.specialization_name


def test_shape_variable_rejects_dim_sort_at_use_site():
    """A Shape binder cannot be used as a Dim binder argument."""
    source = "(define/pi ([s Shape]) (id [x (Array Float s)] (Array Float s)) x) (iapp id 5)"
    with pytest.raises((RemoraTypeError, ValueError)):
        evaluate_source(source, syntax="lisp", include_prelude=False)


def test_shape_preserving_called_twice():
    """Calling a shape-preserving function at different shapes works."""
    source = (
        "(define/pi ([s Shape]) (id [x (Array Float s)] (Array Float s)) x) "
        "(id (id [1.0 2.0]))"
    )
    result = evaluate_source(source, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(result.value, [1.0, 2.0])


# ── Phase 7.4: Dimension arithmetic ────────────────────────────────────────

def test_arithmetic_add_in_result_type():
    """Function declares result shape as (+ a b) where a, b are Dim binders."""
    source = (
        "(define/pi ([a Dim] [b Dim]) "
        "  (append-vecs [xs (Array Float a) ys (Array Float b)] "
        "    (Array Float (+ a b))) "
        "  (append xs ys)) "
        "(append-vecs [1.0 2.0] [3.0 4.0 5.0])"
    )
    result = evaluate_source(source, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(result.value, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_arithmetic_add_in_result_type_compiled():
    """Arithmetic result type specialization works through compiled path."""
    source = (
        "(define/pi ([a Dim] [b Dim]) "
        "  (append-vecs [xs (Array Float a) ys (Array Float b)] "
        "    (Array Float (+ a b))) "
        "  (append xs ys)) "
        "(append-vecs [1.0 2.0] [3.0 4.0 5.0])"
    )
    from remora.compiler import compile_function_source
    from remora.types import ArrayType, FLOAT, StaticDim, FuncType
    artifact = compile_function_source(
        source,
        "append-vecs",
        (ArrayType(FLOAT, (StaticDim(2),)), ArrayType(FLOAT, (StaticDim(3),))),
        verify=False,
        include_prelude=False,
        syntax="lisp",
    )
    assert artifact.specialization_name is not None
    assert "append-vecs__" in artifact.specialization_name
    assert "a_2" in artifact.specialization_name
    assert "b_3" in artifact.specialization_name
    assert artifact.function_type.result == ArrayType(FLOAT, (StaticDim(5),))


# ── Phase 7.6: End-to-end examples ──────────────────────────────────────────

def test_example_dot_product_via_dim():
    src = (
        "(define/pi ([n Dim]) "
        "  (dot [xs (Array Float n) ys (Array Float n)] Float) "
        "  (fold + 0.0 (* xs ys))) "
        "(dot [1.0 2.0 3.0] [4.0 5.0 6.0])"
    )
    r = evaluate_source(src, syntax="lisp", include_prelude=False)
    assert r.value == pytest.approx(32.0)
    rc = evaluate_source_compiled(src, syntax="lisp", include_prelude=False)
    assert rc.value == pytest.approx(32.0)


def test_example_shape_identity_multi_rank():
    src = (
        "(define/pi ([s Shape]) (id [x (Array Float s)] (Array Float s)) x) "
        "(id [1.0 2.0 3.0 4.0])"
    )
    r = evaluate_source(src, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(r.value, [1.0, 2.0, 3.0, 4.0])


def test_example_append_with_arithmetic_result():
    src = (
        "(define/pi ([a Dim] [b Dim]) "
        "  (concat-first [xs (Array Float a) ys (Array Float b)]"
        "    (Array Float (+ a b))) "
        "  (append xs ys)) "
        "(concat-first [1.0 2.0] [3.0 4.0 5.0])"
    )
    r = evaluate_source(src, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(r.value, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_example_dependent_function_same_dim_twice():
    """Pi function where same dim appears in both params; auto-lifted body."""
    src = (
        "(define/pi ([n Dim]) "
        "  (add-vecs [xs (Array Float n) ys (Array Float n)]"
        "    (Array Float n)) "
        "  (+ xs ys)) "
        "(add-vecs [1.0 2.0] [3.0 4.0])"
    )
    r = evaluate_source(src, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(r.value, [4.0, 6.0])


# ── Phase 7.5: Forall / element-type polymorphism ──────────────────────────

def test_forall_identity_int():
    src = "(define/forall (t) (id [x (Array t 3)] (Array t 3)) x) (id [1 2 3])"
    r = evaluate_source(src, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(r.value, [1, 2, 3])


def test_forall_identity_float():
    src = "(define/forall (t) (id [x (Array t 2)] (Array t 2)) x) (id [1.0 2.0])"
    r = evaluate_source(src, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(r.value, [1.0, 2.0])


def test_forall_identity_compiled():
    src = "(define/forall (t) (id [x (Array t 3)] (Array t 3)) x) (id [1 2 3])"
    r = evaluate_source_compiled(src, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(r.value, [1, 2, 3])


# ── Rest variables: append with common suffix ──────────────────────────────

def test_rest_variable_append_rank1():
    src = (
        "(define/pi ([da Dim] [db Dim] [rest Shape]) "
        "  (append2 [xs (Array Float da rest) ys (Array Float db rest)] "
        "    (Array Float (+ da db) rest)) "
        "  (append xs ys)) "
        "(append2 [1.0 2.0] [3.0 4.0 5.0])"
    )
    r = evaluate_source(src, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(r.value, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_rest_variable_append_rank2():
    src = (
        "(define/pi ([da Dim] [db Dim] [rest Shape]) "
        "  (append2 [xs (Array Float da rest) ys (Array Float db rest)] "
        "    (Array Float (+ da db) rest)) "
        "  (append xs ys)) "
        "(append2 [[1.0 2.0] [3.0 4.0]] [[5.0 6.0] [7.0 8.0]])"
    )
    r = evaluate_source(src, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(r.value, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])


def test_rest_variable_rejects_mismatched_trailing():
    src = (
        "(define/pi ([da Dim] [rest Shape]) "
        "  (bad [xs (Array Float da rest) ys (Array Float da rest)] "
        "    (Array Float da rest)) "
        "  (append xs ys)) "
        "(bad [1.0 2.0] [3.0 4.0 5.0])"
    )
    with pytest.raises(RemoraTypeError):
        evaluate_source(src, syntax="lisp", include_prelude=False)


def test_rest_variable_specialization_name():
    from remora.compiler import compile_function_source
    from remora.types import ArrayType, FLOAT, StaticDim
    src = (
        "(define/pi ([da Dim] [db Dim] [rest Shape]) "
        "  (append2 [xs (Array Float da rest) ys (Array Float db rest)] "
        "    (Array Float (+ da db) rest)) "
        "  (append xs ys))"
    )
    art = compile_function_source(
        src, "append2",
        (ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
         ArrayType(FLOAT, (StaticDim(4), StaticDim(3)))),
        verify=False, include_prelude=False, syntax="lisp",
    )
    assert "append2__" in art.specialization_name
    assert "rest_shape_3" in art.specialization_name
    assert art.function_type.result == ArrayType(FLOAT, (StaticDim(6), StaticDim(3)))


# ── Take / drop result-shape arithmetic ────────────────────────────────────

def test_take_result_shape_is_literal():
    src = (
        "(define/pi ([n Dim]) "
        "  (take2 [xs (Array Float n)] (Array Float 2)) "
        "  (take 2 xs)) "
        "(take2 [1.0 2.0 3.0 4.0 5.0])"
    )
    from remora.lisp_reader import parse_lisp
    from remora.typechecker import TypeChecker
    tc = TypeChecker()
    typed = tc.check_program(parse_lisp(src))
    assert typed.type is not None
    from remora.types import ArrayType, FLOAT, StaticDim
    assert typed.type == ArrayType(FLOAT, (StaticDim(2),))


def test_drop_result_shape_uses_dimsub():
    src = (
        "(define/pi ([n Dim]) "
        "  (drop2 [xs (Array Float n)] (Array Float (- n 2))) "
        "  (drop 2 xs)) "
        "(drop2 [1.0 2.0 3.0 4.0 5.0])"
    )
    from remora.lisp_reader import parse_lisp
    from remora.typechecker import TypeChecker
    tc = TypeChecker()
    typed = tc.check_program(parse_lisp(src))
    assert typed.type is not None
    from remora.types import ArrayType, FLOAT, StaticDim
    assert typed.type == ArrayType(FLOAT, (StaticDim(3),))


def test_drop_arithmetic_specialization():
    from remora.compiler import compile_function_source
    from remora.types import ArrayType, FLOAT, StaticDim
    src = (
        "(define/pi ([n Dim]) "
        "  (drop2 [xs (Array Float n)] (Array Float (- n 2))) "
        "  (drop 2 xs))"
    )
    art = compile_function_source(
        src, "drop2",
        (ArrayType(FLOAT, (StaticDim(5),)),),
        verify=False, include_prelude=False, syntax="lisp",
    )
    assert art.function_type.result == ArrayType(FLOAT, (StaticDim(3),))


# ── Combined Forall + Pi ──────────────────────────────────────────────────

def test_combined_forall_pi_identity():
    src = (
        "(define/pi ([n Dim] [t]) "
        "  (id [x (Array t n)] (Array t n)) "
        "  x) "
        "(id [1 2 3])"
    )
    r = evaluate_source(src, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(r.value, [1, 2, 3])


def test_combined_forall_pi_compiled():
    src = (
        "(define/pi ([n Dim] [t]) "
        "  (id [x (Array t n)] (Array t n)) "
        "  x) "
        "(id [1 2 3])"
    )
    r = evaluate_source_compiled(src, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(r.value, [1, 2, 3])


# ── Explicit shape application via iapp ────────────────────────────────────

def test_iapp_with_shape_literal():
    src = (
        "(define/pi ([s Shape]) "
        "  (id [x (Array Float s)] (Array Float s)) "
        "  x) "
        "(iapp id (shape 3))"
    )
    from remora.typechecker import TypeChecker
    from remora.lisp_reader import parse_lisp
    tc = TypeChecker()
    typed = tc.check_program(parse_lisp(src))
    from remora.typechecker import TypedIndexApp
    assert isinstance(typed.body, TypedIndexApp)
    from remora.types import FuncType, ArrayType, FLOAT
    assert isinstance(typed.body.type, FuncType)
    assert typed.body.type.result == ArrayType(FLOAT, (StaticDim(3),))


# ── AD0: grad typing ──────────────────────────────────────────────────────

def test_grad_typechecks_for_scalar_float_function():
    src = (
        "(define/pi ([n Dim]) "
        "  (sq [x (Array Float n)] Float) "
        "  (fold + 0.0 (* x x))) "
        "(grad sq)"
    )
    from remora.typechecker import TypeChecker
    from remora.lisp_reader import parse_lisp
    from remora.types import FuncType, ArrayType, FLOAT, PiType
    tc = TypeChecker()
    typed = tc.check_program(parse_lisp(src))
    assert typed.type is not None
    # AD3: grad of Pi-typed function is also Pi-typed
    inner = typed.type.body if isinstance(typed.type, PiType) else typed.type
    assert isinstance(inner, FuncType)
    # grad sq: input type = array, output type = same shape array
    assert isinstance(inner.params[0], ArrayType)
    assert inner.params[0] == inner.result


def test_grad_accepts_binary_function():
    src = (
        "(define/pi ([n Dim]) "
        "  (add [x (Array Float n) y (Array Float n)] Float) "
        "  (fold + 0.0 (* x y))) "
    )
    import numpy as np
    from remora.runtime import evaluate_source
    result = evaluate_source(
        src + "((grad (iapp add 3)) [1.0 2.0 3.0] [4.0 5.0 6.0])",
        include_prelude=False, syntax='lisp',
    )
    np.testing.assert_array_equal(result.value[0], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(result.value[1], [1.0, 2.0, 3.0])


def test_grad_rejects_non_float_result():
    from remora.types import RemoraTypeError
    import pytest
    src = (
        "(define/pi ([n Dim]) "
        "  (sq [x (Array Float n)] (Array Float n)) "
        "  (+ x x)) "
        "(grad sq)"
    )
    with pytest.raises(RemoraTypeError, match="scalar Float"):
        from remora.typechecker import TypeChecker
        from remora.lisp_reader import parse_lisp
        TypeChecker().check_program(parse_lisp(src))


def test_iapp_with_shape_literal_specialization():
    from remora.compiler import compile_function_source
    from remora.types import ArrayType, FLOAT, StaticDim
    src = (
        "(define/pi ([s Shape]) "
        "  (id [x (Array Float s)] (Array Float s)) "
        "  x)"
    )
    art = compile_function_source(
        src, "id",
        (ArrayType(FLOAT, (StaticDim(3),)),),
        verify=False, include_prelude=False, syntax="lisp",
    )
    assert art.specialization_name is not None
    assert art.function_type.result == ArrayType(FLOAT, (StaticDim(3),))


def test_combined_forall_pi_different_element_type():
    src = (
        "(define/pi ([n Dim] [t]) "
        "  (id [x (Array t n)] (Array t n)) "
        "  x) "
        "(id [1.0 2.0])"
    )
    r = evaluate_source(src, syntax="lisp", include_prelude=False)
    import numpy as np
    np.testing.assert_array_equal(r.value, [1.0, 2.0])
