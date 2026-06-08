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
    with pytest.raises(RemoraTypeError, match="dimension mismatch"):
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
