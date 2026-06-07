import importlib.util
import inspect
from pathlib import Path

import pytest

from remora.compiler import compile_source_to_mlir, compile_source
from remora.compiler import compile_function_source
from remora.defunc import defunctionalize
from remora.errors import RemoraError
from remora.hir import (
    HIRCall,
    HIRFunction,
    HIRLit,
    HIRParam,
    HIRPrimOp,
    HIRProgram,
    HIRVar,
    lower_to_hir,
)
from remora.runtime import evaluate_source_compiled
from remora import lowering as lowering_module
from remora.lowering import MLIRLowering, RemoraLoweringError, type_to_mlir
from remora.parser import parse_program
from remora.typechecker import TypeChecker
from remora.types import BOOL, FLOAT, INT, ArrayType, FuncType, StaticDim


GOLDEN_DIR = Path(__file__).parent / "golden_mlir"

GOLDEN_CASES = {
    "iota_rank1.mlir": "iota 10",
    "map_iota_scale.mlir": "map (* 2.0) (iota 10)",
    "map_rank2_literal_scale.mlir": (
        "let xs = [[1.0, 2.0], [3.0, 4.0]] in map (* 2.0) xs"
    ),
    "fold_map_iota_scale.mlir": "fold (+) 0.0 (map (* 2.0) (iota 10))",
}


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("iree") is None,
    reason="IREE compiler MLIR bindings are not installed",
)


def hir_from_source(source: str):
    typed = TypeChecker().check_program(parse_program(source))
    return defunctionalize(lower_to_hir(typed))


def nested_scalar_literal(rank: int, value: str = "1") -> str:
    return "[" * rank + value + "]" * rank


@pytest.mark.parametrize(("fixture_name", "source"), GOLDEN_CASES.items())
def test_lowering_matches_golden_mlir_fixtures(fixture_name: str, source: str):
    lowered = MLIRLowering().lower_program(hir_from_source(source))

    assert lowered.text.rstrip() + "\n" == (GOLDEN_DIR / fixture_name).read_text()


def test_type_to_mlir_scalars_arrays_and_functions():
    assert type_to_mlir(INT) == "i32"
    assert type_to_mlir(FLOAT) == "f32"
    assert type_to_mlir(BOOL) == "i1"
    assert type_to_mlir(ArrayType(FLOAT, (StaticDim(2), StaticDim(3)))) == (
        "tensor<2x3xf32>"
    )
    assert type_to_mlir(FuncType((INT,), FLOAT)) == "(i32) -> f32"


def test_lower_type_returns_parseable_mlir_types():
    lowering = MLIRLowering()

    assert str(lowering.lower_type(INT)) == "i32"
    assert str(lowering.lower_type(ArrayType(INT, (StaticDim(10),)))) == "tensor<10xi32>"


def test_tensor_let_lowering_uses_builder_not_main_string_splicing():
    source = inspect.getsource(lowering_module._lower_tensor_let_module)

    assert ".find(" not in source
    assert "func.func @main" not in source
    assert "_MLIRMainModuleBuilder" in source


def test_main_module_emitters_use_shared_builder():
    for function in [
        lowering_module._lower_scalar_module,
        lowering_module._lower_iota_module,
        lowering_module._lower_array_literal_module,
        lowering_module._lower_scalar_map_module,
        lowering_module._lower_scalar_map_binary_module,
    ]:
        source = inspect.getsource(function)
        assert "_MLIRMainModuleBuilder" in source, function.__name__
        assert "func.func @main" not in source, function.__name__


@pytest.mark.parametrize(
    ("source", "return_type", "expected_op"),
    [
        ("1 + 2", "i32", "arith.addi"),
        ("1 + 2.0", "f32", "arith.addf"),
        ("4 / 2", "f32", "arith.divf"),
        ("1 < 2", "i1", "arith.cmpi slt"),
        ("1.0 <= 2.0", "i1", "arith.cmpf ole"),
        ("true && false", "i1", "arith.andi"),
        ("true || false", "i1", "arith.ori"),
    ],
)
def test_lowers_scalar_primitive_expressions(source: str, return_type: str, expected_op: str):
    program = hir_from_source(source)
    lowered = MLIRLowering().lower_program(program)

    assert f"func.func @main() -> {return_type}" in lowered.text
    assert expected_op in lowered.text
    assert f"return %" in lowered.text
    assert f": {return_type}" in lowered.text


@pytest.mark.parametrize(
    ("source", "return_type", "constant"),
    [
        ("42", "i32", "arith.constant 42 : i32"),
        ("3.5", "f32", "arith.constant 3.500000e+00 : f32"),
        ("true", "i1", "arith.constant true"),
    ],
)
def test_lowers_scalar_literals(source: str, return_type: str, constant: str):
    lowered = MLIRLowering().lower_program(hir_from_source(source))

    assert f"func.func @main() -> {return_type}" in lowered.text
    assert constant in lowered.text
    assert f": {return_type}" in lowered.text


def test_lowers_scalar_let_with_ssa_value():
    lowered = MLIRLowering().lower_program(hir_from_source("let x = 1 in x + 2"))

    assert "func.func @main() -> i32" in lowered.text
    assert "arith.addi" in lowered.text
    assert "return %" in lowered.text


def test_lowers_scalar_if_expression_to_select():
    lowered = MLIRLowering().lower_program(hir_from_source("if true then 1 else 2"))

    assert "func.func @main() -> i32" in lowered.text
    assert "arith.select" in lowered.text
    assert "return %" in lowered.text


def test_lowers_scalar_hir_function_and_call():
    function = HIRFunction(
        "__double",
        [HIRParam("x", INT)],
        HIRPrimOp("*i", [HIRVar("x", INT), HIRLit(2, INT)], INT),
        INT,
    )
    program = HIRProgram(
        [function],
        HIRCall("__double", [HIRLit(21, INT)], INT),
        INT,
    )

    lowered = MLIRLowering().lower_program(program)

    assert "func.func private @__double(%arg0: i32) -> i32" in lowered.text
    assert "call @__double" in lowered.text
    assert "arith.muli" in lowered.text
    assert "return %" in lowered.text


def test_lowers_iota_to_parseable_linalg_mlir_module():
    program = hir_from_source("iota 10")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<10xi32>" in lowered.text
    assert "tensor.empty() : tensor<10xi32>" in lowered.text
    assert "linalg.generic" in lowered.text
    assert 'iterator_types = ["parallel"]' in lowered.text
    assert "linalg.index 0 : index" in lowered.text
    assert "arith.index_cast" in lowered.text
    assert "return %" in lowered.text
    assert ": tensor<10xi32>" in lowered.text


def test_lowers_reverse_to_parseable_linalg_mlir_module():
    program = hir_from_source("reverse [[1, 2], [3, 4]]")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<2x2xi32>" in lowered.text
    assert "tensor.empty() : tensor<2x2xi32>" in lowered.text
    assert "linalg.generic" in lowered.text
    assert "affine_map<(d0, d1) -> (-d0 + 1, d1)>" in lowered.text
    assert "return %" in lowered.text
    assert ": tensor<2x2xi32>" in lowered.text


def test_lowers_rank_1_array_literal():
    program = hir_from_source("[1, 2, 3]")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<3xi32>" in lowered.text
    assert "tensor.from_elements" in lowered.text
    assert "arith.constant 1 : i32" in lowered.text
    assert "arith.constant 3 : i32" in lowered.text


def test_lowers_rank_2_and_rank_3_array_literals():
    rank2 = MLIRLowering().lower_program(hir_from_source("[[1.0, 2.0], [3.0, 4.0]]"))
    rank3 = MLIRLowering().lower_program(hir_from_source("[[[1], [2]], [[3], [4]]]"))

    assert "func.func @main() -> tensor<2x2xf32>" in rank2.text
    assert "tensor.from_elements" in rank2.text
    assert "func.func @main() -> tensor<2x2x1xi32>" in rank3.text
    assert "tensor.from_elements" in rank3.text


def test_lowers_rank_4_and_rank_10_array_literals():
    rank4 = MLIRLowering().lower_program(hir_from_source(nested_scalar_literal(4)))
    rank10 = MLIRLowering().lower_program(hir_from_source(nested_scalar_literal(10)))

    assert "func.func @main() -> tensor<1x1x1x1xi32>" in rank4.text
    assert "tensor.from_elements" in rank4.text
    assert (
        "func.func @main() -> tensor<1x1x1x1x1x1x1x1x1x1xi32>"
        in rank10.text
    )
    assert "tensor.from_elements" in rank10.text


def test_lowers_static_shape_and_rank():
    shape = MLIRLowering().lower_program(hir_from_source("shape [[1, 2], [3, 4]]"))
    rank = MLIRLowering().lower_program(hir_from_source("rank [[1, 2], [3, 4]]"))
    scalar_shape = MLIRLowering().lower_program(hir_from_source("shape 42"))

    assert "func.func @main() -> tensor<2xi32>" in shape.text
    assert "arith.constant 2 : i32" in shape.text
    assert "tensor.from_elements" in shape.text

    assert "func.func @main() -> i32" in rank.text
    assert "arith.constant 2 : i32" in rank.text

    assert "func.func @main() -> tensor<0xi32>" in scalar_shape.text
    assert "tensor.empty() : tensor<0xi32>" in scalar_shape.text


def test_lowers_full_rank_indexing_to_tensor_extract():
    literal = MLIRLowering().lower_program(hir_from_source("[[1, 2], [3, 4]][1, 0]"))
    iota = MLIRLowering().lower_program(hir_from_source("(iota 10)[3]"))

    assert "func.func @main() -> i32" in literal.text
    assert "tensor.from_elements" in literal.text
    assert "arith.constant 1 : index" in literal.text
    assert "arith.constant 0 : index" in literal.text
    assert "tensor.extract" in literal.text

    assert "func.func @main() -> i32" in iota.text
    assert "tensor.empty() : tensor<10xi32>" in iota.text
    assert "arith.constant 3 : index" in iota.text
    assert "tensor.extract" in iota.text


def test_lowers_partial_indexing_to_rank_reducing_extract_slice():
    lowered = MLIRLowering().lower_program(hir_from_source("[[1, 2], [3, 4]][1]"))

    assert "func.func @main() -> tensor<2xi32>" in lowered.text
    assert "tensor.extract_slice" in lowered.text
    assert "[1, 0] [1, 2] [1, 1]" in lowered.text
    assert "to tensor<2xi32>" in lowered.text


def test_lowers_partial_indexing_from_let_bound_tensor_env():
    lowered = MLIRLowering().lower_program(
        hir_from_source("let xs = [[1, 2], [3, 4]] in xs[1]")
    )

    assert "func.func @main() -> tensor<2xi32>" in lowered.text
    assert "tensor.from_elements" in lowered.text
    assert "tensor.extract_slice" in lowered.text


def test_lowers_primitive_section_map_over_iota():
    program = hir_from_source("map (* 2.0) (iota 10)")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<10xf32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 2
    assert "tensor.empty() : tensor<10xi32>" in lowered.text
    assert "tensor.empty() : tensor<10xf32>" in lowered.text
    assert "arith.sitofp" in lowered.text
    assert "arith.constant 2.000000e+00 : f32" in lowered.text
    assert "arith.mulf" in lowered.text
    assert "return %" in lowered.text
    assert ": tensor<10xf32>" in lowered.text


def test_lowers_rank_2_scalar_map_over_array_literal():
    program = hir_from_source(
        "let xs = [[1.0, 2.0], [3.0, 4.0]] in map (* 2.0) xs"
    )
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<2x2xf32>" in lowered.text
    assert "affine_map<(d0, d1) -> (d0, d1)>" in lowered.text
    assert 'iterator_types = ["parallel", "parallel"]' in lowered.text
    assert "arith.mulf" in lowered.text


def test_lowers_rank_3_scalar_map_over_array_literal():
    program = hir_from_source("let xs = [[[1], [2]], [[3], [4]]] in map (\\x -> x + 1) xs")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<2x2x1xi32>" in lowered.text
    assert "affine_map<(d0, d1, d2) -> (d0, d1, d2)>" in lowered.text
    assert 'iterator_types = ["parallel", "parallel", "parallel"]' in lowered.text
    assert "arith.addi" in lowered.text


def test_lowers_rank_4_scalar_map_over_array_literal():
    program = hir_from_source(
        f"let xs = {nested_scalar_literal(4)} in map (\\x -> x + 1) xs"
    )
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<1x1x1x1xi32>" in lowered.text
    assert "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>" in lowered.text
    assert 'iterator_types = ["parallel", "parallel", "parallel", "parallel"]' in lowered.text
    assert "arith.addi" in lowered.text


def test_lowers_rank_10_scalar_map_over_array_literal():
    program = hir_from_source(
        f"let xs = {nested_scalar_literal(10)} in map (\\x -> x + 1) xs"
    )
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<1x1x1x1x1x1x1x1x1x1xi32>" in lowered.text
    assert (
        "affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> "
        "(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)>"
        in lowered.text
    )
    assert (
        'iterator_types = ["parallel", "parallel", "parallel", "parallel", '
        '"parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]'
        in lowered.text
    )
    assert "arith.addi" in lowered.text


def test_lowers_rank_0_primitive_section_map():
    lowered = MLIRLowering().lower_program(hir_from_source("map (* 2.0) 3.0"))

    assert "func.func @main() -> f32" in lowered.text
    assert "linalg.generic" not in lowered.text
    assert "arith.mulf" in lowered.text
    assert "return %" in lowered.text


def test_lowers_rank_0_lifted_lambda_map():
    lowered = MLIRLowering().lower_program(hir_from_source("map (\\x -> x + 1) 3"))

    assert "func.func @main() -> i32" in lowered.text
    assert "linalg.generic" not in lowered.text
    assert "arith.addi" in lowered.text
    assert "return %" in lowered.text


def test_lowers_rank_0_comparison_lambda_map_with_let():
    lowered = MLIRLowering().lower_program(
        hir_from_source("let x = 3 in map (\\y -> y < 5) x")
    )

    assert "func.func @main() -> i1" in lowered.text
    assert "arith.cmpi slt" in lowered.text


def test_lowers_lifted_lambda_map_over_iota():
    program = hir_from_source("map (\\x -> x * 2.0) (iota 10)")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<10xf32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 2
    assert "arith.sitofp" in lowered.text
    assert "arith.constant 2.000000e+00 : f32" in lowered.text
    assert "arith.mulf" in lowered.text


def test_lowers_lifted_integer_lambda_map_over_iota():
    program = hir_from_source("map (\\x -> x * x) (iota 10)")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<10xi32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 2
    assert "arith.muli" in lowered.text
    assert "arith.sitofp" not in lowered.text


def test_lowers_lifted_comparison_lambda_map_over_iota():
    program = hir_from_source("map (\\x -> x < 5) (iota 10)")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<10xi1>" in lowered.text
    assert lowered.text.count("linalg.generic") == 2
    assert "arith.cmpi slt" in lowered.text


def test_lowers_nested_scalar_maps_over_iota():
    lowered = MLIRLowering().lower_program(
        hir_from_source("map (* 3) (map (* 2) (iota 10))")
    )

    assert "func.func @main() -> tensor<10xi32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 3
    assert lowered.text.count("arith.muli") == 2


def test_lowers_nested_lifted_lambda_map_over_array_literal():
    lowered = MLIRLowering().lower_program(
        hir_from_source("let xs = [1, 2, 3] in map (\\x -> x + 1) (map (* 2) xs)")
    )

    assert "func.func @main() -> tensor<3xi32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 2
    assert "arith.muli" in lowered.text
    assert "arith.addi" in lowered.text


def test_lowers_vector_cell_map_over_rank_2_array():
    lowered = MLIRLowering().lower_program(
        hir_from_source(
            "let xs = [[1.0, 2.0], [3.0, 4.0]] in "
            "map (\\row -> fold (+) 0.0 row) xs"
        )
    )

    assert "func.func @main() -> tensor<2xf32>" in lowered.text
    assert "linalg.fill" in lowered.text
    assert "affine_map<(d0, d1) -> (d0, d1)>" in lowered.text
    assert "affine_map<(d0, d1) -> (d0)>" in lowered.text
    assert 'iterator_types = ["parallel", "reduction"]' in lowered.text
    assert "arith.addf" in lowered.text


def test_lowers_vector_cell_map_over_rank_3_array():
    lowered = MLIRLowering().lower_program(
        hir_from_source(
            "let xs = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] in "
            "map (\\row -> fold (+) 0 row) xs"
        )
    )

    assert "func.func @main() -> tensor<2x2xi32>" in lowered.text
    assert "affine_map<(d0, d1, d2) -> (d0, d1, d2)>" in lowered.text
    assert "affine_map<(d0, d1, d2) -> (d0, d1)>" in lowered.text
    assert 'iterator_types = ["parallel", "parallel", "reduction"]' in lowered.text
    assert "arith.addi" in lowered.text


def test_lowers_fold_over_iota():
    program = hir_from_source("fold (+) 0 (iota 10)")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> i32" in lowered.text
    assert lowered.text.count("linalg.generic") == 2
    assert 'iterator_types = ["reduction"]' in lowered.text
    assert "tensor.from_elements" in lowered.text
    assert "arith.addi" in lowered.text
    assert "tensor.extract" in lowered.text
    assert "return %" in lowered.text
    assert ": i32" in lowered.text


def test_lowers_scalar_fold_with_lifted_lambda():
    program = hir_from_source("fold (\\acc x -> acc + x) 0 (iota 4)")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> i32" in lowered.text
    assert 'iterator_types = ["reduction"]' in lowered.text
    assert "arith.addi" in lowered.text
    assert "linalg.yield" in lowered.text


def test_lowers_scalar_fold_with_nonliteral_init_expression():
    program = hir_from_source("fold (+) (1 - 1) (iota 4)")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> i32" in lowered.text
    assert 'iterator_types = ["reduction"]' in lowered.text
    assert lowered.text.count("arith.subi") == 1
    assert "tensor.from_elements" in lowered.text


def test_lowers_fold_over_array_literal():
    program = hir_from_source("let xs = [1.0, 2.0, 3.0] in fold (+) 0.0 xs")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> f32" in lowered.text
    assert "tensor.from_elements" in lowered.text
    assert 'iterator_types = ["reduction"]' in lowered.text
    assert "arith.addf" in lowered.text


def test_lowers_fold_over_rank_2_array_literal():
    program = hir_from_source(
        "let init = [0, 0] in let xs = [[1, 2], [3, 4]] in fold (+) init xs"
    )
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<2xi32>" in lowered.text
    assert "affine_map<(d0, d1) -> (d0, d1)>" in lowered.text
    assert "affine_map<(d0, d1) -> (d1)>" in lowered.text
    assert 'iterator_types = ["reduction", "parallel"]' in lowered.text
    assert "arith.addi" in lowered.text
    assert "tensor.extract" not in lowered.text


def test_lowers_array_cell_fold_with_mapped_init():
    program = hir_from_source(
        "let init = map (* 0) (iota 2) in "
        "let xs = [[1, 2], [3, 4]] in "
        "fold (+) init xs"
    )
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<2xi32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 3
    assert 'iterator_types = ["reduction", "parallel"]' in lowered.text
    assert "arith.muli" in lowered.text
    assert "arith.addi" in lowered.text


def test_lowers_fold_over_rank_3_array_literal():
    program = hir_from_source(
        "let init = [[0], [0]] in "
        "let xs = [[[1], [2]], [[3], [4]]] in "
        "fold (+) init xs"
    )
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<2x1xi32>" in lowered.text
    assert "affine_map<(d0, d1, d2) -> (d0, d1, d2)>" in lowered.text
    assert "affine_map<(d0, d1, d2) -> (d1, d2)>" in lowered.text
    assert 'iterator_types = ["reduction", "parallel", "parallel"]' in lowered.text
    assert "arith.addi" in lowered.text


def test_lowers_fold_over_rank_4_array_literal():
    program = hir_from_source(
        f"let init = {nested_scalar_literal(3, '0')} in "
        f"let xs = {nested_scalar_literal(4)} in "
        "fold (+) init xs"
    )
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<1x1x1xi32>" in lowered.text
    assert "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>" in lowered.text
    assert "affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>" in lowered.text
    assert 'iterator_types = ["reduction", "parallel", "parallel", "parallel"]' in lowered.text
    assert "arith.addi" in lowered.text


def test_lowers_fold_over_mapped_iota_milestone_shape():
    program = hir_from_source("fold (+) 0.0 (map (* 2.0) (iota 10))")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> f32" in lowered.text
    assert lowered.text.count("linalg.generic") == 3
    assert lowered.text.count('iterator_types = ["parallel"]') == 2
    assert 'iterator_types = ["reduction"]' in lowered.text
    assert "arith.sitofp" in lowered.text
    assert "arith.mulf" in lowered.text
    assert "arith.addf" in lowered.text
    assert "tensor.extract" in lowered.text


def test_lowers_let_bound_iota_map():
    program = hir_from_source("let xs = iota 10 in map (* 2.0) xs")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<10xf32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 2
    assert "arith.mulf" in lowered.text


def test_lowers_chained_tensor_lets_through_tensor_env():
    program = hir_from_source(
        "let xs = iota 4 in let ys = map (* 2) xs in map (+ 1) ys"
    )
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<4xi32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 3
    assert "arith.muli" in lowered.text
    assert "arith.addi" in lowered.text


def test_lowers_top_level_value_definition_map():
    program = hir_from_source("def xs = iota 10\nmap (* 2.0) xs")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<10xf32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 2
    assert "arith.mulf" in lowered.text


def test_lowers_top_level_function_direct_call_by_inlining_static_lambda():
    program = hir_from_source("def add1 x = x + 1\nadd1 41")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> i32" in lowered.text
    assert "arith.addi" in lowered.text


def test_lowers_top_level_function_as_map_callable():
    program = hir_from_source("def double x = x * 2\nmap double (iota 4)")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<4xi32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 2
    assert "arith.muli" in lowered.text


def test_lowers_let_bound_iota_fold():
    program = hir_from_source("let xs = iota 10 in fold (+) 0 xs")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> i32" in lowered.text
    assert 'iterator_types = ["reduction"]' in lowered.text
    assert "arith.addi" in lowered.text


def test_non_rank_1_cell_map_lowering_is_deferred():
    program = hir_from_source(
        "let xs = [[[1], [2]], [[3], [4]]] in map (\\matrix -> fold (+) [0] matrix) xs"
    )

    with pytest.raises(RemoraLoweringError, match="rank-1 cell maps"):
        MLIRLowering().lower_program(program)


def test_lowers_rank_1_binary_scalar_map_over_array_literals():
    program = hir_from_source("let xs = [1, 2] in let ys = [3, 4] in map (*) xs ys")
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<2xi32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 1
    assert "indexing_maps = [#map, #map, #map]" in lowered.text
    assert 'iterator_types = ["parallel"]' in lowered.text
    assert "ins(%from_elements, %from_elements_0 : tensor<2xi32>, tensor<2xi32>)" in lowered.text
    assert "arith.muli" in lowered.text


def test_lowers_rank_0_binary_scalar_map():
    lowered = MLIRLowering().lower_program(hir_from_source("map (*) 2 3"))

    assert "func.func @main() -> i32" in lowered.text
    assert "linalg.generic" not in lowered.text
    assert "arith.muli" in lowered.text
    assert "return %" in lowered.text


def test_lowers_rank_2_binary_scalar_map_over_array_literals():
    program = hir_from_source(
        "let xs = [[1, 2], [3, 4]] in "
        "let ys = [[5, 6], [7, 8]] in "
        "map (+) xs ys"
    )
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<2x2xi32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 1
    assert "affine_map<(d0, d1) -> (d0, d1)>" in lowered.text
    assert "indexing_maps = [#map, #map, #map]" in lowered.text
    assert 'iterator_types = ["parallel", "parallel"]' in lowered.text
    assert "tensor<2x2xi32>, tensor<2x2xi32>)" in lowered.text
    assert "arith.addi" in lowered.text


def test_lowers_rank_3_binary_scalar_map_over_array_literals():
    program = hir_from_source(
        "let xs = [[[1], [2]], [[3], [4]]] in "
        "let ys = [[[5], [6]], [[7], [8]]] in "
        "map (+) xs ys"
    )
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<2x2x1xi32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 1
    assert "affine_map<(d0, d1, d2) -> (d0, d1, d2)>" in lowered.text
    assert "indexing_maps = [#map, #map, #map]" in lowered.text
    assert 'iterator_types = ["parallel", "parallel", "parallel"]' in lowered.text
    assert "arith.addi" in lowered.text


def test_lowers_rank_4_binary_scalar_map_over_array_literals():
    program = hir_from_source(
        f"let xs = {nested_scalar_literal(4, '1')} in "
        f"let ys = {nested_scalar_literal(4, '2')} in "
        "map (+) xs ys"
    )
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<1x1x1x1xi32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 1
    assert "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>" in lowered.text
    assert 'iterator_types = ["parallel", "parallel", "parallel", "parallel"]' in lowered.text
    assert "arith.addi" in lowered.text


def test_lowers_binary_lifted_lambda_map_over_array_literals():
    program = hir_from_source(
        "let xs = [1, 2] in let ys = [3, 4] in map (\\x y -> x + y) xs ys"
    )
    lowered = MLIRLowering().lower_program(program)

    assert "func.func @main() -> tensor<2xi32>" in lowered.text
    assert lowered.text.count("linalg.generic") == 1
    assert "arith.addi" in lowered.text


def test_lowers_prelude_dot_as_binary_map_plus_fold():
    mlir = compile_source_to_mlir(
        "let xs = [1.0, 2.0, 3.0] in "
        "let ys = [4.0, 5.0, 6.0] in "
        "dot xs ys",
        verify=False,
    )

    assert "func.func @main() -> f32" in mlir
    assert mlir.count("linalg.generic") == 2
    assert "indexing_maps = [#map, #map, #map]" in mlir
    assert 'iterator_types = ["parallel"]' in mlir
    assert 'iterator_types = ["reduction"]' in mlir
    assert "arith.mulf" in mlir
    assert "arith.addf" in mlir


def test_lowers_scalar_output_descriptor_export():
    program = hir_from_source("1 + 2.0")
    lowered = MLIRLowering().lower_program(program, export_output_descriptor=True)

    assert "func.func @remora_main_out(%arg0: memref<f32>) attributes {llvm.emit_c_interface}" in lowered.text
    assert "call @main() : () -> f32" in lowered.text
    assert "memref.store %0, %arg0[] : memref<f32>" in lowered.text


def test_lowers_ranked_output_descriptor_export_with_strided_stores():
    program = hir_from_source("map (* 2.0) (iota 4)")
    lowered = MLIRLowering().lower_program(program, export_output_descriptor=True)

    assert (
        "func.func @remora_main_out(%arg0: memref<4xf32, strided<[?], offset: ?>>) "
        "attributes {llvm.emit_c_interface}" in lowered.text
    )
    assert "call @main() : () -> tensor<4xf32>" in lowered.text
    assert "scf.for" in lowered.text
    assert "tensor.extract" in lowered.text
    assert "memref.store" in lowered.text


def test_lowers_descriptor_input_function_export():
    artifact = compile_function_source(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        verify=False,
    )

    assert "func.func private @__remora_entry(%arg0: tensor<4xf32>) -> tensor<4xf32>" in artifact.mlir_text
    assert (
        "func.func @remora_call(%arg0: memref<4xf32, strided<[?], offset: ?>>, "
        "%arg1: memref<4xf32, strided<[?], offset: ?>>) attributes {llvm.emit_c_interface}"
        in artifact.mlir_text
    )
    assert "bufferization.to_tensor" in artifact.mlir_text
    assert "call @__remora_entry" in artifact.mlir_text


def test_builder_region_emitter_produces_correct_text():
    """Stream E2: verify _BuilderRegionEmitter produces equivalent text to
    _RegionEmitter for basic scalar expressions."""
    from importlib import import_module

    ir = import_module("iree.compiler.ir")
    from iree.compiler.dialects import func as func_dialect

    from remora.lowering._builder_emitter import _BuilderRegionEmitter

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    loc = ir.Location.unknown(ctx)

    with ctx, loc:
        module = ir.Module.create(loc)
        fn_type = ir.FunctionType.get([], [ir.F32Type.get()])
        main_op = func_dialect.FuncOp(
            "main", fn_type, ip=ir.InsertionPoint(module.body)
        )
        entry_block = main_op.add_entry_block()
        emitter = _BuilderRegionEmitter(entry_block)

        # (3.14 + 2.0) * 2.0
        a = emitter.emit_expr(HIRLit(3.14, FLOAT), {})
        b = emitter.emit_expr(HIRLit(2.0, FLOAT), {})

        lines = "\n".join(emitter.lines)
        assert "arith.constant 3.140000e+00 : f32" in lines
        assert "arith.constant 2.000000e+00 : f32" in lines
        assert "%v0" in lines
        assert "%v1" in lines
        assert a.ir_value is not None
        assert b.ir_value is not None

        # Test a primitive op: +f
        emitter2 = _BuilderRegionEmitter(entry_block, next_temp=emitter.next_temp)
        add_expr = HIRPrimOp("+f", [HIRLit(1.0, FLOAT), HIRLit(2.0, FLOAT)], FLOAT)
        result = emitter2.emit_expr(add_expr, {})
        assert "arith.addf" in "\n".join(emitter2.lines)
        assert result.ir_value is not None
        assert result.type == "f32"


def test_builder_region_emitter_literal_types():
    """Stream E2: verify _BuilderRegionEmitter handles all scalar literal types."""
    from importlib import import_module

    ir = import_module("iree.compiler.ir")
    from iree.compiler.dialects import func as func_dialect

    from remora.lowering._builder_emitter import _BuilderRegionEmitter

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    loc = ir.Location.unknown(ctx)

    with ctx, loc:
        module = ir.Module.create(loc)
        fn_type = ir.FunctionType.get([], [ir.F32Type.get()])
        main_op = func_dialect.FuncOp(
            "main", fn_type, ip=ir.InsertionPoint(module.body)
        )
        entry_block = main_op.add_entry_block()
        emitter = _BuilderRegionEmitter(entry_block)

        i32_lit = emitter.emit_expr(HIRLit(42, INT), {})
        assert i32_lit.type == "i32"
        assert "arith.constant 42 : i32" in "\n".join(emitter.lines)
        assert i32_lit.ir_value is not None

        f32_lit = emitter.emit_expr(HIRLit(3.14, FLOAT), {})
        assert f32_lit.type == "f32"
        assert f32_lit.ir_value is not None

        bool_lit = emitter.emit_expr(HIRLit(True, BOOL), {})
        assert bool_lit.type == "i1"
        assert "arith.constant true" in "\n".join(emitter.lines)
        assert bool_lit.ir_value is not None


# ---------------------------------------------------------------------------
# Builder ops tests (E3-E5)
# ---------------------------------------------------------------------------


def _builder_lower_and_parse(source: str) -> tuple[str, object]:
    """Lower *source* via the builder API and return (text, parsed_module)."""
    from importlib import import_module

    ir_mod = import_module("iree.compiler.ir")
    from remora.compiler import compile_source
    from remora.lowering._builder_ops import lower_program_via_builder

    artifact = compile_source(source, verify=False)
    mlir_text, _ = lower_program_via_builder(artifact.hir)

    ctx = ir_mod.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir_mod.Location.unknown(ctx):
        parsed = ir_mod.Module.parse(mlir_text)
    return mlir_text, parsed


def test_builder_iota_produces_valid_mlir():
    text, mod = _builder_lower_and_parse("iota 4")
    assert "linalg.generic" in text
    assert "linalg.index" in text
    assert "tensor<4xi32>" in text


def test_builder_unary_map_produces_valid_mlir():
    text, mod = _builder_lower_and_parse("map (* 2) (iota 4)")
    assert "linalg.generic" in text
    assert "tensor<4xi32>" in text


def test_builder_fold_produces_valid_mlir():
    text, mod = _builder_lower_and_parse("fold (+) 0 (iota 4)")
    assert "linalg.generic" in text
    assert 'iterator_types = ["reduction"]' in text or 'reduction' in text
    assert "tensor.extract" in text.lower()


def test_builder_reverse_produces_valid_mlir():
    text, mod = _builder_lower_and_parse("reverse (iota 4)")
    assert "linalg.generic" in text
    assert "tensor<4xi32>" in text


def test_builder_take_produces_valid_mlir():
    text, mod = _builder_lower_and_parse("take 2 (iota 4)")
    assert "tensor.extract_slice" in text
    assert "tensor<2xi32>" in text


def test_builder_drop_produces_valid_mlir():
    text, mod = _builder_lower_and_parse("drop 2 (iota 4)")
    assert "tensor.extract_slice" in text
    assert "tensor<2xi32>" in text


def test_builder_iota_and_text_equivalent():
    """Verify builder and text-based lowering produce equivalent modules."""
    from remora.compiler import compile_source

    source = "iota 4"
    artifact = compile_source(source, verify=False)
    text_lowered = artifact.mlir_text

    from remora.lowering._builder_ops import lower_program_via_builder
    builder_text, _ = lower_program_via_builder(artifact.hir)

    import re
    # Both should contain linalg.generic
    assert "linalg.generic" in text_lowered
    assert "linalg.generic" in builder_text
    # Same return type
    assert "tensor<4xi32>" in builder_text


# ---------------------------------------------------------------------------
# Example program regression tests
# ---------------------------------------------------------------------------


_EXAMPLE_DIR = Path(__file__).parent.parent / "examples"

_EXPECTED_FAILURES = {
    # Known deferred features from plan
    "function_application.remora": "dynamic higher-order functions are deferred",
    "row_norms.remora": "lambda captures outer variables",
}

_EXAMPLE_GPU_EXPECTED = {
    "chained_maps.remora",
    "lift_map.remora",
    "prelude_scale.remora",
    "prelude_sum.remora",
    "reduce_iota.remora",
    "section_right.remora",
    "threshold_mask.remora",
    "top_level_value.remora",
}


@pytest.mark.parametrize("example_path", [
    p for p in sorted(_EXAMPLE_DIR.glob("*.remora"))
])
def test_example_compiles_to_mlir(example_path):
    """Every example either compiles or is a known deferred feature."""
    source = example_path.read_text()
    fname = example_path.name

    if fname in _EXPECTED_FAILURES:
        with pytest.raises(RemoraError, match=_EXPECTED_FAILURES[fname]):
            compile_source(source, verify=False, include_prelude=True)
        return

    compile_source(source, verify=False, include_prelude=True)


@pytest.mark.parametrize("example_path", [
    p for p in sorted(_EXAMPLE_DIR.glob("*.remora"))
    if p.name not in _EXPECTED_FAILURES
])
def test_example_executes_on_cpu(example_path):
    """Every compilable example executes correctly on CPU."""
    source = example_path.read_text()
    result = evaluate_source_compiled(source)
    assert result is not None
    assert result.value is not None


@pytest.mark.skipif(
    importlib.util.find_spec("iree") is None,
    reason="IREE compiler MLIR bindings are not installed",
)
@pytest.mark.parametrize("example_path", [
    p for p in sorted(_EXAMPLE_DIR.glob("*.remora"))
    if p.name in _EXAMPLE_GPU_EXPECTED
])
def test_example_compiles_to_gpu_ptx(example_path):
    """Examples expected to work on GPU produce valid PTX."""
    from remora.codegen import generate_ptx

    source = example_path.read_text()
    artifact = compile_source(source, verify=False, include_prelude=True)
    ptx, kernels = generate_ptx(artifact.mlir_module)
    assert len(kernels) >= 1
    assert ".version" in ptx
