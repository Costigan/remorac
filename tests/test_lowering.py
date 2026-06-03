import importlib.util
import inspect
from pathlib import Path

import pytest

from remora.compiler import compile_source_to_mlir
from remora.compiler import compile_function_source
from remora.defunc import defunctionalize
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
