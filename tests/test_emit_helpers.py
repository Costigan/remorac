"""Unit tests for shared lowering emission helpers."""

from remora.lowering._emit_helpers import (
    emit_2d_decompose,
    emit_delinearize,
    llvm_op,
)


class TestLlvmOp:
    def test_f32_add(self):
        assert llvm_op("+", "f32") == "llvm.fadd"

    def test_f32_sub(self):
        assert llvm_op("-", "f32") == "llvm.fsub"

    def test_f32_mul(self):
        assert llvm_op("*", "f32") == "llvm.fmul"

    def test_f32_div(self):
        assert llvm_op("/", "f32") == "llvm.fdiv"

    def test_i32_add(self):
        assert llvm_op("+", "i32") == "llvm.add"

    def test_i32_sub(self):
        assert llvm_op("-", "i32") == "llvm.sub"

    def test_i32_mul(self):
        assert llvm_op("*", "i32") == "llvm.mul"

    def test_i32_sdiv(self):
        assert llvm_op("/", "i32") == "llvm.sdiv"

    def test_bool_and(self):
        assert llvm_op("&&", "bool") == "llvm.and"

    def test_bool_or(self):
        assert llvm_op("||", "bool") == "llvm.or"

    def test_f32_cmp_eq(self):
        assert llvm_op("==", "f32") == 'llvm.fcmp "oeq"'

    def test_f32_cmp_ne(self):
        assert llvm_op("!=", "f32") == 'llvm.fcmp "one"'

    def test_f32_cmp_lt(self):
        assert llvm_op("<", "f32") == 'llvm.fcmp "olt"'

    def test_i32_cmp_eq(self):
        assert llvm_op("==", "i32") == 'llvm.icmp "eq"'

    def test_i32_cmp_lt(self):
        assert llvm_op("<", "i32") == 'llvm.icmp "slt"'

    def test_i32_cmp_ge(self):
        assert llvm_op(">=", "i32") == 'llvm.icmp "sge"'


class TestEmitDelinearize:
    def test_rank_0_returns_empty(self):
        result = emit_delinearize("%idx", [])
        assert result == []

    def test_rank_1_llvm(self):
        result = emit_delinearize("%idx", ["10"])
        assert len(result) == 1
        assert "llvm." in result[0] or "%i0 =" in result[0]

    def test_rank_2_llvm(self):
        result = emit_delinearize("%idx", ["3", "5"])
        assert len(result) >= 4
        assert any("plane0" in line for line in result)
        assert any("llvm.udiv" in line for line in result)
        assert any("llvm.urem" in line for line in result)

    def test_rank_3_llvm(self):
        result = emit_delinearize("%tid", ["2", "3", "4"])
        assert any("plane0" in line for line in result)
        assert any("plane1" in line for line in result)
        assert any("llvm.udiv" in line for line in result)

    def test_rank_2_arith(self):
        result = emit_delinearize("%idx", ["3", "5"], dialect="arith")
        assert any("arith.divui" in line for line in result)
        assert any("arith.remui" in line for line in result)

    def test_produces_valid_mlir_variable_names(self):
        result = emit_delinearize("%my_idx", ["4", "8"])
        for line in result:
            if "%" in line:
                assert line.strip().startswith("%"), f"variable should start at line start: {line}"


class TestEmit2dDecompose:
    def test_llvm(self):
        result = emit_2d_decompose("%tid", "row", "col", "5")
        assert len(result) == 2
        assert "llvm.udiv" in result[0]
        assert "llvm.urem" in result[1]
        assert "%row" in result[0]
        assert "%col" in result[1]

    def test_arith(self):
        result = emit_2d_decompose("%tid", "r", "c", "5", dialect="arith")
        assert "arith.divui" in result[0]
        assert "arith.remui" in result[1]

    def test_indent(self):
        result = emit_2d_decompose("%tid", "r", "c", "5", indent="    ")
        for line in result:
            assert line.startswith("    ")
