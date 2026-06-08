import pytest

from remora.core_verify import CoreVerificationError, verify_core_program
from remora.elaborate import elaborate_program
from remora.erase import erase_to_hir
from remora.hir import HIRIota
from remora.parser import parse_program
from remora.typechecker import TypeChecker
from remora.types import INT, ArrayType, StaticDim


def elaborate_source(source: str):
    typed = TypeChecker().check_program(parse_program(source))
    return elaborate_program(typed)


def test_monomorphic_program_elaborates_and_erases_to_existing_hir():
    core = elaborate_source("iota 4")
    hir = erase_to_hir(core)

    assert core.body is not None
    assert core.body.kind == "TypedExprNode"
    assert not hasattr(core, "typed")
    assert hir.return_type == ArrayType(INT, (StaticDim(4),))
    assert isinstance(hir.main, HIRIota)


def test_core_verifier_accepts_definition_only_program_without_result_type():
    core = elaborate_source("def xs = iota 4")

    verify_core_program(core)


def test_core_verifier_rejects_inconsistent_program_type():
    core = elaborate_source("1")
    broken = type(core)(
        core.definitions,
        core.body,
        ArrayType(INT, (StaticDim(1),)),
        core.index_applications,
        core.specializations,
    )

    with pytest.raises(CoreVerificationError, match="body type"):
        verify_core_program(broken)
