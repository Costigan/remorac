"""Erase typed core programs to the existing backend HIR."""

from __future__ import annotations

from remora.core_verify import verify_core_program
from remora.elaborated import CoreProgram
from remora.hir import HIRProgram, lower_to_hir


def erase_to_hir(program: CoreProgram) -> HIRProgram:
    """Erase the current typed core into backend-oriented HIR."""
    verify_core_program(program)
    return lower_to_hir(program.typed)
