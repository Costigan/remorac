"""Executable capability matrix for RemoraC backends.

One structured, importable source of truth for operation support and
cost-relevant properties, consumed by docs, tests, ``--explain-lowering``,
and the route registry.

Entries are declared statically and describe the **current** dense-core
implementation.  All ``dynamic_shape`` and ``boxed_ragged`` fields are
``False`` because the compiler only supports static shapes and box/ragged
constructs are type-erasure only.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal

from remora.limits import MAX_DENSE_RANK

RANK_ANY: object = None


class Backend(enum.Enum):
    INTERP = "interp"
    CPU = "cpu"
    GPU = "gpu"


class Context(enum.Enum):
    TOP_LEVEL = "top-level"
    MAP_BODY = "map-body"
    FOLD_BODY = "fold-body"
    SCAN_BODY = "scan-body"
    AD_GENERATED = "ad-generated"


@dataclass(frozen=True)
class Capability:
    op: str
    backend: Backend
    dtypes: frozenset[str]
    ranks: tuple[int, ...] | None
    static_shape: bool
    dynamic_shape: bool
    boxed_ragged: bool
    contexts: frozenset[Context]
    work_estimate: str | None
    memory_estimate: str | None
    requires_launch_transfer: bool
    fallback: Backend | None
    unsupported_reason: str | None
    status: Literal["supported", "limited", "unsupported"]

    @property
    def unique_key(self) -> tuple[str, str, frozenset[str], frozenset[Context]]:
        return (self.op, self.backend.value, self.dtypes, self.contexts)


DTYPES_F32 = frozenset({"f32"})
DTYPES_F32_64 = frozenset({"f32", "f64"})
DTYPES_F32_I32 = frozenset({"f32", "i32"})
DTYPES_F32_I32_BOOL = frozenset({"f32", "i32", "bool"})
DTYPES_F32_I32_BOOL_F64 = frozenset({"f32", "i32", "bool", "f64"})
DTYPES_I32 = frozenset({"i32"})
DTYPES_BOOL = frozenset({"bool"})

ALL_CONTEXTS = frozenset(Context)
TOP_MAP_FOLD = frozenset({Context.TOP_LEVEL, Context.MAP_BODY, Context.FOLD_BODY})


def _register(
    op: str,
    backend: Backend,
    dtypes: frozenset[str] = DTYPES_F32_I32,
    ranks: tuple[int, ...] | None = RANK_ANY,
    static_shape: bool = True,
    dynamic_shape: bool = False,
    boxed_ragged: bool = False,
    contexts: frozenset[Context] | None = None,
    work_estimate: str | None = None,
    memory_estimate: str | None = None,
    requires_launch_transfer: bool = False,
    fallback: Backend | None = None,
    unsupported_reason: str | None = None,
    status: Literal["supported", "limited", "unsupported"] = "supported",
) -> Capability:
    return Capability(
        op=op,
        backend=backend,
        dtypes=dtypes,
        ranks=ranks,
        static_shape=static_shape,
        dynamic_shape=dynamic_shape,
        boxed_ragged=boxed_ragged,
        contexts=contexts if contexts is not None else ALL_CONTEXTS,
        work_estimate=work_estimate,
        memory_estimate=memory_estimate,
        requires_launch_transfer=requires_launch_transfer,
        fallback=fallback,
        unsupported_reason=unsupported_reason,
        status=status,
    )


def _build_registry() -> tuple[Capability, ...]:
    entries: list[Capability] = []

    def add(op, backend, **kw):
        entries.append(_register(op, backend, **kw))

    _interp = Backend.INTERP
    _cpu = Backend.CPU
    _gpu = Backend.GPU

    for backend in (_interp, _cpu):
        add("scalar_arith", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("scalar_logic", backend, dtypes=DTYPES_BOOL)
        add("let", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("if", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("lambda", backend)
        add("var_ref", backend)
        add("call", backend, dtypes=DTYPES_F32_I32_BOOL_F64)

        add("map", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("fold", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("reduce", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("fold_right", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("scan", backend, dtypes=DTYPES_F32_I32_BOOL_F64)

        add("reverse", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("transpose", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("reshape", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("ravel", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("take", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("drop", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("slice", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("subarray", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("index", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("iota", backend, dtypes=DTYPES_F32_I32_BOOL_F64)

        add("sort", backend, dtypes=DTYPES_F32_I32)
        add("grade", backend, dtypes=DTYPES_F32_I32)
        add("append", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("rotate", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("matmul", backend, dtypes=DTYPES_F32_64)
        add("reshape_ravel", backend, dtypes=DTYPES_F32_I32_BOOL_F64)

        add("pair", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("first", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("second", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("box", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("unbox", backend, dtypes=DTYPES_F32_I32_BOOL_F64)

        add("im2col", backend, dtypes=DTYPES_F32)
        add("col2im", backend, dtypes=DTYPES_F32)
        add("compose", backend, dtypes=DTYPES_F32_I32_BOOL_F64)

        add("filter", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("replicate", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("indices_of", backend, dtypes=DTYPES_F32_I32)
        add("with_shape", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("scatter_add", backend, dtypes=DTYPES_F32_I32_BOOL_F64)

        add("define_plain", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("define_pi", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("define_forall", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("recursion", backend, dtypes=DTYPES_F32_I32_BOOL_F64)

        add("rerank", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("grad", backend, dtypes=DTYPES_F32, contexts=ALL_CONTEXTS)

        add("cast", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("array_lit", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("closure", backend, dtypes=DTYPES_F32_I32_BOOL_F64)
        add("hof_call", backend, dtypes=DTYPES_F32_I32_BOOL_F64)

    add("len", _interp, dtypes=DTYPES_F32_I32_BOOL_F64)
    add("len", _cpu, dtypes=DTYPES_F32_I32_BOOL_F64)

    add("shape", _interp, dtypes=DTYPES_F32_I32_BOOL_F64)
    add("shape", _cpu, dtypes=DTYPES_F32_I32_BOOL_F64)
    add("rank", _interp, dtypes=DTYPES_F32_I32_BOOL_F64)
    add("rank", _cpu, dtypes=DTYPES_F32_I32_BOOL_F64)

    # ── GPU entries ──

    add("scalar_arith", _gpu, dtypes=DTYPES_F32,
        ranks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        contexts=TOP_MAP_FOLD, status="limited",
        unsupported_reason="f32 element-wise arithmetic: supported via narrow direct/kernel path; general-expr emitter provides broader coverage")
    add("scalar_arith", _gpu, dtypes=DTYPES_I32,
        ranks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        contexts=TOP_MAP_FOLD, status="limited",
        unsupported_reason="i32 element-wise arithmetic: supported via narrow direct kernel path (1-2 inputs, literal sections)")
    add("scalar_arith", _gpu, dtypes=DTYPES_BOOL, status="unsupported",
        unsupported_reason="bool arithmetic not semantically meaningful")
    add("scalar_arith", _gpu, dtypes=DTYPES_F32_64,
        ranks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        contexts=TOP_MAP_FOLD, status="limited",
        unsupported_reason="f64 element-wise arithmetic: GPU lowering maps f64 to f32 at the narrow map gate")

    add("scalar_logic", _gpu, dtypes=DTYPES_BOOL,
        ranks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        contexts=TOP_MAP_FOLD, status="limited",
        unsupported_reason="bool logic ops (&&, ||, ==, !=) supported via narrow bool map kernel")
    add("let", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="limited",
        unsupported_reason="let in GPU map bodies via general-expr emitter only")
    add("if", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="limited",
        unsupported_reason="if in GPU via general-expr emitter; array-typed conditionals supported")
    add("lambda", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="limited",
        unsupported_reason="lambda in GPU limited to map/fold bodies via general-expr emitter")

    add("map", _gpu, dtypes=DTYPES_F32,
        ranks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        status="limited",
        unsupported_reason="f32 map: direct kernel supports 1-2 inputs, literal sections; fused expression supports 1-10 inputs with subarrays")
    add("map", _gpu, dtypes=DTYPES_I32,
        ranks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        status="limited",
        unsupported_reason="i32 map: narrow direct kernel for 1-2 inputs, literal sections")
    add("map", _gpu, dtypes=DTYPES_BOOL,
        ranks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        contexts=TOP_MAP_FOLD, status="limited",
        unsupported_reason="bool map via narrow direct kernel")
    add("map", _gpu, dtypes=DTYPES_F32_64,
        ranks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        status="limited",
        unsupported_reason="f64 map: lowered through general-expr emitter; f64 mapped to f32 in narrow gate")

    add("fold", _gpu, dtypes=DTYPES_F32,
        contexts=frozenset({Context.TOP_LEVEL}), status="limited",
        unsupported_reason="f32 reduction kernel: scalar result only; fold body limited to supported ops")
    add("fold", _gpu, dtypes=DTYPES_I32,
        contexts=frozenset({Context.TOP_LEVEL}), status="limited",
        unsupported_reason="i32 fold: routed through general reduction path or general-expr map")
    add("fold", _gpu, dtypes=DTYPES_BOOL, status="unsupported",
        unsupported_reason="bool fold not supported in GPU direct kernels")
    add("fold", _gpu, dtypes=DTYPES_F32_64, status="unsupported",
        unsupported_reason="f64 fold not supported in GPU direct kernels")

    add("reduce", _gpu, dtypes=DTYPES_F32,
        contexts=frozenset({Context.TOP_LEVEL}), status="limited",
        unsupported_reason="f32 reduction via f32_reduction or compound_fold or general-expr map")

    add("scan", _gpu, dtypes=DTYPES_F32, status="limited",
        unsupported_reason="f32 scan: single-block and multi-block variants; limited to supported map-body ops")
    add("scan", _gpu, dtypes=DTYPES_I32, status="unsupported",
        unsupported_reason="i32 scan not implemented in GPU kernels")
    add("scan", _gpu, dtypes=DTYPES_BOOL, status="unsupported",
        unsupported_reason="bool scan not implemented in GPU kernels")
    add("scan", _gpu, dtypes=DTYPES_F32_64, status="unsupported",
        unsupported_reason="f64 scan not implemented in GPU kernels")

    for _view_op in ("reverse", "rotate", "take", "drop", "slice", "subarray",
                     "reshape", "ravel", "transpose"):
        add(_view_op, _gpu, dtypes=DTYPES_F32, status="limited",
            unsupported_reason=f"{_view_op}: GPU view kernel hardcodes f32 element type")

    add("index", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="limited",
        unsupported_reason="index via general-expr emitter only")
    add("iota", _gpu, dtypes=DTYPES_F32_I32, status="limited",
        unsupported_reason="iota via general-expr emitter only")

    add("sort", _gpu, dtypes=DTYPES_F32, status="limited",
        unsupported_reason="GPU sort: f32 only, via radix-sort path or bitonic sort")
    add("sort", _gpu, dtypes=DTYPES_I32, status="unsupported",
        unsupported_reason="GPU sort: i32 not implemented")
    add("grade", _gpu, dtypes=DTYPES_F32, status="limited",
        unsupported_reason="GPU grade: f32 values only (output i32 indices)")
    add("grade", _gpu, dtypes=DTYPES_I32, status="unsupported",
        unsupported_reason="GPU grade: i32 values not implemented")

    add("matmul", _gpu, dtypes=DTYPES_F32, status="limited",
        unsupported_reason="GPU matmul: f32 only, tiled and naive kernels")
    add("matmul", _gpu, dtypes=DTYPES_F32_64, status="unsupported",
        unsupported_reason="GPU matmul: f64 not implemented")

    add("filter", _gpu, dtypes=DTYPES_F32, status="limited",
        unsupported_reason="GPU filter: f32 only, parallel multi-kernel plan")
    add("replicate", _gpu, dtypes=DTYPES_F32, status="limited",
        unsupported_reason="GPU replicate: f32 only, multi-kernel plan")
    add("indices_of", _gpu, dtypes=DTYPES_F32, status="limited",
        unsupported_reason="GPU indices-of: f32 only")
    add("scatter_add", _gpu, dtypes=DTYPES_F32, status="limited",
        unsupported_reason="GPU scatter-add: f32 only, parallel and serial variants")

    add("pair", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="unsupported",
        unsupported_reason="GPU pairs not implemented")
    add("first", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="unsupported",
        unsupported_reason="GPU first/pair accessors not implemented")
    add("second", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="unsupported",
        unsupported_reason="GPU first/pair accessors not implemented")
    add("box", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="unsupported",
        unsupported_reason="GPU boxes not implemented")
    add("unbox", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="unsupported",
        unsupported_reason="GPU unbox not implemented")
    add("im2col", _gpu, dtypes=DTYPES_F32, status="unsupported",
        unsupported_reason="GPU im2col not implemented (placeholder in cascade)")
    add("col2im", _gpu, dtypes=DTYPES_F32, status="unsupported",
        unsupported_reason="GPU col2im not implemented")
    add("compose", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="unsupported",
        unsupported_reason="GPU compose not implemented")
    add("closure", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="unsupported",
        unsupported_reason="GPU closures not implemented")
    add("hof_call", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="unsupported",
        unsupported_reason="GPU dynamic higher-order calls not implemented")
    add("rerank", _gpu, dtypes=DTYPES_F32, status="limited",
        unsupported_reason="GPU rerank: via general-expr emitter")
    add("grad", _gpu, dtypes=DTYPES_F32, status="unsupported",
        unsupported_reason="GPU AD not implemented")
    add("recursion", _gpu, dtypes=DTYPES_F32_I32_BOOL, status="limited",
        unsupported_reason="GPU recursion: tail-recursive scalar helper loops in map bodies only")
    add("append", _gpu, dtypes=DTYPES_F32, status="limited",
        unsupported_reason="GPU append: f32 only, via view-ops path")
    add("define_plain", _gpu, dtypes=DTYPES_F32_I32_BOOL, status="limited",
        unsupported_reason="GPU define: limited to monomorphized inline in map kernels")
    add("define_pi", _gpu, dtypes=DTYPES_F32_I32_BOOL, status="unsupported",
        unsupported_reason="GPU define/pi: shape-dependent specialization not plumbed for GPU")
    add("define_forall", _gpu, dtypes=DTYPES_F32_I32_BOOL, status="unsupported",
        unsupported_reason="GPU define/forall: HOF monomorphization before GPU lowering")
    add("cast", _gpu, dtypes=DTYPES_F32_I32_BOOL, status="limited",
        unsupported_reason="GPU cast via general-expr emitter")
    add("array_lit", _gpu, dtypes=DTYPES_F32_I32_BOOL, status="limited",
        unsupported_reason="GPU array lit via general-expr emitter")
    add("len", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="unsupported",
        unsupported_reason="GPU length not implemented (static shapes only)")
    add("shape", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="unsupported",
        unsupported_reason="GPU shape query not implemented (static shapes only)")
    add("rank", _gpu, dtypes=DTYPES_F32_I32_BOOL_F64, status="unsupported",
        unsupported_reason="GPU rank query not implemented (static shapes only)")
    add("with_shape", _gpu, dtypes=DTYPES_F32, status="limited",
        unsupported_reason="GPU with-shape via general-expr emitter, f32 only")

    # ── Record / data-frame placeholder entries (all unsupported) ──

    for _be in (_interp, _cpu, _gpu):
        add("record", _be, dtypes=frozenset({"f32"}), status="unsupported",
            unsupported_reason="records not yet implemented (Workstream 1.5)")
        add("record_field", _be, dtypes=frozenset({"f32"}), status="unsupported",
            unsupported_reason="record field access not yet implemented")
        add("record_constructor", _be, dtypes=frozenset({"f32"}), status="unsupported",
            unsupported_reason="record constructor not yet implemented")
        add("record_update", _be, dtypes=frozenset({"f32"}), status="unsupported",
            unsupported_reason="record update not yet implemented")

    return tuple(entries)


CAPABILITIES: tuple[Capability, ...] = _build_registry()


def lookup(
    op: str,
    backend: Backend,
    *,
    dtype: str | None = None,
    rank: int | None = None,
    context: Context | None = None,
) -> Capability | None:
    best: Capability | None = None
    best_score = -1
    for cap in CAPABILITIES:
        if cap.op != op or cap.backend != backend:
            continue
        score = 0
        if dtype is not None:
            if dtype in cap.dtypes:
                score += 10
            else:
                continue
        else:
            score += 0
        if rank is not None:
            if cap.ranks is RANK_ANY or (isinstance(cap.ranks, tuple) and rank in cap.ranks):
                score += 1
            else:
                continue
        if context is not None:
            if context in cap.contexts:
                score += 5
            else:
                continue
        else:
            score += 0
        if score > best_score:
            best_score = score
            best = cap
    return best


def supported_ops(backend: Backend) -> frozenset[str]:
    return frozenset(cap.op for cap in CAPABILITIES if cap.backend == backend and cap.status in ("supported", "limited"))


def as_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cap in CAPABILITIES:
        rows.append({
            "op": cap.op,
            "backend": cap.backend.value,
            "dtypes": sorted(cap.dtypes),
            "ranks": list(cap.ranks) if isinstance(cap.ranks, tuple) else "any",
            "static_shape": cap.static_shape,
            "dynamic_shape": cap.dynamic_shape,
            "boxed_ragged": cap.boxed_ragged,
            "contexts": sorted(c.value for c in cap.contexts),
            "status": cap.status,
            "unsupported_reason": cap.unsupported_reason,
        })
    return rows
