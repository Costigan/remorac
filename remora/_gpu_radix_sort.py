"""GPU 256-bin LSD radix sort for f32 arrays.

A descriptor-ABI, device-resident radix sort built from kernels validated
incrementally against a NumPy oracle (see the build history).  Four 8-bit
passes over a monotonic uint32 key mapping of the floats:

  f32 -> key  (bitcast + sign-flip)
  per pass d:
    histogram (per-block, digit-major) -> exclusive scan -> stable scatter
      scatter local rank = match.any.sync(digit) & lanemask.lt -> popc,
      aggregated across warps; dest = offset[g*NB+b] + warp_off + warp_rank
  key -> f32

Exposed as a 12-kernel ``ExecutionPlan`` (radix for 1024 < N <= 1024*1024).
Falls back to bitonic sort outside that range (handled by the caller).
"""
from __future__ import annotations

import math

from remora.codegen import KernelMeta
from remora.execution_plan import BufferSpec, ExecutionPlan, KernelStep
from remora.gpu_lowering import GPUScaffoldError, _descriptor_load_lines
from remora.hir import HIRFunction, HIRSort
from remora.types import ArrayType, FLOAT, INT

BS = 1024
NEG = -2147483648  # 0x80000000 as i32
POS = 2147483647   # 0x7fffffff
RADIX_MAX_N = BS * 1024  # 1,048,576


def _key_map_func(name, N, inverse, *, elem="f32", is_int=False):
    inld = "\n".join(_descriptor_load_lines("in", "%in_desc", 1))
    outld = "\n".join(_descriptor_load_lines("out", "%out_desc", 1))
    if is_int:
        # i32: just XOR sign bit, no bitcast needed
        if not inverse:
            body = f"""
      %v = llvm.load %sp : !llvm.ptr -> {elem}
      %key = llvm.xor %v, %neg : {elem}
      %dp = llvm.getelementptr %out_aligned[%di] : (!llvm.ptr, i64) -> !llvm.ptr, {elem}
      llvm.store %key, %dp : {elem}, !llvm.ptr"""
        else:
            body = f"""
      %k = llvm.load %sp : !llvm.ptr -> {elem}
      %v = llvm.xor %k, %neg : {elem}
      %dp = llvm.getelementptr %out_aligned[%di] : (!llvm.ptr, i64) -> !llvm.ptr, {elem}
      llvm.store %v, %dp : {elem}, !llvm.ptr"""
        sp_elem = elem
    elif not inverse:
        body = """
      %f = llvm.load %sp : !llvm.ptr -> f32
      %u = llvm.bitcast %f : f32 to i32
      %sgn = llvm.and %u, %neg : i32
      %isneg = llvm.icmp "ne" %sgn, %zero : i32
      %notu = llvm.xor %u, %allone : i32
      %oru = llvm.or %u, %neg : i32
      %key = llvm.select %isneg, %notu, %oru : i1, i32
      %dp = llvm.getelementptr %out_aligned[%di] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %key, %dp : i32, !llvm.ptr"""
        sp_elem = "f32"
    else:
        body = """
      %k = llvm.load %sp : !llvm.ptr -> i32
      %sgn = llvm.and %k, %neg : i32
      %isneg = llvm.icmp "ne" %sgn, %zero : i32
      %ku = llvm.and %k, %pos : i32
      %notk = llvm.xor %k, %allone : i32
      %u = llvm.select %isneg, %ku, %notk : i1, i32
      %f = llvm.bitcast %u : i32 to f32
      %dp = llvm.getelementptr %out_aligned[%di] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %f, %dp : f32, !llvm.ptr"""
        sp_elem = "i32"
    return f"""    llvm.func @{name}(%in_desc: !llvm.ptr, %out_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{inld}
{outld}
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %bid = nvvm.read.ptx.sreg.ctaid.x : i32
      %bdim = nvvm.read.ptx.sreg.ntid.x : i32
      %bo = llvm.mul %bid, %bdim : i32
      %g32 = llvm.add %bo, %tid : i32
      %gidx = llvm.sext %g32 : i32 to i64
      %N = llvm.mlir.constant({N} : i64) : i64
      %neg = llvm.mlir.constant({NEG} : i32) : i32
      %pos = llvm.mlir.constant({POS} : i32) : i32
      %zero = llvm.mlir.constant(0 : i32) : i32
      %allone = llvm.mlir.constant(-1 : i32) : i32
      %ok = llvm.icmp "ult" %gidx, %N : i64
      llvm.cond_br %ok, ^body, ^done
    ^body:
      %si = llvm.add %in_offset, %gidx : i64
      %sp = llvm.getelementptr %in_aligned[%si] : (!llvm.ptr, i64) -> !llvm.ptr, {sp_elem}
      %di = llvm.add %out_offset, %gidx : i64{body}
      llvm.br ^done
    ^done:
      llvm.return
    }}"""


def _hist_func(name, N, NB, d):
    inld = "\n".join(_descriptor_load_lines("in", "%in_desc", 1))
    outld = "\n".join(_descriptor_load_lines("out", "%out_desc", 1))
    shift = 8 * d
    return f"""    llvm.func @{name}(%in_desc: !llvm.ptr, %out_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{inld}
{outld}
      %tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %tid = llvm.sext %tid32 : i32 to i64
      %bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %bid = llvm.sext %bid32 : i32 to i64
      %BS = llvm.mlir.constant({BS} : i64) : i64
      %N = llvm.mlir.constant({N} : i64) : i64
      %NB = llvm.mlir.constant({NB} : i64) : i64
      %c256 = llvm.mlir.constant(256 : i64) : i64
      %shift = llvm.mlir.constant({shift} : i32) : i32
      %ff = llvm.mlir.constant(255 : i32) : i32
      %one = llvm.mlir.constant(1 : i32) : i32
      %z = llvm.mlir.constant(0 : i32) : i32
      %sh = llvm.mlir.addressof @hist_sh : !llvm.ptr<3>
      llvm.br ^zero(%tid : i64)
    ^zero(%j: i64):
      %zd = llvm.icmp "uge" %j, %c256 : i64
      llvm.cond_br %zd, ^zdone, ^zbody
    ^zbody:
      %zp = llvm.getelementptr %sh[0, %j] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<256 x i32>
      llvm.store %z, %zp : i32, !llvm.ptr<3>
      %jn = llvm.add %j, %BS : i64
      llvm.br ^zero(%jn : i64)
    ^zdone:
      nvvm.barrier0
      %base = llvm.mul %bid, %BS : i64
      %gidx = llvm.add %base, %tid : i64
      %ok = llvm.icmp "ult" %gidx, %N : i64
      llvm.cond_br %ok, ^acc, ^sync2
    ^acc:
      %si = llvm.add %in_offset, %gidx : i64
      %kp = llvm.getelementptr %in_aligned[%si] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %k = llvm.load %kp : !llvm.ptr -> i32
      %sd = llvm.lshr %k, %shift : i32
      %dig = llvm.and %sd, %ff : i32
      %dig64 = llvm.sext %dig : i32 to i64
      %ap = llvm.getelementptr %sh[0, %dig64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<256 x i32>
      %old = llvm.atomicrmw add %ap, %one monotonic : !llvm.ptr<3>, i32
      llvm.br ^sync2
    ^sync2:
      nvvm.barrier0
      llvm.br ^wr(%tid : i64)
    ^wr(%g: i64):
      %wd = llvm.icmp "uge" %g, %c256 : i64
      llvm.cond_br %wd, ^wdone, ^wbody
    ^wbody:
      %sp = llvm.getelementptr %sh[0, %g] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<256 x i32>
      %v = llvm.load %sp : !llvm.ptr<3> -> i32
      %row = llvm.mul %g, %NB : i64
      %idx = llvm.add %row, %bid : i64
      %oi = llvm.add %out_offset, %idx : i64
      %op = llvm.getelementptr %out_aligned[%oi] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %v, %op : i32, !llvm.ptr
      %gn = llvm.add %g, %BS : i64
      llvm.br ^wr(%gn : i64)
    ^wdone:
      llvm.return
    }}"""


def _incl_scan_loop(M):
    max_d = math.ceil(math.log2(M)) if M > 1 else 0
    return f"""      %c0 = llvm.mlir.constant(0 : i64) : i64
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %c2 = llvm.mlir.constant(2 : i64) : i64
      %maxd = llvm.mlir.constant({max_d} : i64) : i64
      nvvm.barrier0
      llvm.br ^loop(%c0, %c1 : i64, i64)
    ^loop(%dd: i64, %stride: i64):
      %ld = llvm.icmp "uge" %dd, %maxd : i64
      llvm.cond_br %ld, ^scandone, ^step
    ^step:
      %active = llvm.icmp "uge" %tid, %stride : i64
      %praw = llvm.sub %tid, %stride : i64
      %psafe = llvm.select %active, %praw, %c0 : i1, i64
      %pp = llvm.getelementptr %sh[0, %psafe] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      %pv = llvm.load %pp : !llvm.ptr<3> -> i32
      %zi = llvm.mlir.constant(0 : i32) : i32
      %tmp = llvm.select %active, %pv, %zi : i1, i32
      nvvm.barrier0
      %cur = llvm.load %me : !llvm.ptr<3> -> i32
      %new = llvm.add %cur, %tmp : i32
      %res = llvm.select %active, %new, %cur : i1, i32
      llvm.store %res, %me : i32, !llvm.ptr<3>
      nvvm.barrier0
      %ndd = llvm.add %dd, %c1 : i64
      %ns = llvm.mul %stride, %c2 : i64
      llvm.br ^loop(%ndd, %ns : i64, i64)
    ^scandone:"""


def _rowscan_func(name, NB):
    ld = "\n".join(
        _descriptor_load_lines("hin", "%hist_desc", 1)
        + _descriptor_load_lines("win", "%within_desc", 1)
        + _descriptor_load_lines("tin", "%total_desc", 1)
    )
    return f"""    llvm.func @{name}(%hist_desc: !llvm.ptr, %within_desc: !llvm.ptr, %total_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{ld}
      %tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %tid = llvm.sext %tid32 : i32 to i64
      %bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %bid = llvm.sext %bid32 : i32 to i64
      %NB = llvm.mlir.constant({NB} : i64) : i64
      %Nm1 = llvm.mlir.constant({NB - 1} : i64) : i64
      %sh = llvm.mlir.addressof @sh_row : !llvm.ptr<3>
      %rowbase = llvm.mul %bid, %NB : i64
      %ridx = llvm.add %rowbase, %tid : i64
      %hoff = llvm.add %hin_offset, %ridx : i64
      %hp = llvm.getelementptr %hin_aligned[%hoff] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %v = llvm.load %hp : !llvm.ptr -> i32
      %me = llvm.getelementptr %sh[0, %tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      llvm.store %v, %me : i32, !llvm.ptr<3>
{_incl_scan_loop(NB)}
      %isz = llvm.icmp "eq" %tid, %c0 : i64
      %prev = llvm.sub %tid, %c1 : i64
      %psel = llvm.select %isz, %c0, %prev : i1, i64
      %pp2 = llvm.getelementptr %sh[0, %psel] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      %pv2 = llvm.load %pp2 : !llvm.ptr<3> -> i32
      %z2 = llvm.mlir.constant(0 : i32) : i32
      %excl = llvm.select %isz, %z2, %pv2 : i1, i32
      %woff = llvm.add %win_offset, %ridx : i64
      %wp = llvm.getelementptr %win_aligned[%woff] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %excl, %wp : i32, !llvm.ptr
      llvm.cond_br %isz, ^wtot, ^rdone
    ^wtot:
      %lp = llvm.getelementptr %sh[0, %Nm1] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      %last = llvm.load %lp : !llvm.ptr<3> -> i32
      %toff = llvm.add %tin_offset, %bid : i64
      %tp = llvm.getelementptr %tin_aligned[%toff] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %last, %tp : i32, !llvm.ptr
      llvm.br ^rdone
    ^rdone:
      llvm.return
    }}"""


def _digitscan_func(name):
    ld = "\n".join(
        _descriptor_load_lines("din", "%dt_desc", 1) + _descriptor_load_lines("bin", "%db_desc", 1)
    )
    return f"""    llvm.func @{name}(%dt_desc: !llvm.ptr, %db_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{ld}
      %tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %tid = llvm.sext %tid32 : i32 to i64
      %sh = llvm.mlir.addressof @sh_dig : !llvm.ptr<3>
      %doff = llvm.add %din_offset, %tid : i64
      %dp = llvm.getelementptr %din_aligned[%doff] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %v = llvm.load %dp : !llvm.ptr -> i32
      %me = llvm.getelementptr %sh[0, %tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      llvm.store %v, %me : i32, !llvm.ptr<3>
{_incl_scan_loop(256)}
      %isz = llvm.icmp "eq" %tid, %c0 : i64
      %prev = llvm.sub %tid, %c1 : i64
      %psel = llvm.select %isz, %c0, %prev : i1, i64
      %pp2 = llvm.getelementptr %sh[0, %psel] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      %pv2 = llvm.load %pp2 : !llvm.ptr<3> -> i32
      %z2 = llvm.mlir.constant(0 : i32) : i32
      %excl = llvm.select %isz, %z2, %pv2 : i1, i32
      %boff = llvm.add %bin_offset, %tid : i64
      %bp = llvm.getelementptr %bin_aligned[%boff] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %excl, %bp : i32, !llvm.ptr
      llvm.return
    }}"""


def _combine_func(name, NTOT, NB):
    ld = "\n".join(
        _descriptor_load_lines("win", "%within_desc", 1)
        + _descriptor_load_lines("bin", "%db_desc", 1)
        + _descriptor_load_lines("oin", "%off_desc", 1)
    )
    return f"""    llvm.func @{name}(%within_desc: !llvm.ptr, %db_desc: !llvm.ptr, %off_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{ld}
      %tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %bdim32 = nvvm.read.ptx.sreg.ntid.x : i32
      %t1 = llvm.mul %bid32, %bdim32 : i32
      %g32 = llvm.add %t1, %tid32 : i32
      %gidx = llvm.sext %g32 : i32 to i64
      %NTOT = llvm.mlir.constant({NTOT} : i64) : i64
      %NB = llvm.mlir.constant({NB} : i64) : i64
      %ok = llvm.icmp "ult" %gidx, %NTOT : i64
      llvm.cond_br %ok, ^body, ^done
    ^body:
      %g = llvm.udiv %gidx, %NB : i64
      %wo = llvm.add %win_offset, %gidx : i64
      %wp = llvm.getelementptr %win_aligned[%wo] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %w = llvm.load %wp : !llvm.ptr -> i32
      %bo = llvm.add %bin_offset, %g : i64
      %bp = llvm.getelementptr %bin_aligned[%bo] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %db = llvm.load %bp : !llvm.ptr -> i32
      %o = llvm.add %w, %db : i32
      %oo = llvm.add %oin_offset, %gidx : i64
      %op = llvm.getelementptr %oin_aligned[%oo] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %o, %op : i32, !llvm.ptr
      llvm.br ^done
    ^done:
      llvm.return
    }}"""


def _scatter_func(name, N, NB, d):
    ld = "\n".join(
        _descriptor_load_lines("kin", "%keys_desc", 1)
        + _descriptor_load_lines("oin", "%off_desc", 1)
        + _descriptor_load_lines("din", "%out_desc", 1)
    )
    shift = 8 * d
    return f"""    llvm.func @{name}(%keys_desc: !llvm.ptr, %off_desc: !llvm.ptr, %out_desc: !llvm.ptr) attributes {{gpu.kernel, nvvm.kernel}} {{
{ld}
      %tid32 = nvvm.read.ptx.sreg.tid.x : i32
      %tid = llvm.sext %tid32 : i32 to i64
      %bid32 = nvvm.read.ptx.sreg.ctaid.x : i32
      %bid = llvm.sext %bid32 : i32 to i64
      %five = llvm.mlir.constant(5 : i32) : i32
      %warpid = llvm.lshr %tid32, %five : i32
      %warp64 = llvm.sext %warpid : i32 to i64
      %ltmask = llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.lanemask.lt"() : () -> i32
      %BS = llvm.mlir.constant({BS} : i64) : i64
      %N = llvm.mlir.constant({N} : i64) : i64
      %NB = llvm.mlir.constant({NB} : i64) : i64
      %shift = llvm.mlir.constant({shift} : i32) : i32
      %ff = llvm.mlir.constant(255 : i32) : i32
      %c256i = llvm.mlir.constant(256 : i32) : i32
      %c256 = llvm.mlir.constant(256 : i64) : i64
      %c0 = llvm.mlir.constant(0 : i64) : i64
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %c32 = llvm.mlir.constant(32 : i64) : i64
      %c8192 = llvm.mlir.constant(8192 : i64) : i64
      %zi = llvm.mlir.constant(0 : i32) : i32
      %allone = llvm.mlir.constant(-1 : i32) : i32
      %wc = llvm.mlir.addressof @wc : !llvm.ptr<3>
      %sdig = llvm.mlir.addressof @sdig : !llvm.ptr<3>
      %srank = llvm.mlir.addressof @srank : !llvm.ptr<3>
      %base = llvm.mul %bid, %BS : i64
      %gidx = llvm.add %base, %tid : i64
      llvm.br ^zloop(%tid : i64)
    ^zloop(%j: i64):
      %zd = llvm.icmp "uge" %j, %c8192 : i64
      llvm.cond_br %zd, ^zdone, ^zb
    ^zb:
      %zp = llvm.getelementptr %wc[0, %j] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8192 x i32>
      llvm.store %zi, %zp : i32, !llvm.ptr<3>
      %jn = llvm.add %j, %BS : i64
      llvm.br ^zloop(%jn : i64)
    ^zdone:
      nvvm.barrier0
      %ok = llvm.icmp "ult" %gidx, %N : i64
      llvm.cond_br %ok, ^ld, ^sentinel
    ^ld:
      %ko = llvm.add %kin_offset, %gidx : i64
      %kp = llvm.getelementptr %kin_aligned[%ko] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %k = llvm.load %kp : !llvm.ptr -> i32
      %sh1 = llvm.lshr %k, %shift : i32
      %dig = llvm.and %sh1, %ff : i32
      llvm.br ^afterdig(%dig : i32)
    ^sentinel:
      llvm.br ^afterdig(%c256i : i32)
    ^afterdig(%digit: i32):
      %match = llvm.call_intrinsic "llvm.nvvm.match.any.sync.i32"(%allone, %digit) : (i32, i32) -> i32
      %below = llvm.and %match, %ltmask : i32
      %rank = llvm.call_intrinsic "llvm.ctpop.i32"(%below) : (i32) -> i32
      %cnt = llvm.call_intrinsic "llvm.ctpop.i32"(%match) : (i32) -> i32
      %sdp = llvm.getelementptr %sdig[0, %tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      llvm.store %digit, %sdp : i32, !llvm.ptr<3>
      %srp = llvm.getelementptr %srank[0, %tid] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<1024 x i32>
      llvm.store %rank, %srp : i32, !llvm.ptr<3>
      %isleader = llvm.icmp "eq" %rank, %zi : i32
      %notsent = llvm.icmp "ult" %digit, %c256i : i32
      %dowrite = llvm.and %isleader, %notsent : i1
      llvm.cond_br %dowrite, ^lead, ^afterlead
    ^lead:
      %dig64 = llvm.sext %digit : i32 to i64
      %wrow = llvm.mul %warp64, %c256 : i64
      %widx = llvm.add %wrow, %dig64 : i64
      %wp = llvm.getelementptr %wc[0, %widx] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8192 x i32>
      llvm.store %cnt, %wp : i32, !llvm.ptr<3>
      llvm.br ^afterlead
    ^afterlead:
      nvvm.barrier0
      %do_scan = llvm.icmp "ult" %tid, %c256 : i64
      llvm.cond_br %do_scan, ^scan, ^afterscan
    ^scan:
      llvm.br ^sloop(%c0, %zi : i64, i32)
    ^sloop(%w: i64, %acc: i32):
      %wd = llvm.icmp "uge" %w, %c32 : i64
      llvm.cond_br %wd, ^afterscan, ^sbody
    ^sbody:
      %srow = llvm.mul %w, %c256 : i64
      %sidx = llvm.add %srow, %tid : i64
      %scp = llvm.getelementptr %wc[0, %sidx] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8192 x i32>
      %cval = llvm.load %scp : !llvm.ptr<3> -> i32
      llvm.store %acc, %scp : i32, !llvm.ptr<3>
      %nacc = llvm.add %acc, %cval : i32
      %wn = llvm.add %w, %c1 : i64
      llvm.br ^sloop(%wn, %nacc : i64, i32)
    ^afterscan:
      nvvm.barrier0
      llvm.cond_br %ok, ^scat, ^done
    ^scat:
      %d2 = llvm.load %sdp : !llvm.ptr<3> -> i32
      %r2 = llvm.load %srp : !llvm.ptr<3> -> i32
      %d2_64 = llvm.sext %d2 : i32 to i64
      %wrow2 = llvm.mul %warp64, %c256 : i64
      %widx2 = llvm.add %wrow2, %d2_64 : i64
      %wp2 = llvm.getelementptr %wc[0, %widx2] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8192 x i32>
      %woff = llvm.load %wp2 : !llvm.ptr<3> -> i32
      %orow = llvm.mul %d2_64, %NB : i64
      %oidx = llvm.add %orow, %bid : i64
      %ooff = llvm.add %oin_offset, %oidx : i64
      %op = llvm.getelementptr %oin_aligned[%ooff] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %base2 = llvm.load %op : !llvm.ptr -> i32
      %s1 = llvm.add %base2, %woff : i32
      %dest = llvm.add %s1, %r2 : i32
      %dest64 = llvm.sext %dest : i32 to i64
      %ko2 = llvm.add %kin_offset, %gidx : i64
      %kp2 = llvm.getelementptr %kin_aligned[%ko2] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %kval = llvm.load %kp2 : !llvm.ptr -> i32
      %deo = llvm.add %din_offset, %dest64 : i64
      %dp = llvm.getelementptr %din_aligned[%deo] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %kval, %dp : i32, !llvm.ptr
      llvm.br ^done
    ^done:
      llvm.return
    }}"""


def _km(name, block, grid, ninputs, out_shape, out_dtype):
    return KernelMeta(
        name=name, grid_dims=1, block_size=block, num_inputs=ninputs, num_outputs=1,
        input_elem_types=["i32"] * ninputs, output_elem_types=[out_dtype],
        output_shape=out_shape, output_dtype="int32" if out_dtype == "i32" else "float32",
        grid_size=grid,
    )


def build_radix_sort_gpu_module(function: HIRFunction, *, kernel_name: str | None = None):
    """Build the f32/i32 radix-sort module, kernel metas, and ExecutionPlan."""
    if not isinstance(function.body, HIRSort):
        raise GPUScaffoldError("radix sort requires HIRSort body")
    if len(function.params) != 1:
        raise GPUScaffoldError("radix sort requires one parameter")
    pt = function.params[0].type
    if not isinstance(pt, ArrayType) or pt.rank != 1:
        raise GPUScaffoldError("radix sort supports rank-1 only")
    if pt.element == FLOAT:
        _elem = "f32"
        _key_elem = "i32"
        _is_int = False
    elif pt.element == INT:
        _elem = "i32"
        _key_elem = "i32"
        _is_int = True
    else:
        raise GPUScaffoldError("radix sort supports f32 and i32 only")
    N = int(pt.shape[0].value)
    if N <= BS or N > RADIX_MAX_N:
        raise GPUScaffoldError(f"radix sort handles 1024 < N <= {RADIX_MAX_N}")

    NB = (N + BS - 1) // BS
    NTOT = 256 * NB
    base = kernel_name or f"remora_{function.name}"
    nm = {
        "kf": f"{base}_keyfwd", "ki": f"{base}_keyinv",
        "rs": f"{base}_rowscan", "ds": f"{base}_digitscan", "cb": f"{base}_combine",
    }
    hnames = [f"{base}_h{d}" for d in range(4)]
    snames = [f"{base}_scat{d}" for d in range(4)]

    funcs = [
        _key_map_func(nm["kf"], N, False, elem=_elem, is_int=_is_int),
        _key_map_func(nm["ki"], N, True, elem=_elem, is_int=_is_int),
        *[_hist_func(hnames[d], N, NB, d) for d in range(4)],
        _rowscan_func(nm["rs"], NB),
        _digitscan_func(nm["ds"]),
        _combine_func(nm["cb"], NTOT, NB),
        *[_scatter_func(snames[d], N, NB, d) for d in range(4)],
    ]
    globals_mlir = """    llvm.mlir.global internal @hist_sh() {addr_space = 3 : i32} : !llvm.array<256 x i32>
    llvm.mlir.global internal @sh_row() {addr_space = 3 : i32} : !llvm.array<1024 x i32>
    llvm.mlir.global internal @sh_dig() {addr_space = 3 : i32} : !llvm.array<1024 x i32>
    llvm.mlir.global internal @wc() {addr_space = 3 : i32} : !llvm.array<8192 x i32>
    llvm.mlir.global internal @sdig() {addr_space = 3 : i32} : !llvm.array<1024 x i32>
    llvm.mlir.global internal @srank() {addr_space = 3 : i32} : !llvm.array<1024 x i32>"""
    text = "module {\n  gpu.module @remora_gpu {\n" + globals_mlir + "\n" + "\n".join(funcs) + "\n  }\n}\n"

    gN = (N + 256 - 1) // 256
    gT = (NTOT + 256 - 1) // 256
    metas = [
        _km(nm["kf"], 256, gN, 1, (N,), "i32"),
        _km(nm["ki"], 256, gN, 1, (N,), "f32"),
        *[_km(hnames[d], BS, NB, 1, (NTOT,), "i32") for d in range(4)],
        _km(nm["rs"], NB, 256, 2, (256,), "i32"),
        _km(nm["ds"], 256, 1, 1, (256,), "i32"),
        _km(nm["cb"], 256, gT, 2, (NTOT,), "i32"),
        *[_km(snames[d], BS, NB, 2, (N,), "i32") for d in range(4)],
    ]

    buffers = [
        BufferSpec("ka", (N,), "i32"), BufferSpec("kb", (N,), "i32"),
        BufferSpec("hist", (NTOT,), "i32"), BufferSpec("within", (NTOT,), "i32"),
        BufferSpec("dtot", (256,), "i32"), BufferSpec("dbase", (256,), "i32"),
        BufferSpec("off", (NTOT,), "i32"), BufferSpec("fout", (N,), "f32"),
    ]
    steps = [KernelStep(nm["kf"], ["input_0"], "ka")]
    cur, nxt = "ka", "kb"
    for d in range(4):
        steps += [
            KernelStep(hnames[d], [cur], "hist"),
            KernelStep(nm["rs"], ["hist", "within"], "dtot"),
            KernelStep(nm["ds"], ["dtot"], "dbase"),
            KernelStep(nm["cb"], ["within", "dbase"], "off"),
            KernelStep(snames[d], [cur, "off"], nxt),
        ]
        cur, nxt = nxt, cur
    steps.append(KernelStep(nm["ki"], [cur], "fout"))

    plan = ExecutionPlan(
        buffers=buffers, steps=steps, final_output="fout",
        output_shape=(N,), output_dtype="f32",
    )
    return text, metas, plan
