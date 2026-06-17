import tempfile
from pathlib import Path

from remora import runtime


def test_shared_library_linker_includes_remora_runtime(monkeypatch, tmp_path):
    calls = []
    fake_rt = tmp_path / "remora_rt.o"
    fake_rt.write_bytes(b"")

    class Toolchain:
        llc = "llc"

    def fake_get_rt():
        return str(fake_rt)

    def fake_run_checked(args, message, temp_dir):
        calls.append((args, message))
        if "-o" in args:
            out_path = Path(args[args.index("-o") + 1])
            out_path.write_bytes(b"")

    monkeypatch.setattr(runtime, "_get_remora_rt_o", fake_get_rt)
    monkeypatch.setattr(runtime, "_run_checked", fake_run_checked)
    monkeypatch.setattr(runtime, "which", lambda name: "cc" if name in {"gcc", "cc"} else None)

    temp_dir, so_path = runtime._compile_llvm_ir_to_shared_library(
        "define void @f() { ret void }",
        Toolchain(),
    )
    try:
        assert so_path.exists()
        link_args = calls[-1][0]
        assert str(fake_rt) in link_args
    finally:
        temp_dir.cleanup()
