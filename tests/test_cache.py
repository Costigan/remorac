from remora import cache
from remora.types import FLOAT


def test_toolchain_fingerprint_changes_with_version_output(monkeypatch):
    class Toolchain:
        mlir_opt = "/tool/mlir-opt"
        mlir_translate = "/tool/mlir-translate"
        llc = "/tool/llc"

    class Stat:
        st_size = 123
        st_mtime_ns = 456

    class Result:
        def __init__(self, text: str) -> None:
            self.returncode = 0
            self.stdout = text
            self.stderr = ""

    versions = {"value": "version-a"}

    monkeypatch.setattr("remora.pipeline.detect_toolchain", lambda: Toolchain())
    monkeypatch.setattr(cache.Path, "stat", lambda self: Stat())
    monkeypatch.setattr(
        cache.subprocess,
        "run",
        lambda *args, **kwargs: Result(versions["value"]),
    )

    first = cache._toolchain_fingerprint()
    versions["value"] = "version-b"
    second = cache._toolchain_fingerprint()

    assert first != second


def test_cache_key_changes_with_toolchain_fingerprint(monkeypatch):
    fingerprints = {"value": "fingerprint-a"}
    monkeypatch.setattr(
        cache,
        "_toolchain_fingerprint",
        lambda: fingerprints["value"],
    )
    monkeypatch.setattr(cache, "_remora_version", lambda: "rev")

    first = cache.compute_cache_key("def f x = x", "f", (FLOAT,))
    fingerprints["value"] = "fingerprint-b"
    second = cache.compute_cache_key("def f x = x", "f", (FLOAT,))

    assert first != second
