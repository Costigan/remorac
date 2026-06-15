"""Native artifact cache for compiled Remora functions (Phase 9).

Caches compiled shared libraries and metadata so repeated compilation of
an unchanged specialized function reuses the existing artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from remora.types import RemoraType


# ---------------------------------------------------------------------------
# Cache location
# ---------------------------------------------------------------------------

def _cache_root() -> Path:
    """Return the cache root directory (``~/.cache/remora/native/`` on Linux)."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif os.environ.get("XDG_CACHE_HOME"):
        base = Path(os.environ["XDG_CACHE_HOME"])
    else:
        base = Path.home() / ".cache"
    return base / "remora" / "native"


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _type_str(param_types: tuple[RemoraType, ...]) -> str:
    return "|".join(str(t) for t in param_types)


def compute_cache_key(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    *,
    cpu_threads: int | None = None,
    cpu_vectorize: bool = False,
) -> str:
    """Return a deterministic cache key for the compilation inputs.

    The key is a SHA-256 hex digest of a canonicalised string that
    includes every input that affects the compiled artifact.
    """
    parts: list[str] = [
        f"source:{_hash_bytes(source.encode('utf-8'))}",
        f"function:{function_name}",
        f"param_types:{_type_str(param_types)}",
        f"cpu_threads:{cpu_threads or 0}",
        f"cpu_vectorize:{'1' if cpu_vectorize else '0'}",
        # Include compiler/toolchain version so upgrading invalidates the cache.
        f"remora_version:{_remora_version()}",
        # Include toolchain info (mlir-opt, llc versions)
        f"toolchain:{_toolchain_fingerprint()}",
        f"pipeline_version:1",
    ]
    canonical = "\n".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _remora_version() -> str:
    """Return a version identifier for the Remora compiler.

    Uses the git commit hash if available; otherwise falls back to a
    hard-coded version string.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "0.1.0"


def _toolchain_fingerprint() -> str:
    """Return a fingerprint of the MLIR/LLVM toolchain."""
    from remora.pipeline import detect_toolchain
    try:
        tc = detect_toolchain()
        parts: list[str] = []
        for attr in ("mlir_opt", "mlir_translate", "llc"):
            val = getattr(tc, attr, None)
            parts.append(f"{attr}:{val}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Cache storage
# ---------------------------------------------------------------------------

@dataclass
class CacheMetadata:
    key: str
    function_name: str
    param_types: str
    return_type_str: str
    cpu_threads: int | None
    cpu_vectorize: bool
    remora_version: str
    toolchain_fingerprint: str
    so_size: int


@dataclass
class CacheHit:
    so_path: Path
    return_type: str


def _cache_entry_dir(key: str) -> Path:
    return _cache_root() / key


def _write_atomic(path: Path, content: bytes | str) -> None:
    """Write *content* to *path* atomically via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        if isinstance(content, str):
            tmp.write_text(content, encoding="utf-8")
        else:
            tmp.write_bytes(content)
        tmp.rename(path)
    finally:
        if tmp.exists() and tmp != path:
            tmp.unlink(missing_ok=True)


def store_in_cache(
    key: str,
    *,
    so_path: Path,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    return_type_str: str,
    cpu_threads: int | None,
    cpu_vectorize: bool,
) -> None:
    """Store a compiled shared library in the cache."""
    entry_dir = _cache_entry_dir(key)
    if entry_dir.exists():
        return  # Already cached

    entry_dir.mkdir(parents=True, exist_ok=True)
    dest_so = entry_dir / "module.so"
    shutil.copy2(str(so_path), str(dest_so))

    metadata = CacheMetadata(
        key=key,
        function_name=function_name,
        param_types=_type_str(param_types),
        return_type_str=return_type_str,
        cpu_threads=cpu_threads,
        cpu_vectorize=cpu_vectorize,
        remora_version=_remora_version(),
        toolchain_fingerprint=_toolchain_fingerprint(),
        so_size=dest_so.stat().st_size,
    )
    _write_atomic(entry_dir / "metadata.json", json.dumps(asdict(metadata), indent=2))


def load_from_cache(key: str) -> CacheHit | None:
    """Return a CacheHit with so_path and return_type, or *None* on miss."""
    entry_dir = _cache_entry_dir(key)
    so_path = entry_dir / "module.so"
    meta_path = entry_dir / "metadata.json"
    if so_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return CacheHit(so_path=so_path, return_type=meta.get("return_type_str", ""))
        except (json.JSONDecodeError, KeyError):
            return None
    return None


def clear_cache() -> None:
    """Remove all cached artifacts."""
    root = _cache_root()
    if root.exists():
        shutil.rmtree(str(root))


def cache_size_bytes() -> int:
    """Return the total size of the cache in bytes."""
    root = _cache_root()
    if not root.exists():
        return 0
    total = 0
    for f in root.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def cache_disabled() -> bool:
    """Return *True* if the environment variable ``REMORA_NO_CACHE`` is set."""
    return os.environ.get("REMORA_NO_CACHE", "").strip() != ""
