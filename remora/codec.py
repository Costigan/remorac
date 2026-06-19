"""Custom Python source codec for Remora.

Enables ``# coding: remora`` in Python source files.  The codec finds
``# remora:begin`` / ``# remora:end`` blocks, compiles the Remora
S-expression source inside them, and replaces them with Python code
that registers the compiled functions as callable wrappers.

Usage in a .py file::

    # coding: remora
    import numpy as np

    # remora:begin
    (define/pi () (scale [xs (Array Float 4)] (Array Float 4))
      (map (* 2.0) xs))
    # remora:end

    print(scale(np.array([1.0, 2.0, 3.0, 4.0])))
"""

from __future__ import annotations

import codecs
import re


_BEGIN_MARKER = re.compile(r"^\s*#\s*remora\s*:\s*begin\s*$", re.MULTILINE)
_END_MARKER = re.compile(r"^\s*#\s*remora\s*:\s*end\s*$", re.MULTILINE)
_CODING_LINE = re.compile(r"^(\s*#.*?coding\s*[:=]\s*)remora", re.MULTILINE)


def _transform_source(source: str) -> str:
    source = _CODING_LINE.sub(r"\g<1>utf-8", source, count=1)

    parts: list[str] = []
    pos = 0

    for begin_match in _BEGIN_MARKER.finditer(source):
        end_match = _END_MARKER.search(source, begin_match.end())
        if end_match is None:
            raise SyntaxError("# remora:begin without matching # remora:end")

        parts.append(source[pos:begin_match.start()])

        remora_source = source[begin_match.end():end_match.start()].strip()

        escaped = remora_source.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")

        replacement = (
            f"import remora.api as __remora_api__\n"
            f"for __n__, __f__ in __remora_api__.compile_all("
            f"'''{escaped}''', syntax='lisp', include_prelude=False).items():\n"
            f"    globals()[__n__] = __f__\n"
            f"del __n__, __f__\n"
        )

        parts.append(replacement)
        pos = end_match.end()

    parts.append(source[pos:])
    return "".join(parts)


def _remora_decode(data: bytes, errors: str = "strict") -> tuple[str, int]:
    utf8_str, length = codecs.utf_8_decode(data, errors)
    return _transform_source(utf8_str), length


class _RemoraIncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input: bytes, final: bool = False) -> str:
        return _remora_decode(input)[0]


class _RemoraStreamReader(codecs.StreamReader):
    def decode(self, input: bytes, errors: str = "strict") -> tuple[str, int]:
        return _remora_decode(input, errors)


def _search_function(name: str) -> codecs.CodecInfo | None:
    if name != "remora":
        return None
    return codecs.CodecInfo(
        name="remora",
        encode=codecs.utf_8_encode,
        decode=_remora_decode,
        incrementaldecoder=_RemoraIncrementalDecoder,
        streamreader=_RemoraStreamReader,
    )


def register() -> None:
    """Register the ``remora`` source codec with Python's codec system."""
    codecs.register(_search_function)


register()
