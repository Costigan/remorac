"""Result display helpers for Remora Dense Core."""

from __future__ import annotations

import numpy as np

from remora.types import ArrayType, BOOL, FLOAT, INT, RemoraType, ScalarType


def format_result(value: object, value_type: RemoraType) -> str:
    if isinstance(value_type, ScalarType):
        return _format_scalar(value, value_type)
    if isinstance(value_type, ArrayType):
        array = np.asarray(value)
        return _format_array(array, value_type)
    return repr(value)


def _format_array(array: np.ndarray, value_type: ArrayType) -> str:
    formatter = {
        "int_kind": lambda item: _format_scalar(item, INT),
        "float_kind": lambda item: _format_scalar(item, FLOAT),
        "bool": lambda item: _format_scalar(item, BOOL),
    }
    return np.array2string(
        array,
        separator=", ",
        formatter=formatter,
    )


def _format_scalar(value: object, value_type: ScalarType) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value_type == BOOL:
        return "true" if bool(value) else "false"
    if value_type == INT:
        return str(int(value))
    if value_type == FLOAT:
        text = f"{float(value):.6g}"
        if "e" not in text and "." not in text:
            return f"{text}.0"
        return text
    return repr(value)
