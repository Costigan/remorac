"""MLIR lowering package for Remora.

Re-exports the public API and internal symbols for backward compatibility
with code that accesses ``remora.lowering`` directly.
"""

from __future__ import annotations

from remora.lowering.types import (
    RemoraLoweringError,
    TensorEnv,
    _TensorValue,
    _expr_result_type,
    _is_scalar_type,
    _join_prefix,
    type_to_mlir,
)

from remora.lowering.scalar import (
    _Operand,
    _RegionEmitter,
    _arith_op,
    _cast_if_needed,
    _comparison_op,
    _hir_prim_op,
    _literal_value,
    _load_iree_ir,
    _lower_callable_operand,
    _lower_let,
    _lower_scalar_module,
    _lower_scalar_value_for_fold_init,
)

from remora.lowering.tensor_ops import (
    _cell_element_affine_map,
    _collect_cell_indices,
    _drop_first_affine_map,
    _flatten_array_literal,
    _fold_iterators,
    _identity_affine_map,
    _lower_array_fold_module,
    _lower_array_fold_result,
    _lower_array_literal_module,
    _lower_binary_map_fold_input,
    _lower_binary_map_module,
    _lower_binary_map_result,
    _lower_binary_primitive_callable_result,
    _lower_fold_callable_body,
    _lower_fold_input,
    _lower_fold_module,
    _lower_fold_result,
    _lower_iota_module,
    _lower_iota_scalar_map_module,
    _lower_iota_scalar_map_result,
    _lower_map_binary_callable_body,
    _lower_map_binary_callable_result,
    _lower_map_callable_body,
    _lower_map_callable_result,
    _lower_map_cell_fold_result,
    _lower_map_cell_index_result,
    _lower_map_cell_module,
    _lower_map_cell_result,
    _lower_primitive_callable_body,
    _lower_primitive_callable_result,
    _lower_scalar_fold_module,
    _lower_scalar_fold_result,
    _lower_scalar_map_binary_module,
    _lower_scalar_map_module,
    _lower_tensor_input,
    _lower_transpose_input,
    _map_cell_iterators,
    _parallel_iterators,
    _reverse_first_axis_affine_map,
    _rewrite_cell_indices,
    _take_first_affine_map,
)

from remora.lowering.view_ops import (
    _lower_transpose_module,
    _lower_transpose_result,
    _lower_view_input,
    _lower_view_module,
    _lower_view_result,
)

from remora.lowering.indexing import (
    _lower_index_module,
    _lower_index_result,
    _lower_scalar_index_expr,
)

from remora.lowering.module import (
    LoweredModule,
    MLIRLowering,
    _MLIRMainModuleBuilder,
    _add_output_descriptor_export,
    _can_lower_as_scalar_expr,
    _inline_callable,
    _inline_lets,
    _lower_descriptor_export_wrapper,
    _lower_descriptor_internal_function,
    _lower_descriptor_scalar_result_body,
    _lower_function,
    _lower_function_descriptor_module,
    _lower_function_with_tensor,
    _lower_functions,
    _lower_main_module,
    _lower_main_result_with_tensor_env,
    _lower_tensor_if_module,
    _lower_tensor_if_result,
    _lower_tensor_let_module,
    _output_descriptor_export_function,
    _output_descriptor_store_lines,
    _output_memref_type,
    _prepare_main_expr,
)
