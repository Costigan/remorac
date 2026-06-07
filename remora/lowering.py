"""HIR to MLIR lowering.

The implementation has been split into focused modules under the
``remora.lowering`` package:

  * ``types.py``      -- MLIR type mapping, shared utilities, error class
  * ``scalar.py``     -- scalar region emission (``_RegionEmitter``, etc.)
  * ``tensor_ops.py`` -- maps, folds, iota, array literals, affine map helpers
  * ``view_ops.py``   -- view operations (index, slice, transpose, reshape, ...)
  * ``indexing.py``   -- indexing lowering
  * ``module.py``     -- module building, ``MLIRLowering``, functions, descriptors

The public API is re-exported from ``remora.lowering.__init__`` so that
``from remora.lowering import MLIRLowering, type_to_mlir`` continues to work.
"""

from __future__ import annotations
