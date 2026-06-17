# Remora Jupyter Integration Plan

## Objective
Integrate the Remora array language with the Jupyter/IPython ecosystem to enable rich data visualization (e.g., HTML tables, matplotlib plots, PIL images). This will be achieved by creating an IPython magic extension (`%%remora`), which allows users to execute Remora code natively in Jupyter Notebooks and return standard NumPy arrays directly to the Python environment.

## Context & Constraints
- **Documentation Only**: This plan serves as the formal architectural design. The current task is documentation-only to avoid conflicting with other agents (e.g., Codex) working in the repository.
- **Additive Change**: The proposed architecture is purely additive, adding new extension files without modifying the core compiler or standard REPL.

## Proposed Architecture
The integration relies on an **IPython Magic Extension**. 
- **Module**: `remora/magics.py`
- **Magic Command**: `%%remora`
- **Execution**: The magic will delegate to `remora.compiler` and `remora.executor` to compile and run the cell's contents.
- **Interoperability**: The execution result (a NumPy array) is returned to the cell output. Users can optionally bind this result to a Python variable in the notebook namespace.

## Implementation Steps

### 1. Create the IPython Extension Module (`remora/magics.py`)
- Define a `RemoraMagics` class inheriting from `IPython.core.magic.Magics`.
- Implement the standard `load_ipython_extension(ipython)` hook to register the class with Jupyter.

### 2. Implement the `%%remora` Cell Magic
- Decorate a method with `@cell_magic` and `@magic_arguments.magic_arguments()`.
- **Arguments**:
  - `--target` (default: `cpu`): Select the execution backend (`cpu`, `interp`, `gpu-nvidia`).
  - `--out` (optional): The name of a Python variable to which the resulting NumPy array will be bound.
- **Execution Flow**:
  1. Capture the Remora source from the cell body.
  2. Pass the source to the appropriate compilation entry point.
  3. Retrieve the evaluated result (Python scalar or NumPy array).
  4. If `--out <var>` is provided, inject the result into the notebook namespace via `self.shell.user_ns[<var>] = result`.
  5. Otherwise, return the result directly so Jupyter's display system automatically renders it.

### 3. Verification & Testing (`tests/test_magics.py`)
- Instantiate a test IPython shell using `IPython.testing.globalipapp.get_ipython()`.
- Load the `remora.magics` extension.
- Execute a simple `%%remora` cell (e.g., `iota 5`) and assert the returned object is a valid NumPy array.
- Test the `--out` argument to ensure variables are correctly injected into the namespace.
- Verify that Remora compilation errors surface cleanly as Python exceptions in the notebook.

### 4. Tutorial & Documentation
- Create a demonstration Jupyter Notebook (`docs/jupyter_tutorial.ipynb`) showcasing:
  - Loading the extension: `%load_ext remora.magics`
  - Executing basic Remora code.
  - Assigning a Remora array to a Python variable.
  - Passing the array to `matplotlib.pyplot.imshow` or `pandas.DataFrame` to demonstrate rich rendering capabilities.

## Migration & Rollback
As an isolated, additive feature, this extension will not affect the core `remorac` compiler or standard `remora` REPL. Rollback simply requires removing the `remora/magics.py` file and unregistering the extension.
