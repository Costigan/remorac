# CI cleanup — status and remaining work

Purpose: track the conversion of this repo's CI from the inherited **ILGPU / .NET**
machinery to a Python-only setup for RemoraC, so we can pause and return to it later.

## Background (why CI never passed)

This repo was derived from ILGPU (a .NET project). The entire `.github/` tree was
ILGPU's, and the only RemoraC-relevant piece was one `python-tests` job buried inside
`ci.yml`. CI was permanently red for two stacked reasons:

1. The merge gate `all-required-checks-done` required .NET jobs (`build`, `test-library`,
   `package-*`, `check-style`, `check-version`) that run `dotnet …` against `Src/`,
   `Samples/`, `Tools/` — directories that do not exist here. They failed on every run.
2. Even the Python job failed: it died at **Validate MLIR toolchain** because
   `mlir-opt` / `mlir-translate` / `llc` were "not found". `iree-compiler` (pip) ships
   IREE's own tools but **not** the standalone LLVM/MLIR binaries, and a bare
   `ubuntu-22.04` runner has none.

## Done (committed)

- `f340bcd` — Replaced `ci.yml` with a single Python-only `Python tests` job
  (checkout → setup-uv → Python 3.11 → `uv sync` → validate toolchain → `pytest`,
  `REMORA_TEST_GPU` defaulting to `0` via repo var). Deleted the dead .NET workflows:
  `codeql-analysis.yml`, `deploy-site.yml`, `update-cuda-versions.yml`,
  `update-copyright-year.yml`, `nudge.yml`, `check-required.yml`, and the `Scripts/` dir.
  Also made GPU first-class locally (`tests/conftest.py`: `REMORA_TEST_GPU` defaults to
  `1` when unset, with a fail-fast probe).
- `9456134` — Added an **Install LLVM/MLIR 18 toolchain** step (apt.llvm.org;
  `llvm-18 llvm-18-tools mlir-18-tools libomp-18-dev`; `/usr/lib/llvm-18/bin` on PATH)
  so `detect_toolchain()` resolves the tools and `runtime.py` finds
  `/usr/lib/llvm-18/lib/libomp.so`. Pinned `runs-on: ubuntu-22.04`.

Current `.github/` contents: `workflows/ci.yml` (Python-only) and `dependabot.yml`
(still partly ILGPU — see below).

## Open — verify the run is actually green

The push to `main` triggers CI (events: `push`, `pull_request`, `workflow_dispatch`,
plus a daily `schedule`). Confirm the latest `Python tests` job is green:

```bash
gh run list --limit 5
gh run view <run-id> --job=<job-id> --log    # or just: gh run view <run-id>
```

If `pytest` now surfaces *clean-runner* failures (it had never run to completion before),
fold them into the next iteration. Likely suspects: LLVM-18-specific MLIR syntax
mismatches, OpenMP/`libomp` threading, or tests that assume a GPU but should skip when
`REMORA_TEST_GPU=0`.

## Remaining tasks

### 1. dependabot.yml cleanup
`.github/dependabot.yml` is still half ILGPU. Remove the dead ecosystems, keep
`github-actions`:
- [ ] Drop `nuget` (`/Src/`), `nuget` (`/Samples/`), `bundler` (`/Site/`) entries.
- [ ] Keep the `github-actions` entry.
- [ ] (Optional) decide whether to track Python deps (uv/pip) — dependabot's uv support
      is limited; may not be worth it.
- [ ] Close the stale open Dependabot PRs that target now-deleted workflows
      (`actions/setup-dotnet`, `actions/deploy-pages`, `actions/github-script`,
      `softprops/action-gh-release`). After the deletions are on the default branch,
      dependabot should stop reopening them.
- [ ] Decide on the `astral-sh/setup-uv` v5→v7 bump PR (ci.yml currently pins `@v5`):
      accept the bump or keep the pin.

### 2. Quality gates (agreed order)
- [ ] **`ruff` F-rules (or `pyflakes`) gate** in CI — undefined/unbound names + unused
      imports only, no style noise. This class shipped the `UnboundLocalError` in
      `codegen.py`. Add a step: `uvx ruff check --select F .` (and add `ruff` appropriately).
- [ ] **"GPU lowering file changed ⇒ a parity test must change" guard** — a CI check
      that fails a PR touching `remora/gpu_lowering.py`, `remora/codegen.py`,
      `remora/_gpu_expr_lowering.py`, or `remora/_gpu_*` without also touching a
      `tests/test_gpu_*` file. Cheap defense against the silent-miscompile class.
- [ ] **Convert the ~40 compile-only `*_compiles` GPU tests to numeric parity.** That's
      the remaining shelf where silent bugs hide. Known lurkers confirmed during the
      view-op fix: `scatter_add`'s fallback `GpuArrayExpr(... .shape)` (no such attr) and
      rank-≥2 sub-array indexing emitting invalid MLIR (`%in0_stride1` undeclared). Both
      still have only compile-only coverage. See `AGENTS.md` → "Coverage rules".

### 3. GPU coverage in CI (strategic)
- [ ] Add a **self-hosted CUDA runner** so the GPU path runs in CI. Today CI is CPU-only
      (`REMORA_TEST_GPU=0`), so the worst failure mode — GPU silent miscompiles — cannot
      be caught by CI at all; it relies entirely on the local GPU dev machine.
- [ ] Once a GPU runner exists, set repo **variable** `REMORA_TEST_GPU=1`
      (Settings → Secrets and variables → Actions → Variables) so CI requires GPU
      coverage. `ci.yml` already reads `${{ vars.REMORA_TEST_GPU || '0' }}`.

### 4. GitHub web-UI settings (only the repo owner can do)
- [ ] **Required status check (optional, once CI is green):** there are currently no
      rulesets/branch-protection rules, so nothing blocks merges and nothing needs
      repointing. If you later want a broken `main` to block PRs, add a ruleset/branch
      rule requiring the **`Python tests`** check.
- [ ] **CodeQL default setup:** Settings → Code security. The deleted
      `codeql-analysis.yml` was C#-only; adjust to Python or disable.
- [ ] **Unused Actions secrets:** `NUGET_API_KEY`, `FEEDZIO_API_KEY`, `NUDGE_WEBHOOKS`
      are referenced by nothing now — safe to delete.

### 5. Repo hygiene
- [ ] Remove `last_ci_log.txt` (~1 MB, untracked) from the repo root.
- [ ] Remove the `AGENTS.md~` editor backup.
- [ ] Consider adding both patterns to `.gitignore`.

## Reference: the current CI workflow

`/.github/workflows/ci.yml` — single job `Python tests` on `ubuntu-22.04`:
checkout → install uv → Python 3.11 → **install LLVM/MLIR 18 toolchain** → `uv sync` →
`uv run python tools/validate_mlir_toolchain.py` → `uv run pytest -q`
(`REMORA_TEST_GPU: ${{ vars.REMORA_TEST_GPU || '0' }}`).

Local vs CI GPU policy is documented in `AGENTS.md` (Testing / CI sections): locally GPU
runs by default and a missing GPU is a hard failure; CI sets `REMORA_TEST_GPU=0` and
degrades GPU-unavailable to a skip.
