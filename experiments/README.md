# Attention performance comparison

`run_attention_perf_comparison.py` benchmarks three attention lowerings across a
range of context lengths and writes the result tables to
`attention_perf_results.md`. See `attention_perf_comparison.md` for a recorded
run and its interpretation.

## What it benchmarks

| # | Version | Lighthouse branch | LLVM revision | Kernel |
|---|---------|-------------------|---------------|--------|
| 1 | upstream-only (unfused) | `attention_upstream_only` | from that branch's pyproject pin | `examples/xegpu/attention_upstream.py` |
| 2 | reduction fusion proposal | `test_linalg_reduction_fusion_v3` | `linalg_reduction_op_fusion_v3` | `examples/xegpu/fused_attention.py` |
| 3 | hand optimized | `main` | from that branch's pyproject pin | `examples/xegpu/fused_attention.py` |

Versions 1 and 3 take their LLVM revision from the
`mlir-python-bindings==<date>+<sha>` pin in their own `pyproject.toml`, so they
follow the repository rather than a constant in the script. Version 2 names a
branch because the proposal is not upstream.

The script groups runs by LLVM revision, so it rebuilds LLVM once per distinct
revision (two rebuilds for the default set), and restores whichever branches were
checked out when it started.

## Prerequisites on the target machine

**Intel GPU stack.** An Intel GPU plus Level Zero (`libze_loader.so`) and `ocloc`.
Confirm the device is visible:

```bash
sycl-ls        # expect a [level_zero:gpu] entry
```

**Both repositories, with the needed branches available locally.** The script
checks out branches by name and reads others with `git show`, so they must exist
as local branches, not just remote refs:

```bash
cd <lighthouse>
for b in main attention_upstream_only test_linalg_reduction_fusion_v3; do
  git rev-parse --verify "$b" >/dev/null || git branch "$b" "origin/$b"
done

cd <llvm-project>
git rev-parse --verify linalg_reduction_op_fusion_v3   # proposal branch
git cat-file -e $(sed -n 's/.*mlir-python-bindings==[0-9]*+\([0-9a-f]*\).*/\1/p' \
    <lighthouse>/pyproject.toml)^{commit}              # pinned commit reachable
```

**A configured LLVM ninja build directory.** SPIR-V, the Level Zero runner and
the python bindings are all required -- without SPIR-V the kernel cannot be
serialized, and the pip-installed `mlir-python-bindings` package is not a
substitute:

```bash
cd <llvm-project> && mkdir -p build && cd build
cmake ../llvm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$(which clang) \
  -DCMAKE_CXX_COMPILER=$(which clang++) \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="SPIRV" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_INSTALL_GTEST=ON \
  -DMLIR_ENABLE_LEVELZERO_RUNNER=1 \
  -DMLIR_ENABLE_BINDINGS_PYTHON=1 \
  -DPython3_EXECUTABLE=$(which python3) \
  -DBUILD_SHARED_LIBS=ON
```

**Python 3.10-3.12 with lighthouse's runtime dependencies** (`numpy`, `pyyaml`).
The script imports lighthouse from the repository via `PYTHONPATH`, so lighthouse
itself does not need to be installed -- but if a `lighthouse` *is* pip-installed
it must not shadow the repository, which the script's `PYTHONPATH` prevents.
Do not rely on a pip-installed `mlir-python-bindings`; the local build is used.

**Clean worktrees.** The script refuses to move a repository with uncommitted
changes to tracked files, because switching to a branch where a modified file
does not exist would lose that edit. Commit or stash first, or pass
`--allow-dirty` if you understand the risk.

## Running it

```bash
cd <lighthouse>
experiments/run_attention_perf_comparison.py --llvm-dir <llvm-project>
```

`--lighthouse-dir` defaults to the repository containing the script;
`--build-dir` defaults to `<llvm-dir>/build`.

Useful flags:

| Flag | Purpose |
|------|---------|
| `--n-ctx 1024 4096` | Subset of context lengths (default 1024 4096 8192 16384) |
| `--nruns` / `--nwarmup` | Iteration counts (default 50 / 20) |
| `--jobs N` | `ninja -j` (default: cores - 8) |
| `--no-build` | Skip the rebuilds. Only safe if the build directory already matches the revision being benchmarked; a stale build silently produces wrong numbers |
| `--fusion-llvm-rev REV` | Override the LLVM revision for version 2 |
| `--output PATH` | Where to write the report |
| `--no-check` | Skip the numpy correctness check at the smallest size |

Budget roughly 12-15 minutes per LLVM rebuild plus a couple of minutes per
benchmark point. A first run on a cold build directory takes considerably longer.

Start with a cheap smoke test before committing to the full sweep:

```bash
experiments/run_attention_perf_comparison.py --llvm-dir <llvm-project> --n-ctx 1024
```

## If it fails

The script runs a preflight check on the python `MemoryEffectsOpInterface`
before building, because both repositories moved across the change from
`get_effects(op, effects)` (out-param) to `get_effects(op) -> list`, and a stale
pairing breaks every lighthouse transform extension. Either direction is caught,
in every `transform_ext` op on the branch -- branch-local extensions such as
`set_fastmath.py` and `fold_exp_div.py` are easy to miss, since a merge that ports
everything else cannot reach files that do not exist on the source branch. The
error names the offending files and both sides of the mismatch. Fix it by pairing
the version with the matching LLVM revision, or by updating those `get_effects`
implementations.

A `TypeError: get_effects() missing 1 required positional argument` at runtime
means the preflight was bypassed, or a file in the working tree differs from what
is committed on the branch.
