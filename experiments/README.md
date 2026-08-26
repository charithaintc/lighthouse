# Performance experiments

| Script | What it compares | Output |
|--------|------------------|--------|
| `run_attention_perf_comparison.py` | Three attention lowerings across context lengths. Spans two LLVM revisions, so it switches branches and rebuilds. | `attention_perf_results.md` |
| `run_reduction_fusion_comparison.py` | The two reduction schedules for a `max -> E -> sum` kernel (softmax, L1/L2 normalize) across reduction-dimension sizes. One checkout, one build, no rebuilds. | `<kernel>_online_results.md` |

---

# Dependent reduction fusion comparison

`run_reduction_fusion_comparison.py` benchmarks the two reduction schedules a
`max -> E -> sum` kernel can emit, with the parallel dimension fixed and the
reduction dimension swept:

| Variant | Loops over the reduction axis | Flag |
|---------|-------------------------------|------|
| baseline | one per reduction: `max`, `sum`, then the epilogue -- 3 passes over the input | (none) |
| online | `max` and `sum` folded into one loop by `transform.structured.fuse_dependant_reduction_ops`, epilogue as the only extra pass -- 2 passes | `--online` |

Three kernels share that chain and differ only in the elementwise term `E`, and
so in the correction the transform derives from it:

| `--kernel` | `E` | Correction | Example |
|------------|-----|------------|---------|
| `softmax` (default) | `exp(x - m)` | `exp(m_old - m_new)` | `examples/xegpu/softmax.py` |
| `normalize-l1` | `\|x / m\|` | `m_old / m_new` | `examples/xegpu/norm.py --norm l1 --normalize` |
| `normalize-l2` | `(x / m)^2` | `(m_old / m_new)^2` | `examples/xegpu/norm.py --norm l2 --normalize` |

Everything comes from the same checkout and the same LLVM build, so unlike the
attention script this one switches no branches and builds nothing. It needs only
the MLIR python bindings from an existing build tree.

```bash
cd <lighthouse>
experiments/run_reduction_fusion_comparison.py
experiments/run_reduction_fusion_comparison.py --kernel normalize-l2
experiments/run_reduction_fusion_comparison.py --reduction-sizes 4096 8192 --nruns 50
```

| Flag | Purpose |
|------|---------|
| `--kernel K` | Which kernel to benchmark (default `softmax`) |
| `--parallel-size N` | Row dimension, held fixed (default 4096) |
| `--reduction-sizes ...` | Column dimensions to sweep (default 4096 8192 16384) |
| `--baseline-tile` / `--online-tile` | Reduction tile size per variant (default 128 / 64) |
| `--nruns` / `--nwarmup` | Iteration counts (default 200 / 200) |
| `--llvm-build DIR` | Build tree supplying the MLIR python bindings |
| `--no-check` | Skip the numpy correctness check on every point |
| `--output PATH` | Where to write the report (default `experiments/<kernel>_online_results.md`) |

The two variants keep **different fixed tile sizes**, each tuned once at the
smallest size. Their optima genuinely differ -- the online form does one
cross-lane row reduction per tile where the baseline does one per row, so it wants
larger tiles -- and holding them fixed shows how one tuned configuration scales
rather than a per-size best-of.

Bandwidth is reported against the minimum traffic the kernel must move
(`2 * M * N * 4` bytes: read the input once, write the output once), so the
baseline's extra pass lowers its achieved figure instead of being credited as
useful traffic.

A bare row-wise norm -- `examples/xegpu/norm.py` without `--normalize`, whose
result is one scalar per row -- is deliberately **not** in the table. Its fused
form would be a single pass rather than two, but its rank-1 output crashes the
XeGPU pipeline (`setupLoadNdAnchorLayout` for the unfused schedule,
`propagateLayouts` for the fused one) with or without fusion; see
`debug_dir/issue-xegpu-rank1-row-reduction-output-crash*.mlir`. The kernel itself
is correct at linalg level.

## If the two variants look suspiciously similar

They are probably both running installed code. There is a pip-installed
`lighthouse` in site-packages on the dev machines, and
`python examples/xegpu/<kernel>.py` puts only the *script's* directory on
`sys.path` -- not the repository root -- so the installed copy wins unless
PYTHONPATH names the repository. Nothing fails: the schedule under test simply
is not there, `--online` is ignored, and the two variants end up differing only by
tile size. The script sets PYTHONPATH itself and runs a preflight that imports
`reduction_schedule` the same way the benchmark will and refuses to start if it
does not resolve inside the repository. Do not work around that by exporting
PYTHONPATH with a *trailing* colon: that adds the current directory, so results
depend on where you invoked the script from.

---

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

**An LLVM source tree only** -- the script configures and builds it itself. You do
not need to run CMake by hand; `--llvm-dir` points at the source, and the build
trees are created under `--build-root`.

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

`--lighthouse-dir` defaults to the repository containing the script.

Useful flags:

| Flag | Purpose |
|------|---------|
| `--n-ctx 1024 4096` | Subset of context lengths (default 1024 4096 8192 16384) |
| `--nruns` / `--nwarmup` | Iteration counts (default 50 / 20) |
| `--jobs N` | `ninja -j` (default: cores - 8) |
| `--build-root DIR` | Where the per-revision build trees live (default `experiments/llvm-builds`) |
| `--cmake-arg ARG` | Extra CMake argument for a *fresh* build tree (repeatable) |
| `--no-build` | Skip the rebuilds entirely. Only safe if every build tree is already up to date for its revision; a stale tree silently produces wrong numbers |
| `--fusion-llvm-rev REV` | Override the LLVM revision for version 2 |
| `--output PATH` | Where to write the report |
| `--no-check` | Skip the numpy correctness check at the smallest size |

## Build trees

The script keeps **one LLVM build tree per revision** under `--build-root`, named
after the revision:

```
experiments/llvm-builds/002905df0/                      # versions 1 and 3
experiments/llvm-builds/linalg_reduction_op_fusion_v3/  # version 2
```

It configures a tree the first time it is used and runs `ninja` in it on every
run, so a rerun rebuilds only what changed. This matters because the comparison
spans two LLVM revisions: with a single shared build tree, every run would
rebuild most of LLVM twice as it moved between them. Reusing per-revision trees
turns a rerun after a small LLVM change into an incremental build.

The trees are keyed on the revision *as named*, not on the resolved commit, so a
branch that gains commits reuses its tree and rebuilds only the delta. They are
made invisible to git (a `.gitignore` of `*` in the build root), so they can live
inside the lighthouse checkout without showing up in `git status`.

Budget roughly 2GB of disk per tree. A cold first build is 15-20 minutes per
revision; incremental rebuilds afterwards are usually a minute or two, and
`ccache` is enabled automatically if it is on PATH.

Start with a cheap smoke test before committing to the full sweep -- it warms both
build trees at one context length:

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
