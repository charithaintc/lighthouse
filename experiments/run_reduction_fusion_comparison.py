#!/usr/bin/env python3
"""Reproduce the 2-way dependent-reduction-fusion comparison.

Benchmarks the two reduction schedules the `max -> E -> sum` kernels can emit,
over a range of reduction-dimension sizes with the parallel dimension fixed:

  baseline -- one loop over the reduction axis per reduction, so `max`, `sum`
              and the epilogue each make their own pass: 3 passes.
  online   -- `max` and `sum` folded into a single loop by
              `transform.structured.fuse_dependant_reduction_ops`, leaving the
              epilogue as the only extra pass: 2 passes.

Three kernels share that chain and differ in the elementwise term `E`, and so in
the online correction the transform derives from it:

  softmax       E = exp(x - m)     correction exp(m_old - m_new)
  normalize-l1  E = |x / m|        correction m_old / m_new
  normalize-l2  E = (x / m)^2      correction (m_old / m_new)^2

Everything runs from the same checkout and the same LLVM build -- the variants
differ only by the `--online` flag -- so unlike `run_attention_perf_comparison.py`
this script switches no branches and builds nothing.

Each variant keeps a fixed reduction tile size, tuned once at the smallest size
(see `--baseline-tile` / `--online-tile`); the two optima differ, and holding
them fixed across the sweep is deliberate, so the table shows how one tuned
configuration scales rather than a per-size best-of.

Usage:
    experiments/run_reduction_fusion_comparison.py
    experiments/run_reduction_fusion_comparison.py --kernel normalize-l2
    experiments/run_reduction_fusion_comparison.py --reduction-sizes 4096 8192
    experiments/run_reduction_fusion_comparison.py --output /tmp/report.md
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

DEFAULT_LIGHTHOUSE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LLVM_BUILD = Path("/home/jovyan/llvm-project/build")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

# kernel -> (example script, extra CLI args, human-readable chain description)
KERNELS = {
    "softmax": (
        "examples/xegpu/softmax.py",
        [],
        "max -> exp(x - m) -> sum, then the normalizing divide",
    ),
    "normalize-l1": (
        "examples/xegpu/norm.py",
        ["--norm", "l1", "--normalize"],
        "max -> |x / m| -> sum, then x / norm",
    ),
    "normalize-l2": (
        "examples/xegpu/norm.py",
        ["--norm", "l2", "--normalize"],
        "max -> (x / m)^2 -> sum, then x / norm",
    ),
}

DEFAULT_PARALLEL = 4096
DEFAULT_REDUCTION_SIZES = [4096, 8192, 16384]

# Reduction tile size per variant, tuned at 4096x4096. The online form wants a
# larger tile than its own optimum would suggest at small sizes because it does
# one cross-lane row reduction *per tile* where the baseline does one per row;
# the baseline peaks later because it has no such per-tile cost to amortize.
DEFAULT_BASELINE_TILE = 128
DEFAULT_ONLINE_TILE = 64

# `sizes=M,N dt=f32 wg-rows=.. .. time(us): 182.04 GFLOPS: 460.82`
RESULT_RE = re.compile(r"time\(us\):\s*([0-9.]+)\s+GFLOPS:\s*([0-9.]+)")

BYTES_PER_ELEM = 4  # f32; the example only supports f32


class Variant:
    """One schedule under test."""

    def __init__(self, key: str, label: str, tile: int, extra_args: list[str]):
        self.key = key
        self.label = label
        self.tile = tile
        self.extra_args = extra_args


def bench_env(llvm_build: Path, lighthouse_dir: Path) -> dict:
    """Environment for a benchmark run.

    PYTHONPATH must name both directories explicitly. The MLIR bindings from the
    local build have to shadow any pip-installed `mlir-python-bindings`, and the
    lighthouse repo has to shadow any pip-installed `lighthouse` -- there is one
    in site-packages on the dev machines, and it silently wins, because
    `python examples/xegpu/softmax.py` puts only the *script's* directory on
    `sys.path`, not the repository root. Getting this wrong does not fail: it
    benchmarks the installed copy, so a schedule change under test simply does
    not appear and both variants collapse onto whatever is installed. Note also
    that a *trailing* colon in PYTHONPATH adds the current directory, which makes
    resolution depend on where the script was invoked from.
    """
    mlir_pkgs = llvm_build / "tools" / "mlir" / "python_packages" / "mlir_core"
    return dict(os.environ, PYTHONPATH=f"{mlir_pkgs}:{lighthouse_dir}")


def run_one(
    lighthouse_dir: Path,
    env: dict,
    kernel: str,
    variant: Variant,
    parallel: int,
    reduction: int,
    nruns: int,
    nwarmup: int,
    check: bool,
) -> dict | None:
    """Run one (variant, size) point; returns None if the run failed."""
    example, kernel_args, _ = KERNELS[kernel]
    cmd = [
        sys.executable,
        example,
        *kernel_args,
        "--sizes",
        str(parallel),
        str(reduction),
        "--reduction-step-size",
        str(variant.tile),
        "--nruns",
        str(nruns),
        "--nwarmup",
        str(nwarmup),
        *variant.extra_args,
    ]
    if check:
        cmd.append("--check-result")

    print(
        f"  {variant.label:<9} {parallel}x{reduction:<6} tile={variant.tile:<4} ... ",
        end="",
        flush=True,
    )
    proc = subprocess.run(
        cmd, cwd=lighthouse_dir, env=env, capture_output=True, text=True
    )
    match = RESULT_RE.search(proc.stdout)
    if proc.returncode != 0 or not match:
        print("FAILED")
        for line in (proc.stdout + proc.stderr).strip().splitlines()[-12:]:
            print(f"      {line}")
        return None

    time_us, gflops = float(match.group(1)), float(match.group(2))
    validated = "Result is correct" in proc.stdout
    # The minimum traffic any softmax must move: read the input once, write the
    # output once. Reported as achieved bandwidth against that floor, so a
    # variant making extra passes shows up as a lower number rather than being
    # credited for the traffic it added.
    min_bytes = 2 * parallel * reduction * BYTES_PER_ELEM
    gbps = min_bytes / time_us / 1e3
    print(
        f"{time_us:9.2f} us  {gbps:6.1f} GB/s"
        + ("  [checked]" if validated else "")
        + ("" if validated or not check else "  [CHECK MISSING]")
    )
    return {
        "time_us": time_us,
        "gflops": gflops,
        "gbps": gbps,
        "validated": validated,
    }


def collect_env_info(lighthouse_dir: Path, llvm_build: Path) -> dict:
    """Record enough about the machine to make the numbers interpretable."""

    def git(repo: Path, *args: str) -> str:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True
            )
            return proc.stdout.strip() if proc.returncode == 0 else "?"
        except OSError:
            return "?"

    llvm_src = llvm_build.parent
    info = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "lighthouse_branch": git(lighthouse_dir, "rev-parse", "--abbrev-ref", "HEAD"),
        "lighthouse_commit": git(lighthouse_dir, "rev-parse", "--short", "HEAD"),
        "lighthouse_dirty": bool(git(lighthouse_dir, "status", "--porcelain", "-uno")),
        "llvm_branch": git(llvm_src, "rev-parse", "--abbrev-ref", "HEAD"),
        "llvm_commit": git(llvm_src, "rev-parse", "--short", "HEAD"),
        "llvm_dirty": bool(git(llvm_src, "status", "--porcelain", "-uno")),
    }
    if shutil.which("sycl-ls"):
        proc = subprocess.run(["sycl-ls"], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            info["sycl-ls"] = proc.stdout.strip().splitlines()[:8]
    return info


def preflight(lighthouse_dir: Path, env: dict) -> None:
    """Fail loudly if the benchmark would import the wrong lighthouse.

    A pip-installed `lighthouse` shadowing the repository does not raise -- it
    just benchmarks code that has none of the schedule under test, and both
    variants report whatever is installed. Check before spending an hour on it.
    """
    probe = (
        "import inspect;"
        "from lighthouse.schedule.xegpu import reduction_schedule as s;"
        "print(inspect.getsourcefile(s))"
    )
    # Run the probe from the example's own directory, not the repository root.
    # `python -c` puts the cwd on `sys.path`, so probing from the root would put
    # the repository there for free and always pass -- while the real benchmark,
    # `python examples/xegpu/softmax.py`, gets only the *script's* directory.
    # Matching that is the whole point of the check.
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=lighthouse_dir / "examples" / "xegpu",
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"preflight: could not import lighthouse\n{proc.stderr}")
    source = Path(proc.stdout.strip())
    if lighthouse_dir.resolve() not in source.resolve().parents:
        raise SystemExit(
            f"preflight: lighthouse resolves to {source}, not to the repository at "
            f"{lighthouse_dir}.\n"
            "           An installed copy is shadowing it; the numbers would not "
            "reflect this checkout."
        )
    if "fuse_dependant_reductions" not in source.read_text():
        raise SystemExit(
            f"preflight: {source} has no `fuse_dependant_reductions` support, so "
            "`--online` would silently run the unfused schedule and the two "
            "variants would differ only by tile size."
        )


def markdown_report(
    kernel: str,
    variants: list[Variant],
    parallel: int,
    reduction_sizes: list[int],
    results: dict,
    info: dict,
    nruns: int,
    nwarmup: int,
) -> str:
    """Render the time / bandwidth / speedup table."""
    base, online = variants

    def cell(variant: Variant, n: int, field: str, fmt: str) -> str:
        point = results.get(variant.key, {}).get(n)
        return format(point[field], fmt) if point else "--"

    lines = [
        f"# {kernel} reduction fusion: baseline vs online (generated)",
        "",
        f"Chain: `{KERNELS[kernel][2]}`.",
        "",
        f"Host: `{info['hostname']}`  |  f32, parallel dim = {parallel}, "
        f"wg-rows=64, sg-rows=8, subgroup-size=16  |  "
        f"mean of {nruns} runs after {nwarmup} warmup",
        "",
        f"Reduction tile size held fixed per variant: baseline {base.tile}, "
        f"online {online.tile}. Both tuned at the smallest size only.",
        "",
        "`GB/s` is against the minimum traffic the kernel must move "
        "(`2 * M * N * 4` bytes: read the input once, write the output once), so "
        "the extra pass the baseline makes shows up as lower achieved bandwidth "
        "rather than being counted as useful traffic.",
        "",
        "| reduction dim | baseline (3 passes) | online (2 passes) | speedup | "
        "baseline GB/s | online GB/s |",
        "|--------------:|--------------------:|------------------:|--------:|"
        "--------------:|------------:|",
    ]
    for n in reduction_sizes:
        b = results.get(base.key, {}).get(n)
        o = results.get(online.key, {}).get(n)
        speedup = f"{b['time_us'] / o['time_us']:.2f}x" if b and o else "--"
        lines.append(
            f"| {n} "
            f"| {cell(base, n, 'time_us', '.2f')} us "
            f"| {cell(online, n, 'time_us', '.2f')} us "
            f"| {speedup} "
            f"| {cell(base, n, 'gbps', '.1f')} "
            f"| {cell(online, n, 'gbps', '.1f')} |"
        )

    unchecked = [
        (v.label, n)
        for v in variants
        for n in reduction_sizes
        if (results.get(v.key, {}).get(n) or {}).get("validated") is False
    ]
    missing = [
        (v.label, n)
        for v in variants
        for n in reduction_sizes
        if results.get(v.key, {}).get(n) is None
    ]

    lines += ["", "## Notes", ""]
    if missing:
        lines.append(
            "- **Incomplete**: no result for "
            + ", ".join(f"{label} at {n}" for label, n in missing)
            + ". Those cells read `--`."
        )
    if unchecked:
        lines.append(
            "- **Unvalidated**: the numpy correctness check did not pass (or was "
            "skipped) for "
            + ", ".join(f"{label} at {n}" for label, n in unchecked)
            + "."
        )
    if not missing and not unchecked:
        lines.append("- Every point passed the numpy correctness check.")
    lines += [
        "",
        "## Environment",
        "",
        f"- lighthouse: `{info['lighthouse_branch']}` @ "
        f"`{info['lighthouse_commit']}`"
        + (
            "  **(uncommitted changes to tracked files)**"
            if info["lighthouse_dirty"]
            else ""
        ),
        f"- llvm-project: `{info['llvm_branch']}` @ `{info['llvm_commit']}`"
        + (
            "  **(uncommitted changes to tracked files)**" if info["llvm_dirty"] else ""
        ),
        f"- python: {info['python']}",
    ]
    if "sycl-ls" in info:
        lines += ["", "```"] + info["sycl-ls"] + ["```"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kernel",
        choices=sorted(KERNELS),
        default="softmax",
        help="Which max -> E -> sum kernel to benchmark",
    )
    parser.add_argument(
        "--lighthouse-dir",
        type=Path,
        default=DEFAULT_LIGHTHOUSE_DIR,
        help="Lighthouse checkout (default: the one containing this script)",
    )
    parser.add_argument(
        "--llvm-build",
        type=Path,
        default=DEFAULT_LLVM_BUILD,
        help="LLVM build directory providing the MLIR python bindings",
    )
    parser.add_argument(
        "--parallel-size",
        type=int,
        default=DEFAULT_PARALLEL,
        help="Parallel (row) dimension, held fixed across the sweep",
    )
    parser.add_argument(
        "--reduction-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_REDUCTION_SIZES,
        help="Reduction (column) dimensions to sweep",
    )
    parser.add_argument(
        "--baseline-tile",
        type=int,
        default=DEFAULT_BASELINE_TILE,
        help="Reduction tile size for the baseline schedule",
    )
    parser.add_argument(
        "--online-tile",
        type=int,
        default=DEFAULT_ONLINE_TILE,
        help="Reduction tile size for the online (fused) schedule",
    )
    parser.add_argument("--nruns", type=int, default=200)
    parser.add_argument("--nwarmup", type=int, default=200)
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip the numpy correctness check on every point",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the markdown report "
        "(default: experiments/<kernel>_online_results.md)",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / f"{args.kernel}_online_results.md"

    bindings = args.llvm_build / "tools/mlir/python_packages/mlir_core"
    if not bindings.is_dir():
        print(
            f"error: MLIR python bindings not found at {bindings}\n"
            "       build them with `ninja MLIRPythonModules` in the LLVM build "
            "tree, or pass --llvm-build",
            file=sys.stderr,
        )
        return 1

    variants = [
        Variant("baseline", "baseline", args.baseline_tile, []),
        Variant("online", "online", args.online_tile, ["--online"]),
    ]

    env = bench_env(args.llvm_build, args.lighthouse_dir)
    preflight(args.lighthouse_dir, env)
    print(
        f"=== {args.kernel}: {args.parallel_size} x "
        f"{{{', '.join(map(str, args.reduction_sizes))}}} ==="
    )
    print(f"    PYTHONPATH={env['PYTHONPATH']}")
    results: dict[str, dict[int, dict]] = {v.key: {} for v in variants}
    for n in args.reduction_sizes:
        for variant in variants:
            point = run_one(
                args.lighthouse_dir,
                env,
                args.kernel,
                variant,
                args.parallel_size,
                n,
                args.nruns,
                args.nwarmup,
                check=not args.no_check,
            )
            if point:
                results[variant.key][n] = point

    report = markdown_report(
        args.kernel,
        variants,
        args.parallel_size,
        args.reduction_sizes,
        results,
        collect_env_info(args.lighthouse_dir, args.llvm_build),
        args.nruns,
        args.nwarmup,
    )
    args.output.write_text(report)
    print(f"\nWrote {args.output}\n")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
