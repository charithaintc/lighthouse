#!/usr/bin/env python3
"""Reproduce the 3-way attention performance comparison.

Benchmarks three attention lowerings over a range of context lengths and emits
the markdown tables in `attention_perf_comparison.md`:

  1. upstream-only (unfused)  -- upstream MLIR tiling/fusion only, 3 kv passes
  2. reduction fusion         -- the dependant-reduction-fusion proposal
  3. hand optimized           -- the replace_with_fused_attention extension

Versions 1 and 3 run against the LLVM commit their lighthouse branch pins in
pyproject.toml; version 2 needs the LLVM branch carrying the proposal. The script
groups the runs by LLVM revision so only one rebuild per distinct revision is
needed, and it restores whatever branches were checked out when it started.

Usage:
    experiments/run_attention_perf_comparison.py
    experiments/run_attention_perf_comparison.py --n-ctx 1024 4096 --no-build
    experiments/run_attention_perf_comparison.py --output /tmp/report.md

All branches are assumed to exist locally in both repositories.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LLVM_DIR = Path("/home/jovyan/llvm-project")
DEFAULT_LIGHTHOUSE_DIR = Path(__file__).resolve().parent.parent
# One LLVM build tree per revision, kept between runs so that a rerun rebuilds
# only what changed instead of thrashing a single tree back and forth.
DEFAULT_BUILD_ROOT = Path(__file__).resolve().parent / "llvm-builds"
DEFAULT_N_CTX = [1024, 4096, 8192, 16384]

# Versions with `llvm_branch = PIN` take their LLVM revision from the
# `mlir-python-bindings==<date>+<sha>` pin in that lighthouse branch's
# pyproject.toml, rather than from a constant here. Both repositories move, and a
# hardcoded revision goes stale silently: a lighthouse branch paired with the
# wrong LLVM fails deep in the run with an opaque interface error.
PIN = "<from-pyproject-pin>"

# LLVM branch carrying the dependant reduction fusion proposal. This one cannot
# come from the pin: the proposal is not upstream, so it only exists on this
# branch.
FUSION_LLVM_REV = "linalg_reduction_op_fusion_v3"

PIN_RE = re.compile(r"mlir-python-bindings==[0-9]+\+([0-9a-f]+)")


@dataclass
class Version:
    """One lowering under test."""

    key: str
    label: str
    llvm_branch: str
    lighthouse_branch: str
    example: str
    args: list[str] = field(default_factory=list)

    @property
    def params(self) -> str:
        return " ".join(self.args) if self.args else "(defaults)"


# Ordered so that the baseline comes first; speedups are relative to it.
VERSIONS = [
    Version(
        key="upstream",
        label="upstream-only",
        llvm_branch=PIN,
        lighthouse_branch="attention_upstream_only",
        example="examples/xegpu/attention_upstream.py",
        # Best config from the parameter sweep; also the committed defaults.
        args=[
            "--wg-rows=256",
            "--sg-rows=8",
            "--kv-tile-size=64",
            "--nb-prefetch=2",
        ],
    ),
    Version(
        key="redfusion",
        label="reduction fusion",
        llvm_branch=FUSION_LLVM_REV,
        lighthouse_branch="test_linalg_reduction_fusion_v3",
        example="examples/xegpu/fused_attention.py",
        # Tile 32 clearly beats the default 64 for this version.
        args=["--inner-loop-tile-size=32"],
    ),
    Version(
        key="handopt",
        label="hand optimized",
        llvm_branch=PIN,
        lighthouse_branch="main",
        example="examples/xegpu/fused_attention.py",
        args=[],
    ),
]

RESULT_RE = re.compile(r"time\(us\):\s*([0-9.]+)\s+GFLOPS:\s*([0-9.]+)")


def run(
    cmd: list[str], cwd: Path, env: dict | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


def git(repo: Path, *args: str) -> str:
    proc = run(["git", *args], cwd=repo)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} in {repo} failed:\n{proc.stderr}")
    return proc.stdout.strip()


def current_branch(repo: Path) -> str:
    branch = git(repo, "branch", "--show-current")
    # Detached HEAD: fall back to the commit so the state can still be restored.
    return branch or git(repo, "rev-parse", "HEAD")


def require_clean(repo: Path) -> None:
    """Refuse to move a repo with staged or unstaged changes to tracked files."""
    dirty = git(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit(
            f"{repo} has uncommitted changes to tracked files; commit, stash or "
            f"pass --allow-dirty:\n{dirty}"
        )


def checkout(repo: Path, rev: str) -> None:
    """Check out a branch or a commit (the latter leaves a detached HEAD)."""
    if git(repo, "rev-parse", rev) == git(repo, "rev-parse", "HEAD"):
        return
    print(f"  git checkout {rev}  ({repo.name})", flush=True)
    git(repo, "checkout", rev)


# Marker present only once the python MemoryEffectsOpInterface returns its
# effects instead of appending to an out-param.
NEW_EFFECTS_API_MARKER = "get_effects must return an iterable"
OLD_EFFECTS_API_RE = re.compile(r"def get_effects\([^)]*,\s*effects\)")


def resolve_pinned_rev(lighthouse_dir: Path, lighthouse_branch: str) -> str:
    """LLVM revision that a lighthouse branch pins in its pyproject.toml."""
    pyproject = git(lighthouse_dir, "show", f"{lighthouse_branch}:pyproject.toml")
    match = PIN_RE.search(pyproject)
    if not match:
        raise SystemExit(
            f"No mlir-python-bindings pin found in {lighthouse_branch}:pyproject.toml"
        )
    return match.group(1)


def check_effects_api(
    llvm_dir: Path, lighthouse_dir: Path, versions: list[Version]
) -> None:
    """Fail early on a python MemoryEffectsOpInterface mismatch.

    The interface changed from `get_effects(op, effects)`, which appended to an
    out-param, to `get_effects(op)`, which returns its effects. Both repositories
    moved across that change independently, so a stale pairing is easy to end up
    with. Either direction breaks every lighthouse transform extension, and
    otherwise only surfaces after a full LLVM rebuild as an opaque
    `TypeError: get_effects() missing 1 required positional argument`.

    Only the extensions living on a given lighthouse branch matter, so every
    transform_ext op on that branch is checked, not just a fixed probe file: a
    branch-local extension (set_fastmath, fold_exp_div) can lag behind a merge
    that ported everything else.
    """
    for version in versions:
        binding = git(
            llvm_dir,
            "show",
            f"{version.llvm_branch}:mlir/lib/Bindings/Python/IRInterfaces.cpp",
        )
        llvm_is_new = NEW_EFFECTS_API_MARKER in binding

        stale: list[str] = []
        listing = git(
            lighthouse_dir,
            "ls-tree",
            "-r",
            "--name-only",
            version.lighthouse_branch,
            "--",
            "lighthouse/dialects/transform",
        )
        for path in listing.splitlines():
            if not path.endswith(".py"):
                continue
            src = git(lighthouse_dir, "show", f"{version.lighthouse_branch}:{path}")
            if "def get_effects" not in src:
                continue
            is_old = bool(OLD_EFFECTS_API_RE.search(src))
            if is_old != (not llvm_is_new):
                stale.append(path)

        if stale:
            wanted = (
                "get_effects(op) -> list" if llvm_is_new else "get_effects(op, effects)"
            )
            listing = "\n".join(f"    {p}" for p in stale[:8])
            more = f"\n    ... and {len(stale) - 8} more" if len(stale) > 8 else ""
            raise SystemExit(
                f"Incompatible pair for '{version.label}':\n"
                f"  LLVM {version.llvm_branch} expects {wanted}\n"
                f"  lighthouse {version.lighthouse_branch} disagrees in:\n"
                f"{listing}{more}\n"
                f"These extensions would fail to verify. Either pair this version "
                f"with the matching LLVM revision, or update those "
                f"get_effects implementations."
            )


def build_dir_for(build_root: Path, rev: str) -> Path:
    """Per-revision build directory.

    Keyed on the revision *as named* (branch name or short sha), not on the
    resolved commit: a branch that gains commits then reuses its directory and
    rebuilds only the delta, which is the whole point of keeping these around. A
    single shared build directory would instead rebuild most of LLVM every time
    the run moves between revisions.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", rev)
    return build_root / slug


# Mirrors the configuration the recorded numbers were produced with. SPIR-V, the
# Level Zero runner and the python bindings are all load-bearing: without SPIR-V
# the kernel cannot be serialized at all.
CMAKE_ARGS = [
    "-G",
    "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLLVM_ENABLE_PROJECTS=mlir",
    "-DLLVM_TARGETS_TO_BUILD=host",
    "-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SPIRV",
    "-DLLVM_ENABLE_ASSERTIONS=ON",
    "-DLLVM_BUILD_EXAMPLES=OFF",
    "-DLLVM_INSTALL_UTILS=ON",
    "-DMLIR_ENABLE_LEVELZERO_RUNNER=1",
    "-DMLIR_ENABLE_BINDINGS_PYTHON=1",
    # Shared libs keep each build directory to ~2GB instead of tens of GB, which
    # matters once there is one per revision.
    "-DBUILD_SHARED_LIBS=ON",
]


def configure_llvm(build_dir: Path, llvm_dir: Path, extra_args: list[str]) -> None:
    """CMake-configure a build directory, unless it is already configured."""
    if (build_dir / "build.ninja").exists():
        return
    build_dir.mkdir(parents=True, exist_ok=True)
    # A .gitignore of '*' ignores the directory's contents and itself, so build
    # trees living inside the lighthouse checkout stay invisible to git on every
    # branch. Tracking one instead would only cover the branch it was added on.
    (build_dir.parent / ".gitignore").write_text("*\n")

    args = list(CMAKE_ARGS)
    # Build the bindings against the interpreter that will import them.
    args.append(f"-DPython3_EXECUTABLE={sys.executable}")
    for tool, flag in (
        ("clang", "CMAKE_C_COMPILER"),
        ("clang++", "CMAKE_CXX_COMPILER"),
    ):
        if path := shutil.which(tool):
            args.append(f"-D{flag}={path}")
    if shutil.which("ccache"):
        args.append("-DLLVM_CCACHE_BUILD=ON")
    args += extra_args

    print(f"  cmake configure ({build_dir})", flush=True)
    proc = subprocess.run(
        ["cmake", str(llvm_dir / "llvm"), *args], cwd=build_dir, text=True
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"CMake configure failed in {build_dir}. Remove that directory to "
            f"retry from scratch."
        )


def build_llvm(build_dir: Path, jobs: int) -> None:
    """Incrementally build LLVM. 'no work to do' when already up to date."""
    print(f"  ninja -j {jobs} ({build_dir})", flush=True)
    proc = subprocess.run(["ninja", "-j", str(jobs)], cwd=build_dir, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"LLVM build failed in {build_dir}")


def bench_env(llvm_build_dir: Path, lighthouse_dir: Path) -> dict:
    """Environment for a benchmark run.

    PYTHONPATH must be set explicitly: the MLIR bindings from the local build
    have to shadow any pip-installed `mlir-python-bindings` (only the local build
    has the SPIR-V target needed to serialize the kernel), and the lighthouse
    repo has to shadow any pip-installed `lighthouse`. Note that a *trailing*
    colon in PYTHONPATH silently adds the current directory, which makes the
    resolution order depend on where the script is invoked from.
    """
    env = dict(os.environ)
    mlir_pkgs = llvm_build_dir / "tools" / "mlir" / "python_packages" / "mlir_core"
    if not mlir_pkgs.exists():
        raise SystemExit(f"MLIR python bindings not found at {mlir_pkgs}")
    env["PYTHONPATH"] = f"{mlir_pkgs}:{lighthouse_dir}"
    return env


def benchmark(
    version: Version,
    n_ctx: int,
    *,
    lighthouse_dir: Path,
    env: dict,
    nwarmup: int,
    nruns: int,
    check_result: bool,
) -> dict | None:
    """Run one (version, n_ctx) point. Returns None if the run failed."""
    cmd = [
        sys.executable,
        version.example,
        f"--n-ctx={n_ctx}",
        f"--nwarmup={nwarmup}",
        f"--nruns={nruns}",
        *version.args,
    ]
    if check_result:
        cmd.append("--check-result")

    label = f"{version.label} n_ctx={n_ctx}"
    print(f"  {label} ...", end=" ", flush=True)
    proc = run(cmd, cwd=lighthouse_dir, env=env)
    match = RESULT_RE.search(proc.stdout)
    if proc.returncode != 0 or not match:
        print("FAILED")
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        for line in tail:
            print(f"      {line}")
        return None

    time_us, gflops = float(match.group(1)), float(match.group(2))
    validated = "Result is correct" in proc.stdout
    print(
        f"{time_us:.1f} us  {gflops / 1000:.2f} TFLOPS"
        + ("  [checked]" if validated else "")
    )
    return {"time_us": time_us, "gflops": gflops, "validated": validated}


def collect_env_info(llvm_dir: Path, lighthouse_dir: Path) -> dict:
    """Record enough about the machine to make the numbers interpretable."""
    info = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
    }
    for name, cmd in (("sycl-ls", ["sycl-ls"]), ("xpu-smi", ["xpu-smi", "discovery"])):
        if shutil.which(cmd[0]):
            proc = run(cmd, cwd=lighthouse_dir)
            if proc.returncode == 0 and proc.stdout.strip():
                info[name] = proc.stdout.strip().splitlines()[:8]
                break
    return info


def markdown_tables(
    versions: list[Version], n_ctx_list: list[int], results: dict
) -> str:
    """Render the time / speedup / TFLOPS tables."""
    baseline = versions[0]
    header = (
        "| n_ctx | "
        + " | ".join(f"{i + 1}. {v.label}" for i, v in enumerate(versions))
        + " |"
    )
    align = "|------:|" + "|".join(["-" * 17 + ":"] * len(versions)) + "|"

    def table(title: str, cell) -> str:
        rows = [f"## {title}", "", header, align]
        for n in n_ctx_list:
            cells = []
            for v in versions:
                r = results.get(v.key, {}).get(n)
                cells.append(
                    cell(r, results.get(baseline.key, {}).get(n)) if r else "n/a"
                )
            rows.append(f"| {n} | " + " | ".join(cells) + " |")
        return "\n".join(rows) + "\n"

    parts = [
        table("Table 1 - Time (microseconds)", lambda r, b: f"{r['time_us']:.1f}"),
        table(
            "Table 2 - Speedup (upstream-only = 1.00x)",
            lambda r, b: f"{b['time_us'] / r['time_us']:.2f}x" if b else "n/a",
        ),
        table(
            "Table 3 - Throughput (TFLOPS)", lambda r, b: f"{r['gflops'] / 1000:.2f}"
        ),
    ]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--llvm-dir", type=Path, default=DEFAULT_LLVM_DIR, help="LLVM source tree"
    )
    parser.add_argument("--lighthouse-dir", type=Path, default=DEFAULT_LIGHTHOUSE_DIR)
    parser.add_argument(
        "--build-root",
        type=Path,
        default=DEFAULT_BUILD_ROOT,
        help="Parent of the per-revision LLVM build directories, one per LLVM "
        "revision so that reruns rebuild only what changed "
        f"(default: {DEFAULT_BUILD_ROOT}, ~2GB each)",
    )
    parser.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument for the CMake configure of a fresh build directory "
        "(repeatable). Ignored for directories already configured.",
    )
    parser.add_argument(
        "--fusion-llvm-rev",
        default=FUSION_LLVM_REV,
        help="LLVM revision for the reduction fusion version "
        f"(default: {FUSION_LLVM_REV}, see the comment on FUSION_LLVM_REV)",
    )
    parser.add_argument("--n-ctx", type=int, nargs="+", default=DEFAULT_N_CTX)
    parser.add_argument("--nwarmup", type=int, default=20)
    parser.add_argument("--nruns", type=int, default=50)
    parser.add_argument(
        "--jobs", type=int, default=max(1, (os.cpu_count() or 8) - 8), help="ninja -j"
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip the LLVM rebuilds. Only safe if the build directory already "
        "matches the branch being benchmarked; a stale build silently produces "
        "wrong numbers.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Switch branches even with uncommitted changes to tracked files.",
    )
    parser.add_argument(
        "--check-smallest",
        action="store_true",
        default=True,
        help="Validate against the numpy reference at the smallest n_ctx "
        "(default: on). The reference is O(n_ctx^2) on the host, so checking the "
        "large sizes is impractically slow.",
    )
    parser.add_argument("--no-check", dest="check_smallest", action="store_false")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "attention_perf_results.md",
    )
    args = parser.parse_args()

    llvm_dir = args.llvm_dir.resolve()
    lighthouse_dir = args.lighthouse_dir.resolve()
    build_root = args.build_root.resolve()

    for repo in (llvm_dir, lighthouse_dir):
        if not (repo / ".git").exists():
            raise SystemExit(f"{repo} is not a git repository")
        if not args.allow_dirty:
            require_clean(repo)

    for version in VERSIONS:
        if version.llvm_branch == FUSION_LLVM_REV:
            version.llvm_branch = args.fusion_llvm_rev
        elif version.llvm_branch == PIN:
            version.llvm_branch = resolve_pinned_rev(
                lighthouse_dir, version.lighthouse_branch
            )
            print(
                f"{version.label}: lighthouse {version.lighthouse_branch} pins "
                f"LLVM {version.llvm_branch}"
            )
    check_effects_api(llvm_dir, lighthouse_dir, VERSIONS)

    original = {
        "llvm": current_branch(llvm_dir),
        "lighthouse": current_branch(lighthouse_dir),
    }
    print(
        f"Restoring on exit: llvm={original['llvm']} lighthouse={original['lighthouse']}"
    )

    n_ctx_list = sorted(args.n_ctx)
    smallest = n_ctx_list[0]
    results: dict[str, dict[int, dict]] = {v.key: {} for v in VERSIONS}
    commits: dict[str, dict[str, str]] = {}

    # Group by LLVM branch so each branch is built at most once.
    llvm_branches = []
    for v in VERSIONS:
        if v.llvm_branch not in llvm_branches:
            llvm_branches.append(v.llvm_branch)

    try:
        for llvm_branch in llvm_branches:
            print(f"\n=== LLVM {llvm_branch} ===", flush=True)
            checkout(llvm_dir, llvm_branch)
            commits.setdefault("llvm", {})[llvm_branch] = git(
                llvm_dir, "rev-parse", "--short", "HEAD"
            )
            build_dir = build_dir_for(build_root, llvm_branch)
            if not args.no_build:
                configure_llvm(build_dir, llvm_dir, args.cmake_arg)
                build_llvm(build_dir, args.jobs)
            elif not (build_dir / "build.ninja").exists():
                raise SystemExit(
                    f"--no-build was passed but {build_dir} is not configured; "
                    f"drop --no-build for the first run of this revision."
                )
            env = bench_env(build_dir, lighthouse_dir)

            for version in (v for v in VERSIONS if v.llvm_branch == llvm_branch):
                print(f"\n-- {version.label} [{version.params}]", flush=True)
                checkout(lighthouse_dir, version.lighthouse_branch)
                commits.setdefault("lighthouse", {})[version.lighthouse_branch] = git(
                    lighthouse_dir, "rev-parse", "--short", "HEAD"
                )
                for n_ctx in n_ctx_list:
                    point = benchmark(
                        version,
                        n_ctx,
                        lighthouse_dir=lighthouse_dir,
                        env=env,
                        nwarmup=args.nwarmup,
                        nruns=args.nruns,
                        check_result=args.check_smallest and n_ctx == smallest,
                    )
                    if point:
                        results[version.key][n_ctx] = point
    finally:
        print("\n=== Restoring original branches ===")
        for repo, branch in (
            (llvm_dir, original["llvm"]),
            (lighthouse_dir, original["lighthouse"]),
        ):
            try:
                checkout(repo, branch)
            except RuntimeError as exc:
                print(f"  WARNING: could not restore {repo}: {exc}")

    info = collect_env_info(llvm_dir, lighthouse_dir)
    info["commits"] = commits

    tables = markdown_tables(VERSIONS, n_ctx_list, results)
    report = [
        "# Attention 3-way performance comparison (generated)",
        "",
        f"Host: `{info['hostname']}`  |  Z=2, H=8, n_head=64, f16  |  "
        f"mean of {args.nruns} runs after {args.nwarmup} warmup",
        "",
        "GFLOPS convention: `Z * H * 4 * n_ctx^2 * n_head`, i.e. the two matmuls "
        "at 2 flops per multiply-accumulate, softmax excluded. This counts "
        "*useful* work, so the upstream-only version is penalised for the Q@K^T "
        "evaluations it repeats.",
        "",
        "| # | Version | LLVM branch | Lighthouse branch | Parameters |",
        "|---|---------|-------------|-------------------|------------|",
    ]
    for i, v in enumerate(VERSIONS):
        report.append(
            f"| {i + 1} | {v.label} | `{v.llvm_branch}` | "
            f"`{v.lighthouse_branch}` | `{v.params}` |"
        )
    report += [
        "",
        tables,
        "## Environment",
        "",
        "```json",
        json.dumps(info, indent=2),
        "```",
        "",
    ]

    validated = [
        f"{v.label} n_ctx={n}"
        for v in VERSIONS
        for n, r in results[v.key].items()
        if r.get("validated")
    ]
    report += [
        "## Validation coverage",
        "",
        ("Checked against the numpy reference: " + ", ".join(validated))
        if validated
        else "No run was validated against the numpy reference.",
        "",
        "All other points are timing-only.",
        "",
    ]

    args.output.write_text("\n".join(report))
    print(f"\nWrote {args.output}\n")
    print(tables)

    missing = [
        f"{v.label} n_ctx={n}"
        for v in VERSIONS
        for n in n_ctx_list
        if n not in results[v.key]
    ]
    if missing:
        print("FAILED points: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
