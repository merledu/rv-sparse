#!/usr/bin/env python3
"""analyze.py — summarize raw rv-sparse SpGEMM benchmark results.

Reads the raw per-run CSV and produces per-(matrix, build, cflags, kernel,
dtype) summaries, including timing statistics, GOP/s, speedup against a
consistent baseline, and a bootstrap 95% CI.

The default speedup denominator is the `arm=baseline` build of the same
dtype and matrix. This is intentionally cross-build: the autovec arm uses
the same scalar source as the baseline, but is compiled with the V extension
available. Comparing it against the gc baseline isolates compiler
autovectorization instead of comparing the build against itself.

cflags are part of the group identity because the tunables are compile-time
macros. The denominator is always the default-cflags baseline.

Usage:
    python3 bench/analyze.py bench/results/spgemm_raw.csv
    python3 bench/analyze.py bench/results/spgemm_raw.csv --csv-out summary.csv
    python3 bench/analyze.py bench/results/spgemm_raw.csv --no-csv
    python3 bench/analyze.py bench/results/spgemm_raw.csv --baseline rvv_f32

Pure standard library; runs on the board without pandas/numpy.
"""

import csv
import os
import random
import statistics
import sys
from collections import defaultdict


BASELINE_ARM = "baseline"
BASELINE_BUILD = "gc"

NO_CFLAGS = "-"

BOOT_ITERS = 2000
BOOT_SEED = 20240517
CI_LO_PCT = 2.5
CI_HI_PCT = 97.5

CLEAN_SPREAD = 0.02
NOISY_SPREAD = 0.10


def _fnum(row, key, cast=float, default=None):
    """Parse an optional numeric column."""
    value = row.get(key)
    if value is None or value == "":
        return default

    try:
        return cast(value)
    except (ValueError, TypeError):
        return default


def _infer_arm(kernel):
    """Best-effort arm for legacy CSVs without an arm column."""
    if kernel.startswith("rvv_"):
        return "intrinsic"
    # scalar_* is ambiguous without build information: the same source can
    # be either the gc baseline or the gcv autovec arm.
    return "-"


def load(path):
    rows = []

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("kernel"):
                continue

            metric = row.get("gops", row.get("gflops"))

            try:
                row["time_s"] = float(row["time_s"])
                row["gops"] = float(metric)
                row["correct"] = int(row["correct"])
                row["nnz_c"] = int(row["nnz_c"])
                row["flops"] = float(row["flops"])
            except (ValueError, KeyError, TypeError):
                continue

            row["build"] = (row.get("build") or "-").strip()
            row["arm"] = (
                (row.get("arm") or "").strip()
                or _infer_arm(row["kernel"])
            )
            row["march"] = (row.get("march") or "-").strip()
            row["cflags"] = (
                (row.get("cflags") or NO_CFLAGS).strip()
                or NO_CFLAGS
            )
            row["cc_version"] = (row.get("cc_version") or "-").strip()

            row["cycles"] = _fnum(row, "cycles")
            row["instructions"] = _fnum(row, "instructions")

            row["op_mean"] = _fnum(row, "op_mean")
            row["op_max"] = _fnum(row, "op_max")
            row["op_var"] = _fnum(row, "op_var")

            rows.append(row)

    return rows


def pick_baseline_build(rows):
    """Choose the build used by the default scalar denominator."""
    builds = {row["build"] for row in rows}

    if BASELINE_BUILD in builds:
        return BASELINE_BUILD, None

    if len(builds) == 1:
        only = next(iter(builds))
        note = (
            f"no '{BASELINE_BUILD}' build in this CSV; using the only build "
            f"present ('{only}') as the scalar baseline. Cross-build "
            f"speedups are NOT available from this file."
        )
        return only, note

    note = (
        f"no '{BASELINE_BUILD}' build in this CSV "
        f"(builds present: {sorted(builds)}). Speedups will be blank."
    )
    return BASELINE_BUILD, note


def _iqr(values):
    """Return the interquartile range."""
    if len(values) < 4:
        return 0.0

    values = sorted(values)
    n = len(values)

    q1 = statistics.median(values[: n // 2])
    q3 = statistics.median(values[(n + 1) // 2:])

    return q3 - q1


def _pct(sorted_values, percent):
    """Nearest-rank percentile on an already-sorted list."""
    if not sorted_values:
        return float("nan")

    n = len(sorted_values)
    index = int(round(percent / 100.0 * (n - 1)))
    index = min(n - 1, max(0, index))

    return sorted_values[index]


def _bootstrap_speedup_ci(base_times, kernel_times):
    """Bootstrap a 95% CI for median(base) / median(kernel)."""
    if len(base_times) < 4 or len(kernel_times) < 4:
        return float("nan"), float("nan")

    rng = random.Random(BOOT_SEED)
    nb = len(base_times)
    nk = len(kernel_times)

    ratios = []

    for _ in range(BOOT_ITERS):
        base_sample = [
            base_times[rng.randrange(nb)]
            for _ in range(nb)
        ]
        kernel_sample = [
            kernel_times[rng.randrange(nk)]
            for _ in range(nk)
        ]

        kernel_median = statistics.median(kernel_sample)

        if kernel_median > 0:
            ratios.append(
                statistics.median(base_sample) / kernel_median
            )

    if not ratios:
        return float("nan"), float("nan")

    ratios.sort()

    return (
        _pct(ratios, CI_LO_PCT),
        _pct(ratios, CI_HI_PCT),
    )


def summarize(rows):
    # cflags must remain part of the key: different compile-time tunables are
    # different binaries and must never be pooled together.
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["label"],
            row["build"],
            row["cflags"],
            row["kernel"],
            row["dtype"],
        )
        groups[key].append(row)

    summary = {}

    for key, rows_in_group in groups.items():
        times = [row["time_s"] for row in rows_in_group]
        gops = [row["gops"] for row in rows_in_group]
        correct = [row["correct"] for row in rows_in_group]

        if 0 in correct:
            status = "FAIL"
        elif all(value == -1 for value in correct):
            status = "noref"
        else:
            status = "ok"

        time_median = statistics.median(times)

        entry = {
            "arm": rows_in_group[0]["arm"],
            "march": rows_in_group[0]["march"],
            "cc_version": rows_in_group[0]["cc_version"],
            "runs": len(rows_in_group),

            # Retained internally for bootstrap resampling.
            "_times": times,

            "time_median": time_median,
            "time_mean": statistics.mean(times),
            "time_std": (
                statistics.pstdev(times)
                if len(times) > 1
                else 0.0
            ),
            "time_min": min(times),
            "time_max": max(times),
            "time_iqr": _iqr(times),

            "gops_median": statistics.median(gops),
            "gops_mean": statistics.mean(gops),
            "gops_std": (
                statistics.pstdev(gops)
                if len(gops) > 1
                else 0.0
            ),
            "gops_min": min(gops),
            "gops_max": max(gops),

            "nnz_c": rows_in_group[0]["nnz_c"],
            "op_mean": rows_in_group[0]["op_mean"],
            "op_max": rows_in_group[0]["op_max"],
            "op_var": rows_in_group[0]["op_var"],
            "status": status,
        }

        entry["time_rel_spread"] = (
            (entry["time_max"] - entry["time_min"]) / time_median
            if time_median > 0
            else 0.0
        )

        cycles = [
            row["cycles"]
            for row in rows_in_group
            if row["cycles"] is not None
        ]
        instructions = [
            row["instructions"]
            for row in rows_in_group
            if row["instructions"] is not None
        ]

        entry["cycles_median"] = (
            statistics.median(cycles) if cycles else None
        )
        entry["instructions_median"] = (
            statistics.median(instructions)
            if instructions
            else None
        )

        entry["ipc"] = (
            entry["instructions_median"] / entry["cycles_median"]
            if (
                entry["cycles_median"]
                and entry["instructions_median"]
            )
            else None
        )

        entry["cycles_per_nnz_c"] = (
            entry["cycles_median"] / entry["nnz_c"]
            if (
                entry["cycles_median"]
                and entry["nnz_c"] > 0
            )
            else None
        )

        summary[key] = entry

    return summary


def _is_baseline(
    key,
    entry,
    base_build,
    baseline_kernel=None,
    baseline_build=None,
):
    """Check whether a summary entry supplies the speedup denominator."""
    _, build, _, kernel, _ = key

    if baseline_kernel is not None:
        return (
            kernel == baseline_kernel
            and build == (baseline_build or base_build)
        )

    return entry["arm"] == BASELINE_ARM


def add_speedups(
    summary,
    base_build,
    baseline_kernel=None,
    baseline_build=None,
):
    """Attach cross-build and within-build speedups plus bootstrap CIs."""
    base_time = {}
    base_times = {}
    same_build_base_time = {}

    # If a baseline appears in a tunable sweep, always prefer the default
    # cflags variant so the denominator remains fixed.
    for key, entry in sorted(summary.items()):
        label, build, cflags, _, dtype = key

        if not _is_baseline(
            key,
            entry,
            base_build,
            baseline_kernel,
            baseline_build,
        ):
            continue

        same_key = (label, build, dtype)
        matrix_key = (label, dtype)

        if (
            same_key not in same_build_base_time
            or cflags == NO_CFLAGS
        ):
            same_build_base_time[same_key] = entry["time_median"]

        if (
            matrix_key not in base_time
            or cflags == NO_CFLAGS
        ):
            base_time[matrix_key] = entry["time_median"]
            base_times[matrix_key] = entry["_times"]

    missing = set()

    for key, entry in summary.items():
        label, build, _, _, dtype = key

        same_base = same_build_base_time.get(
            (label, build, dtype)
        )

        entry["speedup_same_build"] = (
            same_base / entry["time_median"]
            if same_base and entry["time_median"] > 0
            else float("nan")
        )

        matrix_key = (label, dtype)
        base_median = base_time.get(matrix_key)

        if base_median is None:
            missing.add(matrix_key)
            entry["speedup"] = float("nan")
            entry["speedup_lo"] = float("nan")
            entry["speedup_hi"] = float("nan")
            entry["significant"] = None
            continue

        entry["speedup"] = (
            base_median / entry["time_median"]
            if entry["time_median"] > 0
            else float("nan")
        )

        if (
            _is_baseline(
                key,
                entry,
                base_build,
                baseline_kernel,
                baseline_build,
            )
            and key[2] == NO_CFLAGS
        ):
            entry["speedup_lo"] = 1.0
            entry["speedup_hi"] = 1.0
            entry["significant"] = None
            continue

        lo, hi = _bootstrap_speedup_ci(
            base_times[matrix_key],
            entry["_times"],
        )

        entry["speedup_lo"] = lo
        entry["speedup_hi"] = hi

        if lo != lo or hi != hi:
            entry["significant"] = None
        else:
            entry["significant"] = (
                lo > 1.0 or hi < 1.0
            )

    return missing


def _short_cflags(cflags):
    """Make compiler defines readable in the table."""
    if not cflags or cflags == NO_CFLAGS:
        return "-"

    return " ".join(
        token
        .replace("-DRVSP_V2_", "")
        .replace("-D", "")
        for token in cflags.split()
    )


def print_table(
    summary,
    show_perf,
    base_build,
    baseline_kernel=None,
    baseline_build=None,
):
    keys = sorted(
        summary.keys(),
        key=lambda key: (
            key[0],
            key[4],
            key[1],
            key[3],
            key[2],
        ),
    )

    if show_perf:
        tail = f"{'cyc/nnzC':>9} {'IPC':>5}"
    else:
        tail = f"{'nnz_C':>10}"

    header = (
        f"{'matrix':<20} {'bld':<4} {'arm':<14} "
        f"{'kernel':<16} {'cflags':<18} {'dt':<4} "
        f"{'n':>3} {'median_ms':>10} {'spread':>7} "
        f"{'gops':>8} {'speedup':>8} {'95% CI':>15} "
        f"{'sig':>4} {tail} {'st':>5}"
    )

    denominator = (
        f"kernel='{baseline_kernel}'"
        if baseline_kernel
        else f"arm='{BASELINE_ARM}'"
    )

    print(
        f"speedup denominator: {denominator} @ "
        f"{baseline_build or base_build}, same matrix and dtype, "
        f"default cflags\n"
    )
    print(header)
    print("-" * len(header))

    last_matrix = None

    for key in keys:
        label, build, cflags, kernel, dtype = key
        entry = summary[key]

        if last_matrix is not None and label != last_matrix:
            print()

        last_matrix = label

        speedup = entry["speedup"]

        speedup_str = (
            f"{speedup:6.2f}x"
            if speedup == speedup
            else "     - "
        )

        spread_str = (
            f"{entry['time_rel_spread'] * 100:6.1f}%"
        )

        lo = entry.get("speedup_lo", float("nan"))
        hi = entry.get("speedup_hi", float("nan"))

        if lo == lo and hi == hi:
            ci_str = f"[{lo:4.2f},{hi:4.2f}]"
        else:
            ci_str = "         -"

        significant = entry.get("significant")

        if significant is None:
            sig_str = "  - "
        elif significant:
            sig_str = " sig"
        else:
            sig_str = "  ns"

        if show_perf:
            cycles_per_nnz = entry["cycles_per_nnz_c"]

            cyc_str = (
                f"{cycles_per_nnz:9.1f}"
                if cycles_per_nnz
                else "        -"
            )
            ipc_str = (
                f"{entry['ipc']:5.2f}"
                if entry["ipc"]
                else "    -"
            )

            tail_value = f"{cyc_str} {ipc_str}"
        else:
            tail_value = f"{entry['nnz_c']:>10}"

        print(
            f"{label:<20} {build:<4} {entry['arm']:<14} "
            f"{kernel:<16} {_short_cflags(cflags):<18} "
            f"{dtype:<4} {entry['runs']:>3} "
            f"{entry['time_median'] * 1e3:>10.4f} "
            f"{spread_str:>7} {entry['gops_median']:>8.3f} "
            f"{speedup_str:>8} {ci_str:>15} "
            f"{sig_str:>4} {tail_value} {entry['status']:>5}"
        )


def write_csv(summary, path):
    keys = sorted(
        summary.keys(),
        key=lambda key: (
            key[0],
            key[4],
            key[1],
            key[3],
            key[2],
        ),
    )

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "matrix",
            "build",
            "arm",
            "kernel",
            "cflags",
            "dtype",
            "runs",
            "time_median_s",
            "time_mean_s",
            "time_std_s",
            "time_min_s",
            "time_max_s",
            "time_iqr_s",
            "time_rel_spread",
            "gops_median",
            "gops_mean",
            "gops_std",
            "gops_min",
            "gops_max",
            "speedup_vs_gc_scalar",
            "speedup_ci_lo",
            "speedup_ci_hi",
            "speedup_significant",
            "speedup_vs_scalar_same_build",
            "nnz_c",
            "op_mean",
            "op_max",
            "op_var",
            "cycles_median",
            "instructions_median",
            "ipc",
            "cycles_per_nnz_c",
            "march",
            "cc_version",
            "status",
        ])

        for key in keys:
            label, build, cflags, kernel, dtype = key
            entry = summary[key]

            def opt(value, fmt="{:.4f}"):
                return (
                    fmt.format(value)
                    if value is not None
                    else ""
                )

            def num(value, fmt="{:.4f}"):
                return (
                    fmt.format(value)
                    if value is not None and value == value
                    else ""
                )

            significant = entry.get("significant")

            if significant is None:
                significant_str = ""
            else:
                significant_str = (
                    "yes" if significant else "no"
                )

            writer.writerow([
                label,
                build,
                entry["arm"],
                kernel,
                cflags,
                dtype,
                entry["runs"],
                f"{entry['time_median']:.9f}",
                f"{entry['time_mean']:.9f}",
                f"{entry['time_std']:.9f}",
                f"{entry['time_min']:.9f}",
                f"{entry['time_max']:.9f}",
                f"{entry['time_iqr']:.9f}",
                f"{entry['time_rel_spread']:.4f}",
                f"{entry['gops_median']:.4f}",
                f"{entry['gops_mean']:.4f}",
                f"{entry['gops_std']:.4f}",
                f"{entry['gops_min']:.4f}",
                f"{entry['gops_max']:.4f}",
                num(entry["speedup"]),
                num(entry.get("speedup_lo")),
                num(entry.get("speedup_hi")),
                significant_str,
                num(entry.get("speedup_same_build")),
                entry["nnz_c"],
                opt(entry["op_mean"], "{:.4f}"),
                opt(entry["op_max"], "{:.0f}"),
                opt(entry["op_var"], "{:.4f}"),
                opt(entry["cycles_median"], "{:.0f}"),
                opt(entry["instructions_median"], "{:.0f}"),
                opt(entry["ipc"], "{:.4f}"),
                opt(
                    entry["cycles_per_nnz_c"],
                    "{:.4f}",
                ),
                entry["march"],
                entry["cc_version"],
                entry["status"],
            ])

    print(f"\nwrote summary csv -> {path}")


def main():
    argv = sys.argv[1:]

    if not argv:
        print(
            "usage: python3 analyze.py RAW.csv "
            "[--csv-out summary.csv] [--no-csv] "
            "[--baseline KERNEL]"
        )
        sys.exit(2)

    path = argv[0]

    csv_out = os.path.join(
        os.path.dirname(os.path.abspath(path)),
        "summary.csv",
    )

    if "--no-csv" in argv:
        csv_out = None

    if "--csv-out" in argv:
        index = argv.index("--csv-out")

        if index + 1 >= len(argv):
            print("--csv-out needs a path")
            sys.exit(2)

        csv_out = argv[index + 1]

    baseline_kernel = None

    if "--baseline" in argv:
        index = argv.index("--baseline")

        if index + 1 >= len(argv):
            print("--baseline needs a kernel name")
            sys.exit(2)

        baseline_kernel = argv[index + 1]

    rows = load(path)

    if not rows:
        print("no valid rows found in", path)
        sys.exit(1)

    base_build, base_note = pick_baseline_build(rows)

    if base_note:
        print(f"NOTE: {base_note}\n")

    baseline_build = base_build

    if baseline_kernel is not None:
        present = sorted({row["kernel"] for row in rows})

        if baseline_kernel not in present:
            print(
                f"--baseline '{baseline_kernel}' does not appear in "
                f"{path}.\nkernels present: {', '.join(present)}"
            )
            sys.exit(2)

        builds = sorted({
            row["build"]
            for row in rows
            if row["kernel"] == baseline_kernel
        })

        if base_build in builds:
            baseline_build = base_build

        elif len(builds) == 1:
            baseline_build = builds[0]
            print(
                f"NOTE: --baseline '{baseline_kernel}' is not built at "
                f"'{base_build}'; using its only build, "
                f"'{baseline_build}'.\n"
            )

        else:
            print(
                f"--baseline '{baseline_kernel}' exists at builds "
                f"{builds} but not at '{base_build}', so the denominator "
                f"is ambiguous."
            )
            sys.exit(2)

    summary = summarize(rows)

    missing_base = add_speedups(
        summary,
        base_build,
        baseline_kernel,
        baseline_build,
    )

    show_perf = any(
        entry["cycles_median"] is not None
        for entry in summary.values()
    )

    print_table(
        summary,
        show_perf,
        base_build,
        baseline_kernel,
        baseline_build,
    )

    if not show_perf:
        print(
            "\n(no perf counters in this CSV — run the driver with "
            "PERF=1 to add cycles/instructions)"
        )

    if missing_base:
        print(
            f"\n*** NO {base_build.upper()}-SCALAR BASELINE for "
            f"{len(missing_base)} (matrix,dtype) pair(s) — "
            f"speedup left blank ***"
        )

        for label, dtype in sorted(missing_base):
            print(f"    {label} ({dtype})")

    not_significant = [
        (key, entry)
        for key, entry in summary.items()
        if entry.get("significant") is False
    ]

    if not_significant:
        print(
            "\n*** EFFECT WITHIN NOISE — do NOT claim faster/slower "
            "for these ***"
        )
        print(
            "    (95% speedup CI includes 1.0x — the sign of the "
            "effect is not resolved)"
        )

        for key, entry in sorted(not_significant):
            label, build, cflags, kernel, dtype = key

            print(
                f"    {label} / "
                f"{kernel}@{build}[{_short_cflags(cflags)}] "
                f"({dtype}): {entry['speedup']:.3f}x, "
                f"CI [{entry['speedup_lo']:.3f}, "
                f"{entry['speedup_hi']:.3f}]"
            )

    dirty = [
        (key, entry)
        for key, entry in summary.items()
        if (
            entry["time_rel_spread"] > CLEAN_SPREAD
            and entry["runs"] >= 4
        )
    ]

    if dirty:
        print(
            f"\n*** SPREAD > {CLEAN_SPREAD * 100:.0f}% — "
            f"too noisy to trust a small effect here ***"
        )

        for key, entry in sorted(dirty):
            label, build, cflags, kernel, _ = key

            print(
                f"    {label} / "
                f"{kernel}@{build}[{_short_cflags(cflags)}]: "
                f"{entry['time_rel_spread'] * 100:.1f}% "
                f"(min-max). A 2-4% claim needs spread well under 2%."
            )

    noisy = [
        (key, entry)
        for key, entry in summary.items()
        if entry["time_rel_spread"] > NOISY_SPREAD
    ]

    if noisy:
        print(
            f"\n*** HIGH VARIANCE "
            f"(>{NOISY_SPREAD * 100:.0f}% min-max spread) ***"
        )

        for key, entry in sorted(noisy):
            label, build, cflags, kernel, _ = key

            print(
                f"    {label} / "
                f"{kernel}@{build}[{_short_cflags(cflags)}]: "
                f"{entry['time_rel_spread'] * 100:.1f}% "
                f"— more runs or a quieter machine"
            )

    failures = [
        key
        for key, entry in summary.items()
        if entry["status"] == "FAIL"
    ]

    if failures:
        print("\n*** CORRECTNESS FAILURES ***")

        for label, build, cflags, kernel, dtype in failures:
            print(
                f"    {label} / "
                f"{kernel}@{build}[{_short_cflags(cflags)}] "
                f"({dtype}) produced wrong output — perf number invalid"
            )

    if csv_out:
        write_csv(summary, csv_out)


if __name__ == "__main__":
    main()