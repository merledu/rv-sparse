#!/usr/bin/env bash

# Driver for the rv-sparse SpGEMM benchmarks.
#
# Runs one config at a time on one pinned core, validates every result, and
# writes self-describing rows to an append-only CSV.
#
# bench/experiments.tsv is the single source of truth for which experiments
# run. This script handles build, measurement, validation, resume, and
# environment control.
#
# An arm identifies what an experiment is evidence for. It is partly a
# property of the code and partly of how it was compiled:
#
#   baseline       scalar source, rv64gc
#   autovec        same scalar source, rv64gcv
#   intrinsic      handwritten RVV kernel
#   scalar_unroll  manually unrolled scalar kernel
#   adaptive       adaptive RVV kernel
#
# The raw CSV records arm, build, march, cflags, and compiler version so each
# measurement is self-describing.
#
# Measurement controls:
#   - exclusive lock and atomic CSV writes
#   - smoke test for every build variant
#   - performance governor when available
#   - frequency check before/after each config
#   - randomized experiment order per matrix
#
# For effects below about 5%, isolate the benchmark core with isolcpus or a
# cpuset; taskset alone does not prevent other scheduler activity.
#
# Setup:
#   cp bench/env.sh.example bench/env.sh
#   # set CC
#   bash bench/run_bench.sh --check
#
# Run from the repository root.
#
# Flags:
#   --check                 preflight only
#   --check-matrices        validate canonical CSR input matrices
#   --dtype f32             restrict by dtype; repeatable or comma-separated
#   --kernels a,b           restrict by kernel name or arm
#   --baseline NAME         baseline kernel for batch composition
#   --runs N                timed runs per config (default 10)
#   --experiments PATH      alternate experiment table
#   -h | --help             show this message
#
# Common cases:
#   bash bench/run_bench.sh
#   bash bench/run_bench.sh --kernels contig_f32
#   bash bench/run_bench.sh --kernels intrinsic --runs 30
#
# Everything else is configured in bench/env.sh.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || {
    echo "!! cannot cd to $REPO_ROOT" >&2
    exit 1
}

die() {
    echo "!! $*" >&2
    exit 1
}

# CLI
CHECK_ONLY=0
CHECK_MATRICES=0
DTYPE_FILTER=""
KERNEL_FILTER=""
RUNS_FLAG=""
EXP_FILE="bench/experiments.tsv"

BASELINE_KERNEL="scalar_f32"

usage() {
    awk 'NR==1 { next }
         /^#/ {
             sub(/^# ?/, "")
             print
             next
         }
         /^[[:space:]]*$/ { next }
         { exit }' "${BASH_SOURCE[0]}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --check)
            CHECK_ONLY=1
            shift
            ;;
        --check-matrices)
            CHECK_MATRICES=1
            shift
            ;;
        --dtype)
            [ $# -ge 2 ] || die "--dtype needs a value"
            DTYPE_FILTER="${DTYPE_FILTER}${DTYPE_FILTER:+,}$2"
            shift 2
            ;;
        --dtype=*)
            DTYPE_FILTER="${DTYPE_FILTER}${DTYPE_FILTER:+,}${1#*=}"
            shift
            ;;
        --kernels)
            [ $# -ge 2 ] || die "--kernels needs a value"
            KERNEL_FILTER="${KERNEL_FILTER}${KERNEL_FILTER:+,}$2"
            shift 2
            ;;
        --kernels=*)
            KERNEL_FILTER="${KERNEL_FILTER}${KERNEL_FILTER:+,}${1#*=}"
            shift
            ;;
        --baseline)
            [ $# -ge 2 ] || die "--baseline needs a value"
            BASELINE_KERNEL="$2"
            shift 2
            ;;
        --baseline=*)
            BASELINE_KERNEL="${1#*=}"
            shift
            ;;
        --runs)
            [ $# -ge 2 ] || die "--runs needs a value"
            RUNS_FLAG="$2"
            shift 2
            ;;
        --runs=*)
            RUNS_FLAG="${1#*=}"
            shift
            ;;
        --experiments)
            [ $# -ge 2 ] || die "--experiments needs a value"
            EXP_FILE="$2"
            shift 2
            ;;
        --experiments=*)
            EXP_FILE="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown flag: $1 (try --help)"
            ;;
    esac
done

ENV_FILE="${ENV_FILE:-bench/env.sh}"

if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$ENV_FILE" || die "failed to source $ENV_FILE"
else
    die "no $ENV_FILE on this box.

cp bench/env.sh.example bench/env.sh
\$EDITOR bench/env.sh

Set CC to a compiler that supports -march=rv64gcv, then re-run with --check."
fi

OUT_DIR="${OUT_DIR:-bench/results}"
CSV="${CSV:-$OUT_DIR/spgemm_raw.csv}"
RUNS="${RUNS_FLAG:-${RUNS:-10}}"
WARMUP="${WARMUP:-3}"
PIN_CORE="${PIN_CORE:-0}"
CC="${CC:-}"
PERF="${PERF:-0}"
FORCE="${FORCE:-0}"
ONLY_MTX="${ONLY_MTX:-}"
RETRIES="${RETRIES:-1}"
SHUFFLE="${SHUFFLE:-1}"
SHUF_SEED="${SHUF_SEED:-$RANDOM}"
NO_GOVERNOR="${NO_GOVERNOR:-0}"

GEN_CASES=(
    "512 512 0.02 42"
    "1024 1024 0.01 42"
    "2048 2048 0.005 42"
    "4096 4096 0.002 42"
)

MATRICES=(
    wiki-Vote
    email-Enron
    p2p-Gnutella31
    cage12
    2cubes_sphere
    scircuit
    poisson3Da
    mario002
    cop20k_A
    offshore
    m133-b3
    filter3D
    ca-CondMat
    amazon0312
    web-Google
    roadNet-CA
    patents_main
    cit-Patents
    webbase-1M
)

CSV_HEADER="label,kernel,arm,build,march,cflags,cc_version,dtype,rows,cols,nnz_a,nnz_b,nnz_c,flops,op_mean,op_max,op_var,run,time_s,gops,correct,cycles,instructions"
CSV_NFIELDS=23

F_LABEL=1
F_KERNEL=2
F_BUILD=4
F_CFLAGS=6
F_CORRECT=21

VALID_ARMS="baseline autovec intrinsic adaptive"
VALID_BUILDS="gc gcv"
VALID_DTYPES="f32 f64 i8"

# Vector kernels require rv64gcv. Those sources deliberately #error under gc
# rather than silently falling back to scalar code.
vector_kernel() {
    case "$1" in
        rvv_*|contig_*|adaptive_*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

cflags_slug() {
    if [ "$1" = "-" ] || [ -z "$1" ]; then
        echo "default"
    else
        printf '%s' "$1" |
            tr -cs 'A-Za-z0-9' '_' |
            sed 's/^_//; s/_$//'
    fi
}

FREQ_DRIFT_WARN=0.01

# The build-to-march mapping lives here so the experiment table never encodes
# ISA flags directly.
build_march() {
    case "$1" in
        gc)
            echo "-march=rv64gc -mabi=lp64d"
            ;;
        gcv)
            echo "-march=rv64gcv -mabi=lp64d"
            ;;
        *)
            die "unknown build '$1' (expected one of: $VALID_BUILDS)"
            ;;
    esac
}

in_list() {
    local n="$1"
    local h="$2"
    local x

    for x in $h; do
        [ "$x" = "$n" ] && return 0
    done

    return 1
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_uint "$RUNS" ||
    die "RUNS must be a non-negative integer, got '$RUNS'"

is_uint "$WARMUP" ||
    die "WARMUP must be a non-negative integer, got '$WARMUP'"

is_uint "$PIN_CORE" ||
    die "PIN_CORE must be a non-negative integer, got '$PIN_CORE'"

is_uint "$RETRIES" ||
    die "RETRIES must be a non-negative integer, got '$RETRIES'"

is_uint "$SHUF_SEED" ||
    die "SHUF_SEED must be a non-negative integer, got '$SHUF_SEED'"

[ "$RUNS" -ge 1 ] ||
    die "RUNS must be >= 1"

if [ "$RUNS" -lt 4 ]; then
    echo ">> WARNING: RUNS=$RUNS (<4) — analyze.py cannot bootstrap a speedup CI with so few runs. Use RUNS>=10 for a paper."
fi

# ============================================================================
# PREFLIGHT
# ============================================================================

CC_VERSION=""
SYNTHETIC_ONLY=0
PRESENT_MATRICES=()

# Matrix files live flat as matrices/<name>.mtx. The nested layout that
# SuiteSparse tarballs extract to is still accepted so existing checkouts and
# partially fetched trees keep working.
mtx_path() {
    local name="$1"

    [ -f "matrices/$name.mtx" ] || return 1

    echo "matrices/$name.mtx"
}

preflight() {
    echo ">> preflight"

    [ -n "$CC" ] ||
        die "CC is not set. Edit $ENV_FILE and set CC to a compiler that supports -march=rv64gcv."

    command -v "$CC" >/dev/null 2>&1 || [ -x "$CC" ] ||
        die "CC='$CC' is not executable. Fix CC in $ENV_FILE."

    echo "   CC = $CC"

    local missing=()
    local t

    for t in make python3 awk flock mktemp; do
        command -v "$t" >/dev/null 2>&1 || missing+=("$t")
    done

    if [ "${#missing[@]}" -gt 0 ]; then
        die "missing required tool(s): ${missing[*]}"
    fi

    local libgcc
    local libdir

    libgcc="$("$CC" -print-file-name=libgcc_s.so.1 2>/dev/null)"

    if [ -n "$libgcc" ] &&
       [ "$libgcc" != "libgcc_s.so.1" ] &&
       [ -e "$libgcc" ]; then

        libdir="$(cd "$(dirname "$libgcc")" && pwd)"

        export LIBRARY_PATH="${libdir}${LIBRARY_PATH:+:$LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${libdir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

        echo "   lib path (derived from CC, this process only): $libdir"
    else
        echo "   NOTE: could not derive a libgcc path from CC — assuming a system compiler."
    fi

    CC_VERSION="$("$CC" -dumpfullversion 2>/dev/null)"
    [ -n "$CC_VERSION" ] ||
        CC_VERSION="$("$CC" -dumpversion 2>/dev/null)"

    [ -n "$CC_VERSION" ] ||
        die "could not read a version from CC='$CC'"

    case "$CC_VERSION" in
        *,*)
            die "CC version '$CC_VERSION' contains a comma; it would corrupt the CSV"
            ;;
    esac

    echo "   cc_version = $CC_VERSION"

    # Compile and execute a real RVV program so the preflight catches both
    # toolchain support and an environment that cannot execute RVV binaries.
    local tmpd
    local rc

    tmpd="$(mktemp -d)" ||
        die "mktemp -d failed"

    cat > "$tmpd/probe.c" <<'PROBE'
#include <riscv_vector.h>

int main(void) {
    size_t vl = __riscv_vsetvl_e32m1(8);
    if (vl < 1)
        return 1;

    vfloat32m1_t a = __riscv_vfmv_v_f_f32m1(1.5f, vl);
    vfloat32m1_t b = __riscv_vfmacc_vf_f32m1(a, 2.0f, a, vl);
    float out = __riscv_vfmv_f_s_f32m1_f32(b);

    return (out > 4.4f && out < 4.6f) ? 0 : 2;
}
PROBE

    if ! "$CC" -march=rv64gcv -mabi=lp64d -O2 \
        "$tmpd/probe.c" -o "$tmpd/probe" 2>"$tmpd/err"; then

        echo "---- compiler output ----" >&2
        cat "$tmpd/err" >&2
        echo "-------------------------" >&2

        rm -rf "$tmpd"

        die "CC='$CC' cannot compile a -march=rv64gcv program."
    fi

    "$tmpd/probe"
    rc=$?

    if [ "$rc" -eq 0 ]; then
        echo "   rv64gcv probe: compiled and ran correctly"
    elif [ "$rc" -eq 2 ]; then
        rm -rf "$tmpd"

        die "rv64gcv probe compiled and ran but produced the WRONG result.
The vector unit or its emulation is not behaving correctly."
    else
        rm -rf "$tmpd"

        die "rv64gcv probe compiled but could not execute here (exit $rc).
You are most likely cross-compiling: build on the target, or use an emulator."
    fi

    rm -rf "$tmpd"

    local gov_path="/sys/devices/system/cpu/cpu${PIN_CORE}/cpufreq/scaling_governor"

    if [ "$NO_GOVERNOR" = "1" ]; then
        echo "   NOTE: governor pin disabled (NO_GOVERNOR=1)"
    elif [ -w "$gov_path" ]; then
        echo "   cpufreq governor is writable — will pin to 'performance'"
    else
        echo "   WARNING: $gov_path not writable (need root). DVFS stays live — do NOT trust sub-5% effects."
    fi

    if [ "$PERF" = "1" ]; then
        if [ -r /proc/sys/kernel/perf_event_paranoid ]; then
            local p
            p="$(cat /proc/sys/kernel/perf_event_paranoid)"

            if [ "$p" -le 2 ] 2>/dev/null; then
                echo "   perf counters available (perf_event_paranoid=$p)"
            else
                echo "   WARNING: perf_event_paranoid=$p (need <= 2). cycles/instructions will be blank."
            fi
        else
            echo "   WARNING: cannot read perf_event_paranoid; perf counters may be unavailable."
        fi
    fi

    local isolated
    isolated="$(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo '')"

    if [ -z "$isolated" ] ||
       ! grep -qw "$PIN_CORE" <<< "${isolated//,/ }"; then

        echo "   WARNING: core $PIN_CORE is not isolated ('${isolated:-none}')."
        echo "            For sub-5% claims, boot with isolcpus=$PIN_CORE."
    fi

    local name
    local missing_m=()

    for name in "${MATRICES[@]}"; do
        if mtx_path "$name" >/dev/null; then
            PRESENT_MATRICES+=("$name")
        else
            missing_m+=("$name")
        fi
    done

    if [ "${#PRESENT_MATRICES[@]}" -eq 0 ]; then
        SYNTHETIC_ONLY=1
        echo "   WARNING: no .mtx files found under matrices/ — running SYNTHETIC ONLY."
        echo "            Fetch them with matrices/getResources.sh for the real-matrix sweep."
    elif [ "${#missing_m[@]}" -gt 0 ]; then
        echo "   ${#PRESENT_MATRICES[@]}/${#MATRICES[@]} matrices present; skipping ${#missing_m[@]}: ${missing_m[*]}"
    else
        echo "   all ${#MATRICES[@]} matrices present"
    fi

    echo ">> preflight OK"
}

preflight

if [ "$CHECK_ONLY" = "1" ]; then
    echo ""
    echo ">> --check requested: preflight only, nothing was built and nothing was run."
    exit 0
fi

# ============================================================================
# --check-matrices
#
# The kernels assume canonical CSR input. Validate the dataset independently of the
# kernels before benchmarking.
# ============================================================================

if [ "$CHECK_MATRICES" = "1" ]; then
    if [ "${#PRESENT_MATRICES[@]}" -eq 0 ]; then
        die "no .mtx files present under matrices/ — nothing to check."
    fi

    b=gc
    march="$(build_march "$b")"

    objdir="obj/$b"
    libdir="lib/$b"
    bindir="bin/$b"

    tools_objs=(
        "$objdir/tools/genmat.o"
        "$objdir/tools/mtx_to_csr_formatter.o"
        "$objdir/tools/vec.o"
    )

    echo ">> building csr_check ($march)"

    make CC="$CC" \
        ARCH_FLAGS="$march" \
        OBJ_DIR="$objdir" \
        LIB_DIR="$libdir" \
        BIN_DIR="$bindir" \
        "$libdir/librvsparse.a" \
        "${tools_objs[@]}" >/dev/null ||
        die "library build failed"

    "$CC" -Wall -Wextra -std=c11 \
        -Iinclude \
        -Itools/include \
        -Isrc/kernels/spgemm \
        $march \
        -O2 \
        bench/csr_check.c \
        "${tools_objs[@]}" \
        -L"$libdir" \
        -lrvsparse \
        -lm \
        -o bench/csr_check ||
        die "csr_check build failed"

    MTX_PATHS=()

    for name in "${PRESENT_MATRICES[@]}"; do
        MTX_PATHS+=("$(mtx_path "$name")")
    done

    echo ">> checking ${#MTX_PATHS[@]} matrices"
    echo ""

    ./bench/csr_check "${MTX_PATHS[@]}"
    rc=$?

    echo ""

    if [ "$rc" -eq 0 ]; then
        echo ">> all matrices satisfy the canonical-CSR precondition."
    else
        echo ">> PRECONDITION VIOLATED — do not run the kernels on this dataset until fixed."
    fi

    exit "$rc"
fi

# ============================================================================
# EXPERIMENT TABLE
# ============================================================================

EXP_ARM=()
EXP_KERNEL=()
EXP_DTYPE=()
EXP_BUILD=()
EXP_CFLAGS=()

POOL_ARM=()
POOL_KERNEL=()
POOL_DTYPE=()
POOL_BUILD=()
POOL_CFLAGS=()

ALL_KERNELS=()

load_experiments() {
    [ -f "$EXP_FILE" ] ||
        die "experiment table not found: $EXP_FILE"

    local arm kernel dtype build cflags
    local n=0
    local kept=0

    local -a want_dtypes=()

    if [ -n "$DTYPE_FILTER" ]; then
        IFS=',' read -ra want_dtypes <<< "$DTYPE_FILTER"

        local d

        for d in "${want_dtypes[@]}"; do
            in_list "$d" "$VALID_DTYPES" ||
                die "--dtype '$d' is not one of: $VALID_DTYPES"
        done
    fi

    # Validate the whole table before filtering so malformed rows are always
    # reported rather than hidden by the current selection.
    while IFS=$'\t' read -r arm kernel dtype build cflags; do
        n=$((n + 1))

        [ -n "$arm" ] &&
        [ -n "$kernel" ] &&
        [ -n "$dtype" ] &&
        [ -n "$build" ] ||
            die "$EXP_FILE row $n: expected 5 tab-separated fields"

        [ -n "$cflags" ] || cflags="-"

        case "$cflags" in
            *,*)
                die "$EXP_FILE row $n: cflags '$cflags' contains a comma; it would corrupt the CSV"
                ;;
        esac

        in_list "$arm" "$VALID_ARMS" ||
            die "$EXP_FILE row $n: arm '$arm' is not one of: $VALID_ARMS"

        in_list "$dtype" "$VALID_DTYPES" ||
            die "$EXP_FILE row $n: dtype '$dtype' is not one of: $VALID_DTYPES"

        in_list "$build" "$VALID_BUILDS" ||
            die "$EXP_FILE row $n: build '$build' is not one of: $VALID_BUILDS"

        if vector_kernel "$kernel" && [ "$build" = "gc" ]; then
            die "$EXP_FILE row $n: vector kernel '$kernel' is paired with build 'gc'.
Vector kernels require build 'gcv'."
        fi

        case "$kernel" in
            *_"$dtype")
                ;;
            *)
                die "$EXP_FILE row $n: kernel '$kernel' does not end in '_$dtype' — dtype column and kernel name disagree."
                ;;
        esac

        in_list "$kernel" "${ALL_KERNELS[*]:-}" ||
            ALL_KERNELS+=("$kernel")

        if [ -n "$DTYPE_FILTER" ] &&
           ! in_list "$dtype" "${want_dtypes[*]}"; then
            continue
        fi

        POOL_ARM+=("$arm")
        POOL_KERNEL+=("$kernel")
        POOL_DTYPE+=("$dtype")
        POOL_BUILD+=("$build")
        POOL_CFLAGS+=("$cflags")

        kept=$((kept + 1))

    done < <(
        awk -F'\t' '
            /^[[:space:]]*#/ { next }
            /^[[:space:]]*$/ { next }
            $1 == "arm" && $2 == "kernel" { next }
            { print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 }
        ' "$EXP_FILE"
    )

    [ "$n" -gt 0 ] ||
        die "$EXP_FILE has no experiment rows"

    [ "$kept" -gt 0 ] ||
        die "no experiment rows match --dtype '$DTYPE_FILTER'"

    local -a want_kernels=()

    if [ -n "$KERNEL_FILTER" ]; then
        IFS=',' read -ra want_kernels <<< "$KERNEL_FILTER"

        local k

        for k in "${want_kernels[@]}"; do
            if in_list "$k" "${ALL_KERNELS[*]}" ||
               in_list "$k" "$VALID_ARMS"; then
                continue
            fi

            die "--kernels '$k' is neither a kernel in $EXP_FILE nor an arm.

kernels: ${ALL_KERNELS[*]}
arms:    $VALID_ARMS"
        done
    fi

    local i

    for ((i = 0; i < ${#POOL_ARM[@]}; i++)); do
        if [ -n "$KERNEL_FILTER" ]; then
            if ! in_list "${POOL_KERNEL[i]}" "${want_kernels[*]}" &&
               ! in_list "${POOL_ARM[i]}" "${want_kernels[*]}"; then
                continue
            fi
        fi

        EXP_ARM+=("${POOL_ARM[i]}")
        EXP_KERNEL+=("${POOL_KERNEL[i]}")
        EXP_DTYPE+=("${POOL_DTYPE[i]}")
        EXP_BUILD+=("${POOL_BUILD[i]}")
        EXP_CFLAGS+=("${POOL_CFLAGS[i]}")
    done

    [ "${#EXP_ARM[@]}" -gt 0 ] ||
        die "no experiment rows match --kernels '$KERNEL_FILTER'"

    # Filtered batches must contain the requested baseline so speedups can be
    # computed without requiring a second benchmark invocation.
    if [ -n "$KERNEL_FILTER" ] &&
       ! in_list "$BASELINE_KERNEL" "${EXP_KERNEL[*]}"; then

        local -a add=()
        local baseline_idx=-1

        for ((i = 0; i < ${#POOL_ARM[@]}; i++)); do
            [ "${POOL_KERNEL[i]}" = "$BASELINE_KERNEL" ] || continue

            if [ "${POOL_ARM[i]}" = "baseline" ]; then
                baseline_idx="$i"
                break
            fi
        done

        if [ "$baseline_idx" -ge 0 ]; then
            add=("$baseline_idx")
        else
            for ((i = 0; i < ${#POOL_ARM[@]}; i++)); do
                [ "${POOL_KERNEL[i]}" = "$BASELINE_KERNEL" ] || continue
                add+=("$i")
            done
        fi

        [ "${#add[@]}" -gt 0 ] ||
            die "--baseline '$BASELINE_KERNEL' has no row in $EXP_FILE${DTYPE_FILTER:+ for dtype '$DTYPE_FILTER'}."

        local j

        for j in "${add[@]}"; do
            EXP_ARM+=("${POOL_ARM[j]}")
            EXP_KERNEL+=("${POOL_KERNEL[j]}")
            EXP_DTYPE+=("${POOL_DTYPE[j]}")
            EXP_BUILD+=("${POOL_BUILD[j]}")
            EXP_CFLAGS+=("${POOL_CFLAGS[j]}")
        done

        echo ">> baseline '$BASELINE_KERNEL' was not in --kernels; added ${#add[@]} row(s) for it."
    fi

    kept="${#EXP_ARM[@]}"

    echo ">> experiment table: $EXP_FILE — $kept of $n rows selected${DTYPE_FILTER:+ (dtype: $DTYPE_FILTER)}${KERNEL_FILTER:+ (kernels: $KERNEL_FILTER)}"

    for ((i = 0; i < ${#EXP_ARM[@]}; i++)); do
        printf '     %-14s %-17s %-4s %-4s %s\n' \
            "${EXP_ARM[i]}" \
            "${EXP_KERNEL[i]}" \
            "${EXP_DTYPE[i]}" \
            "${EXP_BUILD[i]}" \
            "${EXP_CFLAGS[i]}"
    done
}

load_experiments

# Compile-time tunables require separate build artifacts for every unique
# (build,cflags) pair.
VARIANT_BUILD=()
VARIANT_CFLAGS=()
VARIANT_TAG=()

for ((i = 0; i < ${#EXP_BUILD[@]}; i++)); do
    tag="${EXP_BUILD[i]}__$(cflags_slug "${EXP_CFLAGS[i]}")"

    if in_list "$tag" "${VARIANT_TAG[*]:-}"; then
        continue
    fi

    VARIANT_BUILD+=("${EXP_BUILD[i]}")
    VARIANT_CFLAGS+=("${EXP_CFLAGS[i]}")
    VARIANT_TAG+=("$tag")
done

echo ">> binaries to build: ${#VARIANT_TAG[@]}"

for ((i = 0; i < ${#VARIANT_TAG[@]}; i++)); do
    printf '     bench/bench_%-28s march=%s  cflags=%s\n' \
        "${VARIANT_TAG[i]}" \
        "$(build_march "${VARIANT_BUILD[i]}")" \
        "${VARIANT_CFLAGS[i]}"
done

exp_tag() {
    echo "${EXP_BUILD[$1]}__$(cflags_slug "${EXP_CFLAGS[$1]}")"
}

# ============================================================================
# LOCK / CLEANUP
# ============================================================================

mkdir -p "$OUT_DIR" ||
    die "cannot create $OUT_DIR"

LOCKFILE="$OUT_DIR/.run_bench.lock"

exec 200>"$LOCKFILE"

if ! flock -n 200; then
    die "another run_bench.sh is already running against $OUT_DIR"
fi

GOV_PATH="/sys/devices/system/cpu/cpu${PIN_CORE}/cpufreq/scaling_governor"
FREQ_PATH="/sys/devices/system/cpu/cpu${PIN_CORE}/cpufreq/scaling_cur_freq"
ORIG_GOV=""

read_freq() {
    cat "$FREQ_PATH" 2>/dev/null || echo 0
}

set_governor_performance() {
    [ "$NO_GOVERNOR" = "1" ] && {
        echo ">> governor pin skipped (NO_GOVERNOR=1)"
        return
    }

    if [ -w "$GOV_PATH" ]; then
        ORIG_GOV="$(cat "$GOV_PATH" 2>/dev/null || echo '')"

        if echo performance > "$GOV_PATH" 2>/dev/null; then
            echo ">> cpufreq governor set to 'performance' on core $PIN_CORE (was '${ORIG_GOV:-unknown}')"
        else
            echo ">> WARNING: could not write $GOV_PATH — DVFS is live."
            ORIG_GOV=""
        fi
    else
        echo ">> WARNING: $GOV_PATH not writable (need root). DVFS is live."
    fi
}

restore_governor() {
    if [ -n "$ORIG_GOV" ] && [ -w "$GOV_PATH" ]; then
        echo "$ORIG_GOV" > "$GOV_PATH" 2>/dev/null &&
            echo ">> restored governor to '$ORIG_GOV'"
    fi
}

cleanup() {
    restore_governor
}

trap cleanup EXIT

on_interrupt() {
    echo ""
    echo "!! interrupted — CSV writes are atomic, so $CSV is not corrupted."
    echo "!! re-run this script to resume."
    exit 130
}

trap on_interrupt INT TERM

# ============================================================================
# BUILD
# ============================================================================

PERF_FLAG=""

if [ "$PERF" = "1" ]; then
    PERF_FLAG="-DUSE_PERF"
    echo ">> perf counters ENABLED (cycles, instructions — user-space only)"
fi

build_all() {
    local b="$1"
    local cflags="$2"
    local tag="$3"

    local march
    local extra
    local objdir
    local libdir
    local bindir

    march="$(build_march "$b")"
    extra=""

    [ "$cflags" != "-" ] && extra="$cflags"

    objdir="obj/$tag"
    libdir="lib/$tag"
    bindir="bin/$tag"

    echo ">> [$tag] building librvsparse.a ($march${extra:+ $extra})"

    local tools_objs=(
        "$objdir/tools/genmat.o"
        "$objdir/tools/mtx_to_csr_formatter.o"
        "$objdir/tools/vec.o"
    )

    make CC="$CC" \
        ARCH_FLAGS="$march $extra" \
        OBJ_DIR="$objdir" \
        LIB_DIR="$libdir" \
        BIN_DIR="$bindir" \
        "$libdir/librvsparse.a" \
        "${tools_objs[@]}" >/dev/null ||
        die "[$tag] library build failed"

    local f

    for f in "$libdir/librvsparse.a" "${tools_objs[@]}"; do
        [ -f "$f" ] ||
            die "[$tag] expected build artifact missing: $f"
    done

    echo ">> [$tag] compiling bench.c -> bench/bench_$tag"

    "$CC" \
        -Wall -Wextra \
        -std=c11 \
        -Iinclude \
        -Itools/include \
        -Isrc/kernels/spgemm \
        $march \
        $extra \
        -O3 \
        $PERF_FLAG \
        bench/bench.c \
        "${tools_objs[@]}" \
        -L"$libdir" \
        -lrvsparse \
        -lm \
        -o "bench/bench_$tag" ||
        die "[$tag] bench build failed"

    [ -x "bench/bench_$tag" ] ||
        die "bench/bench_$tag was not produced or is not executable"
}

RUNNER=""

if command -v taskset >/dev/null 2>&1; then
    RUNNER="taskset -c $PIN_CORE"
    echo ">> pinning to core $PIN_CORE"
else
    echo ">> WARNING: taskset not found — runs will NOT be pinned to one core."
fi

# Validate each binary before the real sweep using a tiny configuration.
smoke_test() {
    local b="$1"
    local cflags="$2"
    local tag="$3"

    local march
    local tmp
    local ok

    march="$(build_march "$b")"

    echo ">> [$tag] smoke test"

    tmp="$(mktemp "$OUT_DIR/.smoke_XXXXXX")"

    if ! $RUNNER "./bench/bench_$tag" \
        --kernel scalar_f32 \
        --gen 64 64 0.05 1 \
        --runs 2 \
        --warmup 1 \
        --label smoke \
        --arm baseline \
        --build "$b" \
        --march "$march" \
        --cflags "$cflags" \
        --cc-version "$CC_VERSION" \
        > "$tmp" 2>&1; then

        echo "---- smoke output ----"
        cat "$tmp"
        echo "----------------------"

        rm -f "$tmp"

        die "[$tag] smoke test failed to run"
    fi

    ok="$(
        awk -F, \
            -v nf="$CSV_NFIELDS" \
            -v b="$b" \
            -v cf="$cflags" \
            -v fb="$F_BUILD" \
            -v fk="$F_KERNEL" \
            -v fc="$F_CORRECT" \
            -v fcf="$F_CFLAGS" '
            NF == nf &&
            $fk == "scalar_f32" &&
            $fb == b &&
            $fcf == cf &&
            $fc == 1
            ' "$tmp" |
        wc -l |
        tr -d ' '
    )"

    if [ "$ok" -ne 2 ]; then
        echo "---- smoke output ----"
        cat "$tmp"
        echo "----------------------"

        rm -f "$tmp"

        die "[$tag] smoke test produced $ok/2 valid+correct rows"
    fi

    rm -f "$tmp"

    echo "   [$tag] smoke test passed"
}

for ((i = 0; i < ${#VARIANT_TAG[@]}; i++)); do
    build_all \
        "${VARIANT_BUILD[i]}" \
        "${VARIANT_CFLAGS[i]}" \
        "${VARIANT_TAG[i]}"

    smoke_test \
        "${VARIANT_BUILD[i]}" \
        "${VARIANT_CFLAGS[i]}" \
        "${VARIANT_TAG[i]}"
done

set_governor_performance

# ============================================================================
# CSV SETUP
# ============================================================================

if [ ! -f "$CSV" ]; then
    echo "$CSV_HEADER" > "$CSV" ||
        die "cannot write $CSV"

    echo ">> created $CSV"
else
    existing_header="$(head -1 "$CSV")"

    if [ "$existing_header" != "$CSV_HEADER" ]; then
        die "existing $CSV has a different header than expected."
    fi

    echo ">> appending to existing $CSV ($(( $(wc -l < "$CSV") - 1 )) rows)"

    # Groups are complete only when they contain exactly RUNS rows. A changed
    # RUNS value therefore invalidates existing partial groups.
    stale_groups="$(
        awk -F, \
            -v r="$RUNS" \
            -v fb="$F_BUILD" \
            -v fc="$F_CFLAGS" \
            -v fk="$F_KERNEL" \
            -v fl="$F_LABEL" '
            NR > 1 {
                c[$fb FS $fc FS $fk FS $fl]++
            }
            END {
                n = 0
                for (g in c)
                    if (c[g] != r)
                        n++
                print n
            }
        ' "$CSV"
    )"

    if [ "${stale_groups:-0}" -gt 0 ]; then
        echo ">> NOTE: $stale_groups existing group(s) do not have exactly RUNS=$RUNS rows."
        echo "         Those will be purged and re-run."
    fi
fi

{
    echo "--- run $(date -Iseconds) ---"
    echo "host: $(hostname)"
    echo "uname: $(uname -srm)"
    echo "cc: $CC"
    echo "cc_version: $CC_VERSION"
    echo "cc_banner: $("$CC" --version 2>/dev/null | head -1)"

    for ((i = 0; i < ${#VARIANT_TAG[@]}; i++)); do
        echo "variant[${VARIANT_TAG[i]}]: $(build_march "${VARIANT_BUILD[i]}") ${VARIANT_CFLAGS[i]} -O3 $PERF_FLAG"
    done

    echo "experiments: $EXP_FILE (${#EXP_ARM[@]} rows selected${DTYPE_FILTER:+, dtype=$DTYPE_FILTER})"
    echo "csv: $CSV"
    echo "runs: $RUNS  warmup: $WARMUP  pin_core: $PIN_CORE  retries: $RETRIES"
    echo "shuffle: $SHUFFLE  shuf_seed: $SHUF_SEED"
    echo "governor_requested: performance  governor_orig: ${ORIG_GOV:-not_changed}"
    echo "governor_now: $(cat "$GOV_PATH" 2>/dev/null || echo n/a)"
    echo "core_isolated: $(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo none)"
    echo "freq_khz: $(read_freq)"
    echo "isa: $(grep -m1 isa /proc/cpuinfo 2>/dev/null || echo n/a)"
    echo "synthetic_only: $SYNTHETIC_ONLY"
    echo ""
} | tee -a "$OUT_DIR/env_log.txt" > "$OUT_DIR/env.txt"

echo ">> environment appended to $OUT_DIR/env_log.txt"

# ============================================================================
# HELPERS
#
# Resume key: (label, kernel, build, cflags). Build and cflags distinguish
# otherwise identical kernels compiled into different binaries.
# ============================================================================

atomic_append_csv() {
    local rows_file="$1"

    [ -s "$rows_file" ] || return 0

    local tmp

    tmp="$(mktemp "$OUT_DIR/.csv_XXXXXX")" ||
        die "mktemp failed"

    if cat "$CSV" "$rows_file" > "$tmp" &&
       mv -f "$tmp" "$CSV"; then
        :
    else
        rm -f "$tmp"
        die "atomic CSV write failed"
    fi
}

config_count() {
    awk -F, \
        -v l="$1" \
        -v k="$2" \
        -v b="$3" \
        -v cf="$4" \
        -v fl="$F_LABEL" \
        -v fk="$F_KERNEL" \
        -v fb="$F_BUILD" \
        -v fcf="$F_CFLAGS" '
        NR > 1 &&
        $fl == l &&
        $fk == k &&
        $fb == b &&
        $fcf == cf
        ' "$CSV" |
    wc -l |
    tr -d ' '
}

purge_config() {
    local tmp

    tmp="$(mktemp "$OUT_DIR/.csv_XXXXXX")" ||
        die "mktemp failed"

    if awk -F, \
        -v l="$1" \
        -v k="$2" \
        -v b="$3" \
        -v cf="$4" \
        -v fl="$F_LABEL" \
        -v fk="$F_KERNEL" \
        -v fb="$F_BUILD" \
        -v fcf="$F_CFLAGS" '
        NR == 1 ||
        !($fl == l && $fk == k && $fb == b && $fcf == cf)
        ' "$CSV" > "$tmp" &&
        mv -f "$tmp" "$CSV"; then
        :
    else
        rm -f "$tmp"
        die "purge failed for $1/$2/$3/$4"
    fi
}

filter_valid_rows() {
    awk -F, \
        -v l="$2" \
        -v k="$3" \
        -v b="$4" \
        -v cf="$5" \
        -v nf="$CSV_NFIELDS" \
        -v fl="$F_LABEL" \
        -v fk="$F_KERNEL" \
        -v fb="$F_BUILD" \
        -v fcf="$F_CFLAGS" \
        -v tag="$2/$3/$4/$5" '
        NF == nf &&
        $fl == l &&
        $fk == k &&
        $fb == b &&
        $fcf == cf {
            print
            next
        }
        {
            bad++
        }
        END {
            if (bad > 0)
                print "   !! dropped " bad " malformed/mismatched row(s) for " tag > "/dev/stderr"
        }
        ' "$1" > "$6"
}

# Shuffle order per matrix to reduce systematic thermal/order effects. The
# seed is fixed once so the order is reproducible.
seeded_shuffle_idx() {
    local n="$1"
    local i
    local j
    local tmp

    local -a idx=()

    for ((i = 0; i < n; i++)); do
        idx+=("$i")
    done

    for ((i = n - 1; i > 0; i--)); do
        j=$((RANDOM % (i + 1)))

        tmp="${idx[i]}"
        idx[i]="${idx[j]}"
        idx[j]="$tmp"
    done

    printf '%s\n' "${idx[@]}"
}

experiment_order() {
    local n="${#EXP_ARM[@]}"
    local i

    if [ "$SHUFFLE" = "1" ]; then
        seeded_shuffle_idx "$n"
    else
        for ((i = 0; i < n; i++)); do
            echo "$i"
        done
    fi
}

FAILED_CONFIGS=()
COMPLETED_COUNT=0
ATTEMPTED_COUNT=0

# ============================================================================
# RUN
# ============================================================================

run_config() {
    local ei="$1"
    local label="$2"

    shift 2

    local arm="${EXP_ARM[ei]}"
    local k="${EXP_KERNEL[ei]}"
    local b="${EXP_BUILD[ei]}"
    local cf="${EXP_CFLAGS[ei]}"

    local tag
    tag="$(exp_tag "$ei")"

    local march
    march="$(build_march "$b")"

    local desc="$k@$b"

    [ "$cf" != "-" ] && desc="$k@$b[$cf]"

    ATTEMPTED_COUNT=$((ATTEMPTED_COUNT + 1))

    local have
    have="$(config_count "$label" "$k" "$b" "$cf")"

    if [ "$FORCE" != "1" ] && [ "$have" -eq "$RUNS" ]; then
        echo "   $label  $desc ($arm)  [have]"
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
        return 0
    fi

    if [ "$have" -gt 0 ]; then
        echo "   $label  $desc ($arm)  [partial: $have/$RUNS present — purging and re-running clean]"
        purge_config "$label" "$k" "$b" "$cf"
    else
        echo "   $label  $desc ($arm)"
    fi

    local freq_before
    freq_before="$(read_freq)"

    local attempt=0
    local ok=0

    while [ "$attempt" -le "$RETRIES" ]; do
        attempt=$((attempt + 1))

        local tmp
        local valid
        local n
        local rc

        tmp="$(mktemp "$OUT_DIR/.run_XXXXXX")"
        valid="$(mktemp "$OUT_DIR/.valid_XXXXXX")"

        if $RUNNER "./bench/bench_$tag" \
            --kernel "$k" \
            "$@" \
            --runs "$RUNS" \
            --warmup "$WARMUP" \
            --label "$label" \
            --arm "$arm" \
            --build "$b" \
            --march "$march" \
            --cflags "$cf" \
            --cc-version "$CC_VERSION" \
            > "$tmp" \
            2>"$tmp.err"; then
            :
        else
            rc=$?

            echo "   !! failed (rc=$rc) — $label / $desc (attempt $attempt/$((RETRIES + 1))): $(head -1 "$tmp.err" 2>/dev/null)"
        fi

        filter_valid_rows \
            "$tmp" \
            "$label" \
            "$k" \
            "$b" \
            "$cf" \
            "$valid"

        n="$(wc -l < "$valid" | tr -d ' ')"

        if [ "$n" -eq "$RUNS" ]; then
            atomic_append_csv "$valid"

            rm -f "$tmp" "$tmp.err" "$valid"

            ok=1
            break
        fi

        echo "   .. attempt $attempt/$((RETRIES + 1)) produced $n/$RUNS valid rows"

        rm -f "$tmp" "$tmp.err" "$valid"
    done

    local freq_after
    freq_after="$(read_freq)"

    if [ "$freq_before" -gt 0 ] 2>/dev/null &&
       [ "$freq_after" -gt 0 ] 2>/dev/null; then

        local drift

        drift="$(
            awk \
                -v a="$freq_before" \
                -v b2="$freq_after" '
                BEGIN {
                    d = (a > b2 ? a - b2 : b2 - a)
                    printf "%.4f", (a > 0 ? d / a : 0)
                }'
        )"

        if awk \
            -v d="$drift" \
            -v w="$FREQ_DRIFT_WARN" \
            'BEGIN { exit !(d > w) }'; then

            echo "   !! FREQ DRIFT $label / $desc: ${freq_before}kHz -> ${freq_after}kHz ($(awk -v d="$drift" 'BEGIN {printf "%.1f", d*100}')%) — number may be DVFS-tainted"
        fi
    fi

    if [ "$ok" -eq 1 ]; then
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
    else
        echo "   !! GIVING UP on $label / $desc after $((RETRIES + 1)) attempt(s) — no rows written for this config"
        FAILED_CONFIGS+=("$label | $desc")
    fi
}

RANDOM="$SHUF_SEED"

echo ">> experiment order per matrix: $(
    [ "$SHUFFLE" = "1" ] &&
        echo "shuffled (seed $SHUF_SEED)" ||
        echo "fixed"
)"

# ============================================================================
# SWEEP
# ============================================================================

echo ">> synthetic sweep"

for case in "${GEN_CASES[@]}"; do
    read -r R C D S <<< "$case"

    label="gen_${R}x${C}_d${D}"

    mapfile -t ORDER < <(experiment_order)

    echo "   [$label] order: $(
        for i in "${ORDER[@]}"; do
            printf '%s@%s ' "${EXP_KERNEL[i]}" "$(exp_tag "$i")"
        done
    )"

    for i in "${ORDER[@]}"; do
        run_config "$i" "$label" \
            --gen "$R" "$C" "$D" "$S"
    done
done

if [ "$SYNTHETIC_ONLY" = "1" ]; then
    echo ">> real matrix sweep SKIPPED — no .mtx files present"
else
    echo ">> real matrix sweep (C = A*A)"

    if [ -n "$ONLY_MTX" ]; then
        MATRIX_LIST=("$ONLY_MTX")

        mtx_path "$ONLY_MTX" >/dev/null ||
            die "ONLY_MTX=$ONLY_MTX but neither matrices/$ONLY_MTX.mtx nor matrices/$ONLY_MTX/$ONLY_MTX.mtx exists"
    else
        MATRIX_LIST=("${PRESENT_MATRICES[@]}")
    fi

    for name in "${MATRIX_LIST[@]}"; do
        mtx="$(mtx_path "$name")"

        mapfile -t ORDER < <(experiment_order)

        echo "   [$name] order: $(
            for i in "${ORDER[@]}"; do
                printf '%s@%s ' "${EXP_KERNEL[i]}" "$(exp_tag "$i")"
            done
        )"

        for i in "${ORDER[@]}"; do
            run_config "$i" "$name" \
                --mtx-sq "$mtx"
        done
    done
fi

# ============================================================================
# FINAL INTEGRITY CHECK
# ============================================================================

echo ""
echo ">> integrity check: re-scanning $CSV"

BAD_GROUPS=0

while IFS= read -r grp; do
    [ -z "$grp" ] && continue

    IFS=$'\t' read -r lbl kern bld cfl <<< "$grp"

    n="$(config_count "$lbl" "$kern" "$bld" "$cfl")"

    if [ "$n" -ne "$RUNS" ]; then
        echo "   !! $lbl / $kern@$bld[$cfl] has $n/$RUNS rows (not exactly RUNS)"
        BAD_GROUPS=$((BAD_GROUPS + 1))
    fi

done < <(
    awk -F, \
        -v fl="$F_LABEL" \
        -v fk="$F_KERNEL" \
        -v fb="$F_BUILD" \
        -v fcf="$F_CFLAGS" '
        NR > 1 {
            print $fl "\t" $fk "\t" $fb "\t" $fcf
        }
        ' "$CSV" |
    sort -u
)

echo ""
echo ">> done. $CSV has $(( $(wc -l < "$CSV") - 1 )) rows"
echo ">> configs attempted this run: $ATTEMPTED_COUNT, completed: $COMPLETED_COUNT, failed: ${#FAILED_CONFIGS[@]}"

if [ "${#FAILED_CONFIGS[@]}" -gt 0 ]; then
    echo ">> FAILED configs (no rows written, will retry next run):"

    for f in "${FAILED_CONFIGS[@]}"; do
        echo "     $f"
    done
fi

if [ "$BAD_GROUPS" -gt 0 ]; then
    echo ">> WARNING: $BAD_GROUPS group(s) in the CSV do not have exactly $RUNS rows."
else
    echo ">> integrity check passed: every group has exactly $RUNS rows."
fi

echo ">> analyze with: python3 bench/analyze.py $CSV --baseline $BASELINE_KERNEL"

[ "${#FAILED_CONFIGS[@]}" -eq 0 ] || exit 1

exit 0