#!/usr/bin/env bash
# Matched A/B harness for the experiment switches at the top of pinning.cu.
#
# The switch block says "ab.sh overrides them with -D on the build line", but no
# such script was ever committed, so each rebuild-and-compare had to be redone by
# hand. This is that script.
#
#   ./candidates/pinning/ab.sh                          # baseline + default grid
#   ./candidates/pinning/ab.sh "QSB_PREFETCH=1" "QSB_UNROLL=13 QSB_PK_UNROLL=1"
#   QSB_AB_SECONDS=300 ./candidates/pinning/ab.sh       # longer per variant
#   QSB_AB_DRYRUN=1 ./candidates/pinning/ab.sh          # print plan, build nothing
#
# Each argument is one variant: a space-separated list of NAME=VALUE switches,
# passed as -DNAME=VALUE. The empty variant is the committed baseline, and it is
# always measured first so later rows have something to be compared against.
#
# Rules this script follows so a number means something:
#   * every variant grinds the SAME problem seed, at the same N, for the same
#     wall-clock window (the research log calls this a matched measurement);
#   * the kernel is built BEFORE the timed window and handed the build stamp
#     gpu_wrap.py expects, so no nvcc invocation lands inside the measurement;
#   * the score comes from the harness's own clock and its independently
#     verified hits, never from the kernel's advisory M/s line.
#
# A short run ranks variants; it does not promote one. Confirm a winner with a
# full 1200-second run on a fresh seed before editing the defaults in pinning.cu.
#
# Note: QSB_PROBE_MASK deliberately computes wrong math as a speed probe. Runs
# using it fail hit verification and are scored as failures here, which is
# correct -- it is a profiling aid, not a candidate setting.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

src="candidates/pinning/pinning.cu"
bin="candidates/pinning/pinning"
stamp="candidates/pinning/.pinning.build"

seconds="${QSB_AB_SECONDS:-180}"
seed="${QSB_AB_SEED:-424242}"
zeros="${QSB_AB_ZEROS:-$(python3 -c "import json;print(json.load(open('harness/config.json'))['leading_zero_bits'])")}"
outroot="${QSB_AB_OUTDIR:-benchmark-results/ab}"
dryrun="${QSB_AB_DRYRUN:-0}"

# Default grid: single-switch deltas against the baseline, each one a knob whose
# effect the switch comments describe as unmeasured or hardware-dependent.
if (($# > 0)); then
  variants=("$@")
else
  variants=(
    ""                                  # committed defaults
    "QSB_PREFETCH=1"                    # next table chunk one step ahead
    "QSB_EARLY_LOAD=1"                  # load next record once cx/cy die
    "QSB_UNROLL=13"                     # fully unroll the chain loop
    "QSB_PK_UNROLL=1"                   # interleave the two recid SHA chains
    "QSB_TREE_N=128"                    # narrower product tree (adjusts block sizes)
    "QSB_BATCH=8388608"                 # half-size launches
    "QSB_STREAM=1"                      # evict-first hints on pipeline traffic
  )
fi

if [[ "${dryrun}" != "1" ]]; then
  command -v nvcc >/dev/null 2>&1 || {
    echo "ab.sh: nvcc not found — this needs a CUDA host (use QSB_AB_DRYRUN=1 to check the plan)" >&2
    exit 1
  }
fi

mkdir -p "${outroot}"
results="${outroot}/results.tsv"
printf 'variant\tscore_cand_per_s\tverified_hits\telapsed_s\n' > "${results}"

echo "ab.sh: N=${zeros} seed=${seed} window=${seconds}s variants=${#variants[@]}"
echo

for variant in "${variants[@]}"; do
  label="${variant:-baseline}"
  slug="$(echo "${label}" | tr ' =' '__' | tr -cd '[:alnum:]_-')"
  echo "=== ${label} ==="

  defines=()
  for kv in ${variant}; do defines+=("-D${kv}"); done

  build_cmd=(nvcc -O3 "-DQSB_ZEROS_N=${zeros}" "${defines[@]+"${defines[@]}"}"
             -o "${bin}" "${src}" -lcrypto -lm)
  if [[ "${dryrun}" == "1" ]]; then
    printf '  build: %s\n' "${build_cmd[*]}"
    printf '  run  : QSB_PROBLEM_SEED=%s QSB_SECONDS=%s ./benchmark.sh pinning → %s/%s\n\n' \
           "${seed}" "${seconds}" "${outroot}" "${slug}"
    continue
  fi

  "${build_cmd[@]}"
  # Tell gpu_wrap.py this binary is current for N, so it reuses it instead of
  # recompiling once the clock is running.
  python3 -c "
from pathlib import Path
src = Path('${src}')
Path('${stamp}').write_text(f'QSB_ZEROS_N=${zeros} {src.stat().st_mtime_ns}')
"

  outdir="${outroot}/${slug}"
  mkdir -p "${outdir}"
  if QSB_GRINDER="cmd:python3 harness/gpu_wrap.py --src ${src}" \
     QSB_ZEROS_N="${zeros}" QSB_MODE=fixed_time QSB_SECONDS="${seconds}" \
     QSB_PROBLEM_SEED="${seed}" QSB_MAX_REL_VAR=none \
     QSB_OUTPUT_DIR="${outdir}" \
     ./benchmark.sh pinning > "${outdir}/log.txt" 2>&1; then
    python3 - "${outdir}/score-pinning.json" "${label}" "${results}" <<'PY'
import json, sys
score_path, label, results = sys.argv[1:4]
d = json.loads(open(score_path).read())
m = d["metrics"]
open(results, "a").write(
    f"{label}\t{d['score']:.0f}\t{m['verified_hits']}\t{m['elapsed_s']:.1f}\n")
print(f"  {d['score']:.0f} candidates/s from {m['verified_hits']} verified hits")
PY
  else
    echo "  FAILED — see ${outdir}/log.txt" >&2
    printf '%s\tFAILED\t-\t-\n' "${label}" >> "${results}"
  fi
  echo
done

[[ "${dryrun}" == "1" ]] && exit 0

echo "=== results (higher is better) ==="
python3 - "${results}" <<'PY'
import sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
base = next((float(r[1]) for r in rows if r[0] == "baseline" and r[1] != "FAILED"), None)
scored = sorted((r for r in rows if r[1] != "FAILED"), key=lambda r: -float(r[1]))
failed = [r for r in rows if r[1] == "FAILED"]
width = max((len(r[0]) for r in rows), default=8)
for r in scored + failed:          # a failed variant must stay visible
    if r[1] == "FAILED":
        print(f"  {r[0]:<{width}}  FAILED")
        continue
    delta = f"{(float(r[1]) / base - 1) * 100:+6.2f}%" if base else "     -"
    print(f"  {r[0]:<{width}}  {float(r[1]):>14,.0f} cand/s  {delta}  ({r[2]} hits)")
if base:
    print("\n  Promotion needs +1.00% over the current ranked score, and a short")
    print("  run carries roughly 1/sqrt(hits) noise — treat anything under that as a tie.")
PY
