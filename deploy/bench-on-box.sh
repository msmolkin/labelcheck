#!/usr/bin/env bash
#
# bench-on-box.sh — measure labelcheck on the machine that will serve it, then write the
# resolution knobs that measurement chose to /etc/labelcheck.env.
#
# This is step 4 of the deploy: the unit is not enabled until this has run, so a slow box
# never ends up serving an over-budget demo. Run it standalone any time to re-measure:
#
#   deploy/bench-on-box.sh --app-dir /opt/labelcheck --budget-ms 5000
#
# It consumes the JSON that `python -m app.bench --json <path>` writes. The fields it needs:
#
#   p95_ms                     float, end-to-end p95 in milliseconds  (also accepted:
#                              summary.p95_ms / totals.p95_ms / e2e.p95_ms / *.p95)
#   recommended{...}           the knobs the bench chose for this box, either as settings
#                              names (pass1_long_side) or as full LABELCHECK_* names
#                              (also accepted: recommended_knobs / recommendation / env)
#
# Anything else in the JSON is printed by the bench itself and ignored here.

set -euo pipefail

APP_DIR="${APP_DIR:-/opt/labelcheck}"
ENV_OUT="${ENV_OUT:-/etc/labelcheck.env}"
JSON_OUT="${JSON_OUT:-/tmp/labelcheck-bench.json}"
BUDGET_MS="${BUDGET_MS:-5000}"
# The bench must land the p95 at or under 80% of the budget, which is the headroom the
# design reserves for a busier box than the one being measured.
TARGET_FRACTION="${TARGET_FRACTION:-0.80}"
FORCE=0

die() { echo "bench-on-box: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --app-dir)   APP_DIR="$2"; shift 2 ;;
        --env-out)   ENV_OUT="$2"; shift 2 ;;
        --json)      JSON_OUT="$2"; shift 2 ;;
        --budget-ms) BUDGET_MS="$2"; shift 2 ;;
        --force)     FORCE=1; shift ;;
        -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
        *)           die "unknown argument: $1" ;;
    esac
done

PY="$APP_DIR/venv/bin/python"
[ -x "$PY" ] || die "no interpreter at $PY — run the venv step of deploy.sh first"

echo "==> benchmarking labelcheck on $(hostname) (budget ${BUDGET_MS} ms)"
cd "$APP_DIR"
"$PY" -m app.bench --json "$JSON_OUT"

[ -s "$JSON_OUT" ] || die "app.bench wrote no JSON to $JSON_OUT"

ENV_TMP="$(mktemp)"
trap 'rm -f "$ENV_TMP"' EXIT

# Parse the bench output. stdlib only, no network, no third-party imports.
MEASURED_P95="$("$PY" - "$JSON_OUT" "$ENV_TMP" <<'PYEOF'
import json
import sys

SETTING_KNOBS = (
    "pass1_long_side",
    "pass2_long_side",
    "pass2_mode",
    "bold_long_side",
    "ocr_threads",
    "budget_ms",
)

data = json.load(open(sys.argv[1]))


def find_p95(node, depth=0):
    """First p95 the bench reports, top level preferred, then one nesting level down."""
    if not isinstance(node, dict):
        return None
    for key in ("p95_ms", "p95", "total_p95_ms"):
        value = node.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    if depth == 0:
        for key in ("summary", "totals", "total", "e2e", "overall"):
            found = find_p95(node.get(key), depth + 1)
            if found is not None:
                return found
    return None


def find_knobs(node):
    for key in ("recommended", "recommended_knobs", "recommendation", "knobs", "env"):
        value = node.get(key)
        if isinstance(value, dict) and value:
            return value
        if isinstance(value, list) and value:  # already-formatted env lines
            return dict(line.split("=", 1) for line in value if "=" in line)
    return None


p95 = find_p95(data)
knobs = find_knobs(data)
if p95 is None:
    sys.exit("bench JSON has no p95_ms field (looked at the top level, summary, totals, e2e)")
if not knobs:
    sys.exit("bench JSON has no recommended knobs (looked for recommended/recommendation/env)")

lines = []
for name, value in knobs.items():
    name = name.strip().upper()
    if not name.startswith("LABELCHECK_"):
        if name.lower() not in SETTING_KNOBS:
            continue  # not a knob we install
        name = "LABELCHECK_" + name
    lines.append(f"{name}={value}")
if not lines:
    sys.exit(f"bench recommended no knobs this script knows how to install: {knobs!r}")

with open(sys.argv[2], "w") as fh:
    fh.write("# Written by deploy/bench-on-box.sh — measured on this box, not assumed.\n")
    fh.write("# Re-run the script to re-measure; hand edits are lost on the next deploy.\n")
    fh.write("# Every other knob keeps its shipped default (deploy/labelcheck.env.example).\n")
    for line in sorted(lines):
        fh.write(line + "\n")

print(f"{p95:.1f}")
PYEOF
)"

TARGET_MS="$(awk -v b="$BUDGET_MS" -v f="$TARGET_FRACTION" 'BEGIN { printf "%.0f", b * f }')"
OVER="$(awk -v p="$MEASURED_P95" -v t="$TARGET_MS" 'BEGIN { print (p > t) ? 1 : 0 }')"

echo "==> measured p95 ${MEASURED_P95} ms against a ${TARGET_MS} ms target (${BUDGET_MS} ms budget)"
echo "==> knobs chosen for this box:"
sed 's/^/    /' "$ENV_TMP"

if [ "$OVER" = "1" ] && [ "$FORCE" != "1" ]; then
    echo "bench-on-box: p95 ${MEASURED_P95} ms exceeds the ${TARGET_MS} ms target." >&2
    echo "              $ENV_OUT was NOT written and the unit must not be enabled." >&2
    echo "              Lower LABELCHECK_PASS1_LONG_SIDE (960 -> 832 -> 736) and re-run," >&2
    echo "              or pass --force to accept a slower box deliberately." >&2
    exit 3
fi

install -m 0644 "$ENV_TMP" "$ENV_OUT"
echo "==> wrote $ENV_OUT"
if [ "$OVER" = "1" ]; then
    echo "==> WARNING: over target, installed anyway because --force was given"
fi
exit 0
