#!/usr/bin/env bash
#
# deploy.sh — put labelcheck on labelcheck.apps.smolkin.org, or show exactly what that
# would do without touching the box.
#
#   deploy/deploy.sh --dry-run      local preflight only; prints every remote command in
#                                   order and connects to nothing
#   deploy/deploy.sh                the real thing
#   deploy/deploy.sh --force        deploy even if the on-box benchmark misses the budget
#
# The target box is shared with dozens of other sites, so three habits are enforced here
# rather than remembered:
#
#   * the TLS certificate paths are READ from `nginx -T` on the box, never guessed;
#   * `nginx -t` gates every nginx change and nginx is RELOADED, never restarted;
#   * the benchmark runs before the unit is enabled, so a slow box cannot end up serving
#     an over-budget demo.
#
# Every path this script creates is listed in the decommission runbook kept with the
# project's working notes. If you add a path here, add a row there.

set -euo pipefail

# --- configuration (override any of these from the environment) --------------------------
SSH_HOST="${LABELCHECK_SSH_HOST:-167.233.238.194}"
SSH_USER="${LABELCHECK_SSH_USER:-root}"
SSH_KEY="${LABELCHECK_SSH_KEY:-$HOME/.ssh/bookmap-hetzner}"
DOMAIN="${LABELCHECK_DOMAIN:-labelcheck.apps.smolkin.org}"
APP_DIR="${LABELCHECK_APP_DIR:-/opt/labelcheck}"
ENV_FILE="${LABELCHECK_ENV_FILE:-/etc/labelcheck.env}"
PORT="${LABELCHECK_PORT:-8712}"
BUDGET_MS="${LABELCHECK_BUDGET_MS:-5000}"
# Normally discovered from the box (see step 5). Set both to skip discovery entirely.
FULLCHAIN="${LABELCHECK_FULLCHAIN:-}"
PRIVKEY="${LABELCHECK_PRIVKEY:-}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="https://$DOMAIN"
SSH_TARGET="$SSH_USER@$SSH_HOST"
SSH_ARGS=(-i "$SSH_KEY" -o ConnectTimeout=10 -o BatchMode=yes)
NGINX_AVAILABLE="/etc/nginx/sites-available/labelcheck.conf"
NGINX_ENABLED="/etc/nginx/sites-enabled/labelcheck.conf"

DRY_RUN=0
FORCE=0
SKIPS=""

# --- plumbing ----------------------------------------------------------------------------
step() { printf '\n=== %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '    WARNING: %s\n' "$*" >&2; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

skip() {  # a preflight step that could not run; fatal for a real deploy, noted for a dry run
    SKIPS="$SKIPS
      - $*"
    if [ "$DRY_RUN" -eq 1 ]; then
        warn "skipped: $*"
    else
        die "$* — cannot deploy for real without it (try --dry-run)"
    fi
}

have() { command -v "$1" >/dev/null 2>&1; }

run() {  # print, then run locally unless this is a dry run
    printf '    $ %s\n' "$*"
    if [ "$DRY_RUN" -eq 1 ]; then return 0; fi
    "$@"
}

# The two functions below expand their argument locally on purpose: that is what makes the
# printed line and the executed line the same string, which is the whole point of --dry-run.
remote() {  # print, then run on the box unless this is a dry run
    printf '    ssh %s $ %s\n' "$SSH_TARGET" "$*"
    if [ "$DRY_RUN" -eq 1 ]; then return 0; fi
    # shellcheck disable=SC2029  # deliberate: see the note above
    ssh "${SSH_ARGS[@]}" "$SSH_TARGET" "$*"
}

remote_capture() {  # same, but stdout is the box's output (the banner goes to stderr)
    printf '    ssh %s $ %s\n' "$SSH_TARGET" "$*" >&2
    if [ "$DRY_RUN" -eq 1 ]; then return 0; fi
    # shellcheck disable=SC2029  # deliberate: see the note above
    ssh "${SSH_ARGS[@]}" "$SSH_TARGET" "$*"
}

print_rollback() {
    cat <<EOF

--- rollback (leaves the box exactly as it was, hostname falls back to the static vhost) ---
ssh -i $SSH_KEY $SSH_TARGET '
  systemctl disable --now labelcheck.service labelcheck-purge.timer
  rm -f $NGINX_ENABLED
  nginx -t && systemctl reload nginx'
-------------------------------------------------------------------------------------------
EOF
}

# Any non-zero exit — a failed remote command, a failed preflight, an interrupt — prints the
# rollback. A half-configured shared box must never be left without the undo in front of you.
on_exit() {
    rc=$?
    if [ "$rc" -ne 0 ]; then
        printf '\nDEPLOY FAILED (exit %s). The box may be half-configured — roll back:\n' "$rc" >&2
        print_rollback >&2
    fi
}
trap on_exit EXIT

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --force)   FORCE=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *)         die "unknown argument: $1 (see --help)" ;;
    esac
done

cd "$REPO_ROOT"
if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY RUN — nothing is executed on %s. Every remote command is printed.\n' "$SSH_HOST"
fi

# --- 1. local preflight -------------------------------------------------------------------
step "1/7  local preflight"

PY_LOCAL="$REPO_ROOT/.venv/bin/python"
[ -x "$PY_LOCAL" ] || PY_LOCAL="$(command -v python3 || true)"
[ -n "$PY_LOCAL" ] || die "no python3 on PATH — needed to read JSON in the verification step"

RUFF="$REPO_ROOT/.venv/bin/ruff"
[ -x "$RUFF" ] || RUFF="$(command -v ruff || true)"
if [ -n "$RUFF" ] && [ -d "$REPO_ROOT/app" ]; then
    run "$RUFF" check .
    run "$RUFF" format --check .
else
    skip "ruff lint/format check (ruff or app/ not present)"
fi

PYTEST="$REPO_ROOT/.venv/bin/pytest"
[ -x "$PYTEST" ] || PYTEST="$(command -v pytest || true)"
if [ -n "$PYTEST" ] && [ -d "$REPO_ROOT/tests" ]; then
    run "$PYTEST" -q
else
    skip "test suite (pytest or tests/ not present)"
fi

if [ -f "$REPO_ROOT/package.json" ] && have npm && grep -q '"build"' "$REPO_ROOT/package.json"; then
    run npm run build
else
    skip "frontend build (package.json with a build script, or npm, not present)"
fi

# The bundle must not reference an outside origin: that is the offline claim, checked
# mechanically rather than trusted.
if [ -f "$REPO_ROOT/static/dist/app.js" ]; then
    if grep -Eq 'https?://' "$REPO_ROOT/static/dist/app.js"; then
        die "static/dist/app.js contains an http(s) URL — the frontend must fetch nothing external"
    fi
    note "static/dist/app.js present and free of external URLs"
else
    skip "static/dist/app.js check (bundle not built)"
fi

# Hash-pinned requirements get --require-hashes; an unpinned file is installed plainly so a
# work-in-progress tree still deploys, with the difference stated rather than silent.
# --no-deps always: rapidocr hard-declares opencv-python (GUI build), which must never
# install — requirements.txt is the complete pinned closure. And never --only-binary:
# antlr4-python3-runtime ships sdist-only, so a wheels-only install cannot resolve.
PIP_ARGS="--no-deps -r requirements.txt"
if [ -f "$REPO_ROOT/requirements.txt" ]; then
    if grep -q -- '--hash=' "$REPO_ROOT/requirements.txt"; then
        PIP_ARGS="--no-deps --require-hashes -r requirements.txt"
        note "requirements.txt is hash-pinned — installing with --require-hashes"
    else
        note "requirements.txt has no hashes — installing without --require-hashes"
    fi
    if grep -Eq '^opencv-python[=<>]' "$REPO_ROOT/requirements.txt"; then
        die "requirements.txt pins opencv-python; only opencv-python-headless may be installed"
    fi
else
    skip "requirements.txt (not present)"
fi

GOLDEN="$REPO_ROOT/tests/golden.json"
[ -f "$GOLDEN" ] || skip "tests/golden.json (post-deploy verdict check will be limited)"

# --- 2. box preflight and sync -------------------------------------------------------------
step "2/7  box preflight and sync"

remote "test -x /usr/sbin/nginx || command -v nginx >/dev/null"
remote "command -v python3 >/dev/null && python3 -c 'import venv'"
# Port 8712 must be free, unless it is already ours from a previous deploy.
remote "if ss -ltn | grep -q ':$PORT '; then systemctl is-active --quiet labelcheck || { echo 'port $PORT is in use by something that is not labelcheck'; exit 1; }; fi"
remote "install -d -m 0755 $APP_DIR"

run rsync -az --delete \
    --exclude venv --exclude .venv --exclude __pycache__ --exclude '*.pyc' \
    --exclude .git --exclude .data --exclude .pytest_cache --exclude node_modules \
    --exclude __fixtures__ \
    -e "ssh -i $SSH_KEY -o BatchMode=yes" \
    app static deploy requirements.txt "$SSH_TARGET:$APP_DIR/"
remote "chmod 0755 $APP_DIR/deploy/*.sh"

# --- 3. virtualenv on the box ----------------------------------------------------------------
step "3/7  virtualenv"

remote "test -x $APP_DIR/venv/bin/python || python3 -m venv $APP_DIR/venv"
remote "cd $APP_DIR && ./venv/bin/pip install --disable-pip-version-check $PIP_ARGS"
remote "cd $APP_DIR && ./venv/bin/python -c 'import rapidocr, onnxruntime, cv2, fastapi; print(\"imports ok\")'"

# --- 4. benchmark before the unit is ever enabled ---------------------------------------------
step "4/7  benchmark on the box (chooses the resolution knobs, writes $ENV_FILE)"

BENCH_CMD="$APP_DIR/deploy/bench-on-box.sh --app-dir $APP_DIR --env-out $ENV_FILE --budget-ms $BUDGET_MS"
if [ "$FORCE" -eq 1 ]; then
    BENCH_CMD="$BENCH_CMD --force"
    note "--force: a p95 over budget will not stop the deploy"
fi
if ! remote "$BENCH_CMD"; then
    die "the box benchmark missed the latency target; $ENV_FILE was not written and the unit
       was not enabled. Re-run with --force to deploy anyway, or lower the resolution knobs."
fi

# --- 5. install the unit, the timer and the vhost ------------------------------------------------
step "5/7  install units and nginx vhost"

# Certificate paths are discovered, not assumed, and discovery asks the only question that
# matters: does this certificate actually cover $DOMAIN? Matching on the path would be
# wrong on this box — /etc/letsencrypt/live/apps.smolkin.org/ covers the bare hostname only,
# while the wildcard that covers labelcheck.apps.smolkin.org lives under a differently named
# directory. Ambiguity aborts rather than picking.
CERT_CMD="for cert in \$(nginx -T 2>/dev/null | tr -d ';' |
        sed -n 's/^[[:space:]]*ssl_certificate[[:space:]][[:space:]]*//p' | sort -u); do
    openssl x509 -in \"\$cert\" -noout -checkhost $DOMAIN 2>/dev/null | grep -q ' does match ' || continue
    key=\"\$(dirname \"\$cert\")/privkey.pem\"
    [ -f \"\$key\" ] && echo \"\$cert \$key\"
done"

if [ -n "$FULLCHAIN" ] && [ -n "$PRIVKEY" ]; then
    note "certificate paths given on the command line, skipping discovery"
elif [ "$DRY_RUN" -eq 1 ]; then
    remote_capture "$CERT_CMD" >/dev/null
    FULLCHAIN="@@discovered-at-deploy@@"
    PRIVKEY="@@discovered-at-deploy@@"
    note "(dry run) the certificate covering $DOMAIN would be selected from that output"
else
    CERT_PAIR="$(remote_capture "$CERT_CMD" | sort -u)"
    [ "$(printf '%s\n' "$CERT_PAIR" | grep -c .)" = "1" ] ||
        die "expected exactly one certificate covering $DOMAIN, found:
$CERT_PAIR
       Pass LABELCHECK_FULLCHAIN and LABELCHECK_PRIVKEY to choose explicitly."
    FULLCHAIN="${CERT_PAIR% *}"
    PRIVKEY="${CERT_PAIR#* }"
    note "certificate: $FULLCHAIN"
    note "key:         $PRIVKEY"
fi

remote "install -m 0644 $APP_DIR/deploy/labelcheck.service /etc/systemd/system/labelcheck.service"
remote "install -m 0644 $APP_DIR/deploy/labelcheck-purge.service /etc/systemd/system/labelcheck-purge.service"
remote "install -m 0644 $APP_DIR/deploy/labelcheck-purge.timer /etc/systemd/system/labelcheck-purge.timer"
remote "sed -e 's|@@FULLCHAIN@@|$FULLCHAIN|' -e 's|@@PRIVKEY@@|$PRIVKEY|' $APP_DIR/deploy/nginx-labelcheck.conf > $NGINX_AVAILABLE"
remote "ln -sfn $NGINX_AVAILABLE $NGINX_ENABLED"
# If the config is bad, unlink it again before anything reloads: the other sites on this
# box must never be one `nginx -t` away from an outage.
remote "nginx -t || { rm -f $NGINX_ENABLED; echo 'nginx -t failed; vhost unlinked'; exit 1; }"
remote "systemctl daemon-reload"
remote "systemctl enable --now labelcheck.service labelcheck-purge.timer"
remote "systemctl reload nginx"

# Hardening is verified, not assumed: the kernel-level outbound block, and the containment
# property that the app's transient user has no read path to the box's secrets. Both are
# differential tests — the unsandboxed control has to succeed first, or "it failed" would
# prove nothing (a missing curl would look exactly like a blocked one).
remote "systemd-analyze security labelcheck.service --no-pager | tail -5"
remote "systemctl show labelcheck.service -p IPAddressDeny -p DynamicUser"
remote "curl -sS --max-time 5 -o /dev/null https://example.com && echo 'control: the box itself has outbound network'"
remote "systemd-run --pipe --wait --property=IPAddressDeny=any --property=IPAddressAllow=localhost \
    curl -sS --max-time 5 -o /dev/null https://example.com >/dev/null 2>&1 && \
    { echo 'FAIL: outbound network reachable under the unit sandbox'; exit 1; } || \
    echo 'outbound network blocked under the unit sandbox, as designed'"
remote "ls /root >/dev/null && echo 'control: root can list /root'"
remote "systemd-run --pipe --wait --property=DynamicUser=yes --property=ProtectHome=yes --property=ProtectSystem=strict \
    ls /root >/dev/null 2>&1 && \
    { echo 'FAIL: the sandboxed user can read /root'; exit 1; } || \
    echo '/root unreadable by the sandboxed user, as designed'"

# --- 6. verify what was actually deployed ----------------------------------------------------------
step "6/7  verification"

if [ "$DRY_RUN" -eq 1 ]; then
    note "(dry run) would GET $URL/healthz until engine_ready"
    note "(dry run) would POST $URL/api/demo/sample-batch {\"size\":22} and poll results"
    note "(dry run) would check every verdict against tests/golden.json and p95 < $BUDGET_MS ms"
else
    # The engine loads its models at startup (~1 s), so give /healthz a moment to say ready
    # rather than declaring victory on the first 200.
    HEALTH=""
    READY=0
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        HEALTH="$(curl -sf --max-time 10 "$URL/healthz" || true)"
        case "$HEALTH" in
            *'"engine_ready":true'*|*'"engine_ready": true'*) READY=1; break ;;
        esac
        sleep 3
    done
    [ "$READY" -eq 1 ] || die "$URL/healthz never reported engine_ready. Last response: ${HEALTH:-<none>}"
    note "healthz: $HEALTH"

    BATCH_ID="$(curl -sf --max-time 60 -X POST "$URL/api/demo/sample-batch" \
        -H 'Content-Type: application/json' -d '{"size":22}' |
        "$PY_LOCAL" -c 'import json,sys; print(json.load(sys.stdin).get("batch_id",""))')"
    [ -n "$BATCH_ID" ] || die "the demo batch was not created"
    note "demo batch $BATCH_ID — polling"

    POLLS="$(mktemp)"
    CURSOR=0
    WAITED=0
    BATCH_DONE=""
    while [ "$WAITED" -lt 900 ]; do
        BODY="$(curl -sf --max-time 30 "$URL/api/batches/$BATCH_ID/results?since=$CURSOR&limit=100")"
        printf '%s\n' "$BODY" >>"$POLLS"
        CURSOR="$(printf '%s' "$BODY" | "$PY_LOCAL" -c 'import json,sys; print(json.load(sys.stdin).get("cursor",0))')"
        BATCH_DONE="$(printf '%s' "$BODY" | "$PY_LOCAL" -c 'import json,sys; print(json.load(sys.stdin).get("done",False))')"
        case "$BATCH_DONE" in True|true|1) break ;; esac
        sleep 2
        WAITED=$((WAITED + 2))
    done
    [ "$BATCH_DONE" = "True" ] || [ "$BATCH_DONE" = "true" ] || die "the demo batch did not finish in 15 minutes"

    "$PY_LOCAL" - "$POLLS" "$GOLDEN" "$BUDGET_MS" <<'PYEOF'
import json
import sys

polls_path, golden_path, budget_ms = sys.argv[1], sys.argv[2], float(sys.argv[3])


def stem(name):
    return name.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()


rows = {}
for line in open(polls_path):
    line = line.strip()
    if line:
        for row in json.loads(line).get("results", []):
            rows[row.get("image_id") or row.get("filename")] = row

counts, times, problems = {}, [], []
for row in rows.values():
    verdict = (row.get("overall") or row.get("status") or "").lower()
    counts[verdict] = counts.get(verdict, 0) + 1
    if isinstance(row.get("processing_ms"), (int, float)):
        times.append(float(row["processing_ms"]))

print("    verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
if times:
    times.sort()
    p95 = times[max(0, -(-len(times) * 95 // 100) - 1)]
    print(
        f"    timing: median {times[len(times) // 2]:.0f} ms, p95 {p95:.0f} ms, "
        f"max {times[-1]:.0f} ms over {len(times)} images"
    )
    if p95 > budget_ms:
        problems.append(f"p95 {p95:.0f} ms exceeds the {budget_ms:.0f} ms budget")
else:
    problems.append("no processing times came back")

try:
    golden = json.load(open(golden_path))
except OSError:
    golden = {}
    print("    tests/golden.json absent — checked timings and errors only")
if not isinstance(golden, dict):
    sys.exit(f"    tests/golden.json is not an object keyed by filename: {type(golden).__name__}")
for _key in ("images", "labels"):  # tolerate the expectations being nested
    if isinstance(golden.get(_key), dict):
        golden = golden[_key]
        break
expected = {stem(k): v for k, v in golden.items() if not k.startswith("_") and isinstance(v, dict)}

seen = set()
for row in rows.values():
    key = stem(row.get("filename", ""))
    seen.add(key)
    rule = expected.get(key)
    verdict = (row.get("overall") or row.get("status") or "").lower()
    if not rule:
        continue
    allowed = [s.lower() for s in rule.get("allowed", [])]
    forbidden = [s.lower() for s in rule.get("forbidden", [])]
    if allowed and verdict not in allowed:
        problems.append(f"{key} came back {verdict}, allowed {allowed}")
    if verdict in forbidden:
        problems.append(f"{key} came back {verdict}, which is forbidden")

for missing in sorted(set(expected) - seen):
    problems.append(f"{missing} is in the golden set but produced no result")

if problems:
    print("\n    VERIFICATION FAILED:")
    for problem in problems:
        print("      - " + problem)
    sys.exit(1)
print("    every verdict is inside its golden expectation")
PYEOF
    rm -f "$POLLS"
fi

# --- 7. done ----------------------------------------------------------------------------------------
step "7/7  done"

if [ "$DRY_RUN" -eq 1 ]; then
    printf '    DRY RUN complete — nothing was changed on %s.\n' "$SSH_HOST"
    if [ -n "$SKIPS" ]; then
        printf '    Preflight steps skipped because their inputs do not exist yet:%s\n' "$SKIPS"
    fi
else
    printf '    labelcheck is live at %s\n' "$URL"
fi
print_rollback
