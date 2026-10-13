#!/usr/bin/env bash
# Model-checks the TLA+ specs in this folder with TLC. See README.md.
#
#   verification/check.sh             the code as it is: every property of every model must hold
#   verification/check.sh --original  the code before the fixes: every bug must show up again,
#                                     with its counterexample (the shortest, as TLC searches
#                                     breadth first)
#   verification/check.sh --limits    what the fixes don't cover: a worker that dies mid-job
#
# Needs `tlc` on PATH, or TLA2TOOLS=/path/to/tla2tools.jar and Java 11+, or Nix, in which case
# it runs itself inside `nix shell nixpkgs#tlaplus`. Logs go to $TLC_LOGS (default: a new
# temporary folder).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
mode=${1:-models}

if command -v tlc >/dev/null 2>&1; then
    tlc=(tlc)
elif [ -n "${TLA2TOOLS:-}" ]; then
    tlc=(java -XX:+UseParallelGC -cp "$TLA2TOOLS" tlc2.TLC)
elif command -v nix >/dev/null 2>&1; then
    exec nix --extra-experimental-features "nix-command flakes" shell nixpkgs#tlaplus \
        -c bash "$0" "$@"
else
    echo "needs tlc on PATH, TLA2TOOLS=/path/to/tla2tools.jar, or nix" >&2
    exit 2
fi

logs=${TLC_LOGS:-$(mktemp -d)}
mkdir -p "$logs"
failed=0

# run NAME SPEC CFG EXPECT: model-check SPEC.tla with CFG; EXPECT is "pass" or "violation".
run() {
    local name=$1 spec=$2 cfg=$3 expect=$4 dir log
    dir=$(mktemp -d)
    log="$logs/$name.log"
    cp "$here/$spec.tla" "$dir/"
    cp "$cfg" "$dir/$spec.cfg"
    (cd "$dir" && "${tlc[@]}" -workers auto -deadlock -config "$spec.cfg" "$spec.tla") >"$log" 2>&1 || true
    rm -rf "$dir"
    local states
    states=$(grep -oE '[0-9]+ distinct states found' "$log" | tail -1 | cut -d' ' -f1 || true)
    if grep -q "No error has been found" "$log"; then
        if [ "$expect" = pass ]; then
            echo "ok        $name ($states states)"
        else
            echo "MISSING   $name: expected a violation, TLC found none (log: $log)"
            failed=1
        fi
    elif grep -qE "is violated|Temporal properties were violated" "$log"; then
        if [ "$expect" = violation ]; then
            echo "violated  $name, as expected:"
            grep -E '^State [0-9]+: ' "$log" |
                sed -E 's/^State ([0-9]+): <([A-Za-z]+).*/\2/; s/^State ([0-9]+): (.*)/\2/' |
                grep -vx "Initial" | paste -sd' ' | fold -s -w 96 | sed 's/^/            /' || true
        else
            echo "VIOLATED  $name (log: $log)"
            grep -E "is violated|Temporal properties were violated" "$log" | sed 's/^/          /'
            failed=1
        fi
    else
        echo "ERROR     $name (log: $log)"
        tail -5 "$log" | sed 's/^/          /'
        failed=1
    fi
}

# variant SPEC PROPERTY KIND SEDSCRIPT: SPEC.cfg with its constants edited by SEDSCRIPT and
# only PROPERTY checked (KIND: INVARIANTS or PROPERTIES).
variant() {
    local spec=$1 prop=$2 kind=$3 edit=$4 cfg
    cfg=$(mktemp)
    { sed -e '/^INVARIANTS/,$d' -e "$edit" "$here/$spec.cfg"; printf '%s\n    %s\n' "$kind" "$prop"; } >"$cfg"
    echo "$cfg"
}

models=(JobLifecycle InputLog Serving AdminConfig)

case "$mode" in
models)
    for m in "${models[@]}"; do
        run "$m" "$m" "$here/$m.cfg" pass
    done
    ;;
--original)
    # SPEC PROPERTY KIND: what each fix repaired.
    while read -r spec prop kind; do
        run "$spec-before-$prop" "$spec" \
            "$(variant "$spec" "$prop" "$kind" 's/Fixed = TRUE/Fixed = FALSE/')" violation
    done <<'EOF'
JobLifecycle CancelledNeverDeploys INVARIANTS
JobLifecycle ServedIsDone INVARIANTS
JobLifecycle FinishedJobsStay PROPERTIES
JobLifecycle MemoryMatchesDb INVARIANTS
JobLifecycle StatusHasJob INVARIANTS
JobLifecycle ThresholdOnlyWhileCollecting INVARIANTS
JobLifecycle ClaimedJobsFinish PROPERTIES
InputLog ClaimSeesEarlierInputs INVARIANTS
InputLog NoLossOnShutdown INVARIANTS
Serving LoadFailsOnlyOnItsOwnError INVARIANTS
AdminConfig CacheMatchesDb INVARIANTS
AdminConfig OverridesMatchDb INVARIANTS
EOF
    ;;
--limits)
    # No lease: a job whose worker dies stays running until an admin cancels it.
    run JobLifecycle-worker-dies JobLifecycle \
        "$(variant JobLifecycle ClaimedJobsFinish PROPERTIES 's/WorkerCrash = FALSE/WorkerCrash = TRUE/')" violation
    ;;
*)
    echo "usage: $0 [--original | --limits]" >&2
    exit 2
    ;;
esac

echo "logs: $logs"
exit $failed
