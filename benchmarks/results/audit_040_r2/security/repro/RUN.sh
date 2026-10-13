#!/usr/bin/env bash
# Reproducers for benchmarks/results/audit_040_r2/security/REPORT.md
# All sockets are on 127.0.0.1.  Nothing writes into the repo or the venv.
set -u
WT=/home/nick/MSF/msf/MADDENING-wt/audit-r2-security   # detached at 0219b82
PY=/home/nick/MSF/msf/.venv/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"

export PYTHONPATH="$WT/src" JAX_PLATFORMS=cpu

for f in r01_wiretap_differential r02_bearer_token_yields_curve_keys \
         r03_receiver_silent_on_curve_mismatch r04_cloudsession_health_probe_401 \
         r05_zap_untested_on_two_of_three_servers r06_token_in_process_listing \
         r07_default_binds_over_real_sockets r08_key_derivation_and_weak_tokens; do
  echo "=============== $f"
  ( cd "$WT" && timeout 300 "$PY" "$HERE/$f.py" 2>&1 | grep -v '^INFO\|^DEBUG' )
done

# Mutation scoreboard (mutation_results.md).  Each run applies one seeded
# fault, runs a test selection and reverts with `git checkout --`.
echo "=============== mutations"
for m in peer_rule_and_not_or routable_peer_always_false exempt_add_graph verify_always_true; do
  ( cd "$WT" && timeout 600 "$PY" "$HERE/mutate_harness.py" "$m" tests/api/test_bearer_auth.py )
done
for m in zap_allow_any no_authenticator no_authenticator_cmdpub no_authenticator_coord \
         role_not_separated person_changed secure_none_returns_false; do
  ( cd "$WT" && timeout 900 "$PY" "$HERE/mutate_harness.py" "$m" tests/security/test_zmq_transport_auth.py )
done
( cd "$WT" && timeout 300 "$PY" "$HERE/mutate_harness.py" skypilot_hardcode_ports tests/cloud/test_skypilot_ports.py )
( cd "$WT" && timeout 300 "$PY" "$HERE/mutate_harness.py" signaling_accepts_no_token tests/cloud/test_signaling_auth.py )
