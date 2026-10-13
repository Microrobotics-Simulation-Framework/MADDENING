"""Apply one seeded mutation, run a test selection, revert. Audit-only."""
import subprocess, sys, os
WT = "/home/nick/MSF/msf/MADDENING-wt/audit-r2-security"
VENV = "/home/nick/MSF/msf/.venv/bin/python"

MUTATIONS = {
 "http_route_no_auth": ("src/maddening/api/server.py",
   '        @app.get("/healthz", tags=["meta"])',
   '        @app.get("/audit/leak", tags=["meta"])\n'
   '        def audit_leak():\n'
   '            return {"state": "leaked"}\n\n'
   '        @app.get("/healthz", tags=["meta"])'),
 "peer_rule_and_not_or": ("src/maddening/api/auth.py",
   "        return self.enforced or is_routable_peer(peer_host)",
   "        return self.enforced and is_routable_peer(peer_host)"),
 "routable_peer_always_false": ("src/maddening/api/auth.py",
   "    return not is_loopback(normalised)",
   "    return False  # MUTANT"),
 "exempt_add_graph": ("src/maddening/api/auth.py",
   '    "/viz/auth.js",\n})',
   '    "/viz/auth.js",\n    "/graph",\n})'),
 "verify_always_true": ("src/maddening/api/auth.py",
   "        if not presented:\n            return False",
   "        return True  # MUTANT\n        if not presented:\n            return False"),
 "zap_allow_any": ("src/maddening/transport_auth.py",
   "        presented = key.encode(\"ascii\") if isinstance(key, str) else key",
   "        return True  # MUTANT\n        presented = key.encode(\"ascii\") if isinstance(key, str) else key"),
 "no_authenticator": ("src/maddening/viz/network.py",
   "            self._authenticator = auth.start_authenticator(self._context)\n            auth.secure_server(self._socket)\n        self._socket.bind(address)\n        self._step_count = 0",
   "            auth.secure_server(self._socket)\n        self._socket.bind(address)\n        self._step_count = 0"),
 "role_not_separated": ("src/maddening/transport_auth.py",
   '            role.encode("ascii") + b"\\x00" + self.token.encode("utf-8"),',
   '            self.token.encode("utf-8"),  # MUTANT: no role separation'),
 "person_changed": ("src/maddening/transport_auth.py",
   '_CURVE_PERSON = b"maddening-curve"',
   '_CURVE_PERSON = b"maddening-CURVE"  # MUTANT'),
 "secure_none_returns_false": ("src/maddening/transport_auth.py",
   "    if secure is None:\n        return address_requires_security(address)",
   "    if secure is None:\n        return False  # MUTANT"),
 "skypilot_hardcode_ports": ("src/maddening/cloud/_skypilot.py",
   'run=f"docker run --gpus all {_port_flags(ports)}"',
   'run=f"docker run --gpus all -p 5555:5555 -p 5556:5556 {_port_flags(ports)}"'),
 "signaling_accepts_no_token": ("src/maddening/cloud/selkies_session.py",
   "        if not validate_session_token(session_id, token, secret):",
   "        if False and not validate_session_token(session_id, token, secret):"),

 "no_authenticator_cmdpub": ("src/maddening/viz/network.py",
   "            self._authenticator = auth.start_authenticator(self._context)\n            auth.secure_server(self._socket)\n        self._socket.bind(address)\n\n    @property\n    def secure(self) -> bool:\n        \"\"\"Whether this socket is encrypted and authenticated with CURVE.\"\"\"\n        return self._secure\n\n    def send(",
   "            auth.secure_server(self._socket)\n        self._socket.bind(address)\n\n    @property\n    def secure(self) -> bool:\n        \"\"\"Whether this socket is encrypted and authenticated with CURVE.\"\"\"\n        return self._secure\n\n    def send("),
 "no_authenticator_coord": ("src/maddening/cloud/multigpu/coordinator.py",
   "            authenticator = auth.start_authenticator(ctx)\n            auth.secure_server(sock)",
   "            auth.secure_server(sock)"),
}

def run(name, tests):
    path, old, new = MUTATIONS[name]
    full = os.path.join(WT, path)
    src = open(full).read()
    assert src.count(old) == 1, f"anchor count {src.count(old)} for {name}"
    open(full, "w").write(src.replace(old, new, 1))
    try:
        env = dict(os.environ, PYTHONPATH=WT+"/src", JAX_PLATFORMS="cpu",
                   PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        p = subprocess.run([VENV, "-m", "pytest", *tests, "-q",
                            "-p", "no:cacheprovider", "--no-header", "-rf"],
                           cwd=WT, env=env, capture_output=True, text=True, timeout=900)
        out = (p.stdout + p.stderr)
        print(f"=== {name} -> rc={p.returncode}")
        print("\n".join(out.strip().splitlines()[-18:]))
    finally:
        subprocess.run(["git", "checkout", "--", path], cwd=WT, check=True)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:])
