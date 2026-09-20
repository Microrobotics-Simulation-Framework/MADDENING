# Sub-report: `docs/release_notes/v0.4.0.md` — breaking changes and known anomalies

Produced by a delegated auditor at commit `0219b82`, read-only, in the same
worktree (`/home/nick/MSF/msf/MADDENING-wt/audit-r2-compliance`). Reproduced
verbatim below. What the lead auditor independently confirmed is marked at the
end; everything else is the delegate's own measurement and should be re-checked
before being quoted as evidence.

---

## F1 — The known-anomalies section lists 3 ids; 8 anomalies are open or partially resolved

`docs/release_notes/v0.4.0.md:1939-1961`. The whole `## Known anomalies` section
enumerates `MADD-ANO-006` (resolved), `MADD-ANO-010` (open) and `MADD-ANO-003`
(open). Script output:

```
Known-anomalies section: lines 1939..1961 of docs/release_notes/v0.4.0.md
ids mentioned in section: ['MADD-ANO-003', 'MADD-ANO-006', 'MADD-ANO-010']
with line numbers: {'MADD-ANO-003': 1956, 'MADD-ANO-006': 1941, 'MADD-ANO-010': 1949}

registry OPEN or PARTIALLY_RESOLVED: ['MADD-ANO-002', 'MADD-ANO-003',
  'MADD-ANO-005', 'MADD-ANO-010', 'MADD-ANO-011', 'MADD-ANO-012',
  'MADD-ANO-013', 'MADD-ANO-014'] (8)
mentioned in section                : ['MADD-ANO-003', 'MADD-ANO-006', 'MADD-ANO-010'] (3)

OMITTED from known-anomalies section: ['MADD-ANO-002', 'MADD-ANO-005',
  'MADD-ANO-011', 'MADD-ANO-012', 'MADD-ANO-013', 'MADD-ANO-014'] (6)

--- status words asserted in section ---
MADD-ANO-003: registry='open' | notes says -> ['(open', 'open,']
MADD-ANO-006: registry='resolved' | notes says -> ['resolved in 0.4.0']
MADD-ANO-010: registry='open' | notes says -> ['(open', 'open,']
```

The three statuses that are stated are correct and there is no stated count in
the section to disagree with, so this is an enumeration failure rather than a
misstatement. All six omitted anomalies are discussed correctly elsewhere in the
narrative body — 014 at 463/485, 002 at 543/549, 011/012/013 at 575/580/585,
005 at 938/2175. The brief recorded five missing; it is six.

## F2 — The breaking-changes table has 11 rows, 10 distinct, and states "nine"

`docs/release_notes/v0.4.0.md:2223` — "Beyond the floors, nine things change
behaviour in a way you may have to act on."

```
table header at line 2226; 11 data rows (lines 2228..2238)
stated count: nine

rows (first cell):
  2228: `CouplingGroup.solver` defaults to `"ift"`
  2229: `run_sweep` and `run_scan` may differ by ~1 ulp
  2230: `maddening.core.simulation.checkpoint.download_and_load_state`
  2231: `blindness_ratio` / `blindness_threshold` on `AdaptiveNode`
  2232: `bound_valid` / `gradient_error_bound` in `coupling_diagnostics()`
  2233: **A coupling group's convergence criterion is now an error *estimate* in a relative norm**
  2234: **`HeatNode` imposes its Dirichlet data at the rod ends, and refuses an unstable timestep**
  2235: **The API requires a bearer token unless it is bound to loopback**
  2236: **`JobConfig.ports` defaults to `[]`, not `[8000]`**
  2237: `validate_anomaly_registry()` requires and enum-checks `resolution_status`
  2238: **A coupling group's convergence criterion is now an error *estimate* in a relative norm**

DUPLICATED first cells:
  x2: **A coupling group's convergence criterion is now an error *estimate* in a relative norm**
       line 2233, row length 1599 chars
       line 2238, row length 1325 chars

actual rows = 11; distinct first cells = 10; stated = 'nine' (9)
```

The duplicate is not verbatim: line 2238 is a *stale* copy that silently drops
the `atol` default change. Confirmed against the code —
`git show v0.3.1:src/maddening/core/coupling/group.py:132` `atol: float = 1e-8`
vs `src/maddening/core/coupling/group.py:239` `atol: float = 0.0`. A reader who
reads the last row misses that the noise floor moved from 1e-8 to 0.0.

## F3 — "Status codes on `POST /graph/nodes` are unchanged" is false

`docs/release_notes/v0.4.0.md:1749-1752`:

> Status codes on `POST /graph/nodes` are unchanged for every input that already
> had one: a *non-finite* constructor constant -- including an integer literal
> too large to be a `float` at all -- is still the 400 "value must be finite"
> that names the parameter.

The example given is itself the counter-example: at v0.3.1 there was no
finiteness check on this endpoint at all.

```
$ git show v0.3.1:src/maddening/api/server.py | grep -n '_non_finite_param\|_state_elements\|_dry_run_node\|MAX_NODE\|isfinite\|field_validator\|422'
(no output)
```

The v0.3.1 handler (from `@app.post("/graph/nodes", ..., status_code=201)` at
line 275) is: unknown type → 400; constructor raises → 400; `gm.add_node`
ValueError → 409; otherwise 201.

Current behaviour, measured against `src/maddening/api/server.py:718-782`:

```
400  non-finite param (NaN): params.thermal_diffusivity: value must be finite
400  oversized state (dims multiply): timestep 0.01 is unstable for this rod: ...
400  param a constructor rejects: stencil_order must be 2 or 4, got 3
400  unstable timestep (new ctor check): timestep 1.0 is unstable for this rod: ...
201  plain valid node: {'status': 'ok', 'node': {'type': 'HeatNode', ...
422  int too large: [{'type': 'value_error', 'loc': ['body', 'params'], ...
```

Inputs whose status changed **201 → 400**, all 201 at v0.3.1:

1. any params leaf that is NaN/±Infinity (`server.py:731-735`);
2. a node whose `initial_state()` raises (`server.py:748-753`) — not called at
   all at v0.3.1;
3. a node whose initial state exceeds `MAX_NODE_STATE_ELEMENTS = 20_000_000`
   (`server.py:755-764`);
4. a node failing the new abstract dry-run trace `_dry_run_node`
   (`server.py:769-775`);
5. every `HeatNode` above its Fourier limit, because the new constructor
   `ValueError` (`heat.py:394-402`) is caught as a 400.

Additionally **409 → 400**: a non-duplicate `ValueError` out of `gm.add_node`
(e.g. a name containing `/`, `#`, `->`) is now 400 (`server.py:780-781`).
Only the oversized-integer 422 is correctly described as new.

## F4 — The `POST /graph/nodes` status change is not in the breaking-changes table

No row at 2228–2238 mentions REST node-creation validation. Row 2234 mentions
the HeatNode constructor `ValueError` but says nothing about the HTTP status a
client sees. The change is described only in the prose at 1749–1761, where the
prose asserts the opposite (F3).

## F5 — The three ZMQ bind defaults moving to loopback are not in the table

```
$ git show v0.3.1:src/maddening/viz/network.py | grep -n 'address: str ='
76:        address: str = "tcp://*:5555",
223:    def __init__(self, address: str = "tcp://*:5556",
$ git show v0.3.1:src/maddening/cloud/multigpu/coordinator.py | grep -n '0\.0\.0\.0'
194:            sock.bind(f"tcp://0.0.0.0:{self._port}")
```
vs current `src/maddening/viz/network.py:95,302` (`tcp://127.0.0.1:5555/5556`)
and `src/maddening/cloud/multigpu/coordinator.py:105`
(`bind_host: str = "127.0.0.1"`). Genuinely breaking for anyone reaching a
default-constructed publisher from another host. Described in prose only, under
"The ZeroMQ transports: loopback by default, CURVE off-box" at line 1866. Row
2235 covers the HTTP bearer token, not the ZMQ binds; row 2236 covers cloud
firewall ports, not bind addresses.

## F6 — MEDIUM. The cited CI pass/skip/xfail numbers do not add up

`docs/release_notes/v0.4.0.md:1967-1969` cites job `test (3.12)` of run
35371285787 at `201da73` on 2026-09-18: "**3184 passed, 28 skipped, 27
deselected, 4 xfailed** from 3241 collected". 3241 − 27 = 3214 selected, but
3184 + 28 + 4 = 3216. Two tests unaccounted for, with no failures, errors or
xpasses stated. (`201da73` exists; its date matches.)

## F7 — MEDIUM. "3440 tests collected" is ~880 behind the tip

`docs/release_notes/v0.4.0.md:1964-1965` pins the count to `c51cd6a`. Collection
only (not a run) at `0219b82`: `4294/4322 tests collected (28 deselected)`.
Explicitly pinned to a commit, so not a lie — but it is five commits and 882
tests behind the release tip, which is the failure mode the same bullet warns
about ("Count-bearing lines should be generated at release time"). Low-confidence
sub-note: `c51cd6a`'s author and commit date are 2026-09-18T23:42:38+02:00, not
the 2026-09-19 the line carries.

## F8 — LOW. `pyproject.toml:7` is still `0.4.0.dev0`

Expected during release prep; flagged so the final bump is not forgotten.

---

## Part C — completeness of the breaking-changes table

| # | Change | Genuinely breaking? | In the table? |
|---|---|---|---|
| 1 | `HeatNode` `T[0] != left_temperature` | Yes | **LISTED**, line 2234 — all three sub-claims verified true |
| 2 | `POST /graph/nodes` 201 → 400 | Yes | **NOT LISTED** (F4); prose asserts the opposite (F3) |
| 3 | Three ZMQ bind defaults → loopback | Yes | **NOT LISTED** (F5) |
| 4 | `FitResult` keyword-only | No — `maddening.sysid` does not exist at v0.3.1 | Correctly absent |
| 5 | Rank-cutoff move of `crb` finiteness | No — same reason | Correctly absent; documented in prose at 263 |

C1 measurement:
```
after 1 step: T[0] = 2.0  left_temperature = 100.0  equal? False
after 51 steps: T[0] = 54.24066925048828  equal to 100.0? False
no-BC: T[0] before 0.0 after 0.009999999776482582 frozen? False
stencil_order=2 unstable -> ValueError: ... Fourier number ... is 1, above the 0.5 limit
stencil_order=4 unstable -> ValueError: ... is 0.4, above the 0.3125 limit
MAX_FOURIER_NUMBER = {2: 0.5, 4: 0.3125}
```

C4: `git cat-file -e v0.3.1:src/maddening/sysid.py` → does not exist;
`git grep -l -E 'FitResult|\bcrb\b|def fim\(' v0.3.1 -- src/` → nothing. The
keyword-only rule is in force (`sysid.py:1309`, `:396`), it just cannot break a
v0.3.1 caller.

C5: the cutoff in `_resolve_rank_rtol` moved from `n * eps` to
`max(n, sqrt(m)) * eps`. Observable:
```
float32 eps=1.192e-07  n*eps=2.384e-07  sqrt(m)*eps=7.539e-06
  n_residual= None: rank=2  crb=[500000.47 500000.47]  all finite=True
  n_residual= 4000: rank=1  crb=[inf inf]  all finite=False
```

---

## Rows verified true

* **2228** `solver` defaults to `"ift"` — `group.py:255`; `"fori"` still works
  and warns (`group.py:367-375`).
* **2230** old checkpoint path warns — `checkpoint.py:591-619`, "removed in 1.0".
* **2231** AdaptiveNode aliases present, `blindness_threshold=` warns —
  `adaptive/base.py:369,390,399-407`.
* **2232** `bound_valid` / `gradient_error_bound` return the new value and warn —
  `graph_manager.py:1024-1025`.
* **2233** `atol: float = 0.0` (`group.py:239`); read under every norm
  (`group.py:450-451,477-489`).
* **2236** `JobConfig.ports` defaults to `[]` — `cloud/launcher.py:136`.
* **2237** `validate_anomaly_registry()` enum-checks `resolution_status`;
  the shipped registry validates with 0 schema-only errors.
* **2235** bearer token enforced iff not loopback — `api/auth.py:300-328`.
* Dependency-floor table (2208–2211) matches `pyproject.toml:11,45-46` against
  `git show v0.3.1:pyproject.toml` exactly.
* All four cited SHAs (`c51cd6a`, `201da73`, `c93bc89`, `6a597f2`) exist.
* Row 2229 (`run_sweep`/`run_scan` ~1 ulp) was not cheaply falsifiable and was
  not tested.

---

## Independently confirmed by the lead auditor

* The registry side of F1: 8 entries are `open` or `partially_resolved`
  (002, 003, 005, 010, 011, 012, 013, 014), against 15 registered.
* The v0.3.1 ZMQ bind defaults and the current loopback defaults (F5) —
  see `repro/compdocs_ano015_defaults.py` and `git show v0.3.1:...`.
* The `bound_valid` / `gradient_error_bound` deprecation (row 2232) —
  see `repro/compdocs_ano005_keys.py`.
* `MAX_FOURIER_NUMBER = {2: 0.5, 4: 0.3125}` and the constructor refusals (C1) —
  see `repro/compdocs_ano002.py` and `repro/compdocs_ano009.py`.

Not independently re-measured: the release-notes line numbers, the row-by-row
table extraction, the `POST /graph/nodes` status-code matrix, the CI-count
arithmetic (F6) and the `crb` measurement (C5).
