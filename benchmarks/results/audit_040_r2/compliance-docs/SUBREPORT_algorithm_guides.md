# Sub-report: `docs/algorithm_guide/**` — Implementation Mapping rows, stated orders, citations

Produced by a delegated auditor at commit `0219b82`, read-only, in the same
worktree. Reproduced below with one correction by the lead auditor (marked).
Findings appear in the main report as L1–L5.

---

## F1 (main report L1) — `heat_node.md:45` states a rejected closure is a whole order worse than it is

Verbatim:

> A cubic is the lowest degree that works.  The 5-point stencil divides ghost
> errors by $12\Delta x^2$, so **a linear or quadratic extrapolation caps the
> scheme at order 1 or 3 (measured 0.95 and 2.97)**; a quartic is accurate
> enough but spectrally unstable.

Re-ran the node's own `stencil_order=4` MMS spatial study (profile
`sin(2 pi x) + 0.5x + 1 + 0.4x^2`, Fo = 0.3, float64, 10/20/40/80/160) with
`maddening.nodes.heat._dirichlet_ghosts_4th_order` monkeypatched to a Lagrange
extrapolation of degree 1 / 2 / 3 through the rod end and the nearest cell
centres:

```
linear (degree 1):
   errors : 1.7111e-03  1.8538e-04  4.2374e-05  1.0548e-05  2.6363e-06
   orders : 3.206  2.129  2.006  2.000    <-- finest pair: 2.000
quadratic (degree 2):
   errors : 4.4773e-03  6.8296e-04  9.0836e-05  1.1608e-05  1.4639e-06
   orders : 2.713  2.910  2.968  2.987    <-- finest pair: 2.987
cubic (degree 3, shipped):
   errors : 5.5202e-04  4.0750e-05  2.8643e-06  1.9015e-07  1.2244e-08
   orders : 3.760  3.831  3.913  3.957    <-- finest pair: 3.957
```

The degree-3 arm reproduces the shipped numbers exactly, which validates the
harness (and the lead auditor measured the same four numbers independently, via
`repro/compdocs_mms_bands.py`). Linear extrapolation measures **2.000**, not
order 1 / 0.95.

Mechanism: the guide conflates the pre-0.4.0 *defect* with a *linear closure*.
0.954 is MADD-ANO-008's measurement of the mispositioned ghosts
(`ghost = T_left`, `2*T_left - T[1]`, read at the wrong x), not of a linear
extrapolation. Every other copy in the tree says 2.00:
`src/maddening/nodes/heat.py:127-129`, `known_anomalies.yaml:588-590`
(MADD-ANO-008 `residual_risk`), `tests/verification/test_mms_order.py:135-137`.
The guide is the only artefact carrying the stale number.

## F2 (L2) — `explicit_integrators.md:157-158` asserts an error agreement its own Known Limitations section denies

Verbatim, in *Stability Conditions*:

> under a frozen time-varying input all three methods measure order ~1.02 **with
> errors agreeing to three significant figures**, so a ladder cannot distinguish
> one scheme from another.

Replicated the frozen-input study of `tests/verification/test_integrator_order.py`
(x' = -x + u(t), u manufactured so x*(t) = sin(2 pi t), T = 0.75, float64):

```
    n         dt          euler           heun            rk4   rk4/euler
   10   0.075000   9.087449e-02   1.329754e-01   1.319463e-01   1.452
   20   0.037500   4.053470e-02   6.096666e-02   6.071380e-02   1.498
   40   0.018750   1.911007e-02   2.914170e-02   2.907929e-02   1.522
   80   0.009375   9.273520e-03   1.424008e-02   1.422460e-02   1.534
  160   0.004687   4.567314e-03   7.037920e-03   7.034064e-03   1.540

Observed order over the finest pair (160 vs 80):
  euler : 1.022   heun : 1.017   rk4 : 1.016

At dt = 0.0047:  euler=0.004567  heun=0.007038  rk4=0.007034
  3-significant-figure renderings: euler=0.00457 heun=0.00704 rk4=0.00703
  agree to three significant figures? NO
  rk4/euler = 1.540   heun/euler = 1.541
```

The same document, 30 lines earlier (`:128-130`), says the opposite as a Known
Limitation: "`rk4_step` can be less accurate than `euler_step` under that regime
— measured 1.5x worse at dt = 0.0047". MADD-ANO-014 records 7.03e-3 vs 4.57e-3,
so the Known-Limitations version is the correct one. The false sentence is the
justification offered for why the stability-polynomial identity check is needed;
the conclusion survives, the stated evidence does not. Propagated from
`tests/verification/test_integrator_order.py:26-28`. No test pins it;
`test_integrator_order.py` passes 32/32.

## F3 (L3) — `uq/index.md:9` names a module that does not exist

Verbatim: "The `UncertaintySpec` and `UncertainParameter` dataclasses in
`maddening.core.uq` define the interface for nodes to declare their UQ
capabilities."

```
$ python -c "import maddening.core.uq"
ModuleNotFoundError: No module named 'maddening.core.uq'
$ grep -rn "class UncertaintySpec\|class UncertainParameter" src/
src/maddening/core/compliance/uq.py:26:class UncertainParameter:
src/maddening/core/compliance/uq.py:56:class UncertaintySpec:
$ grep -rn "maddening\.core\.uq\b" --include='*.py' --include='*.md' .
docs/algorithm_guide/uq/index.md:9   (the only occurrence in the repository)
```

Gate gap that lets it survive: `scripts/check_impl_mapping.py` only parses rows
under a `## Implementation Mapping` heading. Three in-scope guides have no such
section and no `MIN_MAPPINGS` pin, so no gate checks a single symbol claim in
them:

```
docs/algorithm_guide/uq/index.md                     pinned=False  rows=0
docs/algorithm_guide/coupling/interface_mapping.md   pinned=False  rows=0
docs/algorithm_guide/coupling/unit_transforms.md     pinned=False  rows=0
```

Every backticked symbol in all three was hand-resolved; this is the only bad one.

## F4 (L4) — `adaptive_node.md:301` cites design evidence that is not in the package

> - Design evidence: `plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md`
>   (seven spike rounds behind the constants above).

```
$ git ls-files plans/      (empty)
$ git log --oneline --all -- 'plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md'   (empty)
$ find . -iname '*SPIKE_FINDING*' -not -path './.git/*'   (no output)
```

**Lead auditor's correction.** The delegate concluded the file "has never
existed". It does exist — at `/home/nick/MSF/msf/plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md`,
in the maintainer's workspace one directory above the repository, with
substantive content (sections Q1–Q6 plus two investigations, 40 references to
spike rounds). `plans/` is not gitignored; it is simply outside the repo root.
So the correct finding is narrower: a shipped algorithm guide sources every
constant in its *Validated Physical Regimes* table to a path that resolves to
nothing for any recipient of the release. `docs/developer_guide/adaptive_node.md:39`
cites it too. The constants themselves are correct against the class attributes,
and the measurement claims are pinned by a passing test.

## F5 (L5) — `check_citations.py` claims 50 verified; 45 were checked

```
$ python scripts/check_citations.py
WARNING: .../docs/bibliography.bib entry 'ShanChen1993' is not cited by any document
OK: 50 citation(s) verified (18 unique keys, 19 bib entries)
EXIT=0
```

`main()` prints `len(citations)` (`check_citations.py:187`), the raw scan count,
but the loop at `:165-166` `continue`s past the five `_TEMPLATE_CITATIONS`
entries before testing them against the bibliography. Scoping the gate to the
directory holding all five:

```
$ python scripts/check_citations.py docs/developer_guide
OK: 21 citation(s) verified (12 unique keys, 19 bib entries)
$ (independent count)  total [@..] citations: 21; whitelisted: 5; checked: 16
```

No real citation is dangling, so this is count honesty only. Note that the
"18 unique keys" *is* honest (19 distinct keys minus the whitelisted `Key`),
which is what makes the mismatch easy to miss.

---

## Checked and sound

**1. Implementation Mapping arithmetic is honest — no silently skipped row.**
`check_impl_mapping.py` → `OK: 58 implementation mapping(s) verified`, EXIT=0.
Independent parse of the Markdown: 56 data rows carrying 58 distinct
`maddening.*` symbols (two rows carry two each: `ball_node.md:61` and
`spring_node.md:45`, both `...derivatives` + `...integrate_node`). 58 = 58.
Per-guide, the gate's count and `MIN_MAPPINGS` match exactly, zero slack:

```
guide                                                 rows  checked  errs  skip   MIN
docs/algorithm_guide/nodes/adaptive_node.md             12       12     0     0    12
docs/algorithm_guide/nodes/ball_node.md                  6        7     0     0     7
docs/algorithm_guide/nodes/heart_pump_node.md            9        9     0     0     9
docs/algorithm_guide/nodes/heat_node.md                  9        9     0     0     9
docs/algorithm_guide/nodes/rigid_body_2d_node.md         7        7     0     0     7
docs/algorithm_guide/nodes/spring_node.md                8        9     0     0     9
docs/algorithm_guide/solvers/explicit_integrators.md     5        5     0     0     5
TOTAL                                                   56       58
Rows the gate saw but resolved ZERO maddening.* symbols for: none
```

Raw `awk` count confirms 56 (70 pipe-lines − 14 header/separator), so the gate's
header heuristic drops nothing.

**2. All 58 symbols resolve independently of the gate** — `importlib` + `getattr`
against the *defining* class's `__dict__`: every one resolves, every one is
callable, **none is inherited**, and each matches what its row's Notes claim.

**3. `check_impl_mapping.py` mutation-tested; it can fail.** Renamed symbol →
"does not resolve". Backticks stripped → "has no code reference in its
Implementation column". Base-class-only symbol without an "inherited" note →
"resolves only through base class SimulationNode". Whole table deleted →
`check_pinned` → "0 ... at least 7 expected". One hole: a code span present but
not `maddening.`-prefixed does not fail `check_guide` alone, only the pin — so
the hole is live exactly for an unpinned guide, which is F3's territory.

**4. Order of accuracy agrees across guide, `NodeMeta` and benchmark for all six
node guides.**

| guide | guide's declaration | `NodeMeta.discretization_order` | benchmark |
|---|---|---|---|
| ball_node.md:40 | `(spatial=None, temporal=1.0)` | `(None, 1.0)` | MADD-VER-010, 1.002 (guide:122 matches) |
| spring_node.md:34 | `(None, 1.0)` | `(None, 1.0)` | MADD-VER-009, 1.029 (guide:115) |
| rigid_body_2d_node.md:50 | `(None, 1.0)` | `(None, 1.0)` | MADD-VER-011, 1.000 (guide:129) |
| heart_pump_node.md:48 | `(None, 1.0)` | `(None, 1.0)` | MADD-VER-012, 1.000, 200/400/800/1600 (guide:159) |
| heat_node.md:28-29 | 2nd/4th central in space, forward Euler in time | class `(2.0, 1.0)`; instance hook gives `spatial == 4.0` for `stencil_order=4` | VER-005 2.000, VER-006 0.998, 4th-order 3.957 (guide:139-142) |
| adaptive_node.md | declares none | `None` | MADD-VER-004, guide:276-281 matches the registry row |

The three open node anomalies are correctly *asserted* by the guides, not
contradicted: `ball_node.md:43-51` carries the forward-vs-semi-implicit warning
box (MADD-ANO-011) matching `NodeMeta.discretization`; `heart_pump_node.md:51-64`
carries the end-of-step-inflow warning (012) and the float32 backpressure note
(013); `heat_node.md` documents the 0.4.0 `stencil_order=4` fix consistently with
the code. Guide headers (Algorithm ID, Version, Stability, Module) match
`NodeMeta` on all six.

**5. MADD-ANO-014's caveat is carried in `explicit_integrators.md`.** The Butcher
table at 41-45 gives orders 1/2/4 with no inline caveat, but the section
immediately below is titled "What the order claim is conditional on" and says at
59-61: "the hold contributes an O(h^2) local error independently of the stage
arithmetic, so **every method in this module converges at order 1**. This is
registered as `MADD-ANO-014`." Plus Known Limitations 1 and 2 (124-130) and three
Validated-Regime rows (117-119). Every quantitative figure reproduces:
autonomous 1.003/2.005/4.006, per-step 1.022/1.017/1.016 (measured identically),
time-in-state 1.022/1.999/4.002, the "1.5x worse" (measured 1.540). Stability
intervals analytically exact: RK4 real boundary at z = -2.785293563 (guide says
-2.785), Heun and Euler at exactly -2. The identity-check row's
z in {-2.5, -0.3, 0.7} at 1e-13 relative matches
`test_integrator_order.py:314,342`. 32 passed. F2 is the only defect here.

**6. Every algorithm-guide citation resolves.** 19 bib entries, no duplicates, no
`%`-commented entries in play. Independent scan of `docs/**/*.md` finds 50
bracketed citations over 19 distinct keys — identical to the gate's scan — of
which the only non-resolving are the 5 whitelisted `[@Key]` teaching examples.
Per guide: adaptive 9/5, ball 3/3, heart_pump 4/3, heat 4/2, rigid_body_2d 3/3,
spring 3/3, explicit_integrators 3/2 — all resolve. Zero `{cite}` roles anywhere
in `docs/`, so nothing escapes the `[@Key]` scanner. `_template.md` is correctly
skipped. Only F5's count applies.

**7. `heat_node.md`'s quantitative stability claims reproduce exactly.** Building
the full discrete operator (boundary rows included) from the shipped
`HeatNode._compute_laplacian` and solving for the sharp forward-Euler bound:

```
     N   ord=4 sharp Fo_max   ord=2 sharp Fo_max
     5             0.316920             0.500000
     6             0.320629             0.500000
     8             0.323688             0.500000
    10             0.324543             0.500000
    20 … 1280      0.324851             0.500000
monotone non-decreasing over the sampled N: True
```

Guide:78,103 claims "0.3169 at N=5 ... rising monotonically to 0.3249" and
"5/16 = 0.3125 is below all of them" — all three exact. Constructor refuses
Fo = 0.5050 (order 2) and Fo = 0.3156 (order 4), accepts 0.4500 / 0.2812;
`n_cells=4, stencil_order=4` → "4th-order stencil requires at least 5 cells".
The cubic ghost formulas at 40-42 are algebraically exact (Lagrange weights
3.2/−3/1/−0.2 and 12.8/−18/8/−1.8, i.e. `(16,−15,5,−1)/5` and
`(64,−90,40,−9)/5`). Limitation 6 confirmed: `compute_boundary_fluxes` returns
`-alpha * (T[1] - T[0]) / dx_left` (`heat.py:892`).

**8. `unit_transforms.md`.** Header module `maddening.core.transforms_unit` is the
real implementation home (`transforms.py:282` re-exports, which is what the
examples import). All five conversion factors match at (dx,dt,rho)=(2,3,5):
length dx, velocity dx/dt, force rho·dx^4/dt^2, torque rho·dx^5/dt^2, pressure
rho·(dx/dt)^2. `BoundaryInputSpec.expected_units` and
`BoundaryFluxSpec.output_units` both exist. `GraphManager.validate()` produces
exactly the two documented unit warnings and nothing on the matching control.

**9. `interface_mapping.md`.** Every symbol exists: `Mapping`,
`StaticLinearMapping`, `MappingSpec`, the four factories with the documented
keywords (`ridge=1e-8`, `polynomial=True`, `epsilon`, `mode`,
`source_ref`/`target_ref`, `matrix_mapping(..., kind=, asset=)`), `rbf_matrix`,
`rbf_interpolation`, `PointReferenceError`, `MappingRebuildError` (both
`ValueError` subclasses). Constants exact: `INLINE_POINT_LIMIT = 64`,
`INLINE_ELEMENT_LIMIT = 1024`, `MAX_ASSET_BYTES = 256 MiB`. The `describe()`
kind-vs-spec-kind claim holds live. All four RBF kernels reproduce a constant
field.

**10. Constructor defaults and boundary specs.** Every *Parameters* and *Boundary
Inputs* row across the six node guides matches the live signature — Ball
(0.0/0.0/0.8/−9.81, 1e-4 chatter threshold present), Heat (10/1.0/0.01/0.0/2/None),
Spring (100/1/1/1/0/0), RigidBody2D (1/1/(0,−9.81)/0…), HeartPump
(1/1/72/70/0/0.35/80), Adaptive (`n_max` required and structural, threshold 0.7,
delta 0.05, `D_threshold` 5, `blindness_gate=True`, `on_blind="warn"`, all three
aliases present, `__abstractmethods__ == {compute_active_set, solve_frozen}`),
`integrate_node(..., method='rk4')`. Adaptive's K-ratio claim is pinned by the
test it names (2 passed). All 16 other test/doc files the guides cite as
evidence exist — the only dead pointer is F4.
