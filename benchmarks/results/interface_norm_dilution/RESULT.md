# The interface norm and a grid-sized delivered value

Measured 2026-10-07 on `release/0.4.0` at `fcf7e6cb` (float64, CPU, `solver="ift"`, `rtol=1e-4`,
loop gain 0.54). Script: `measure.py`; numbers: `result.json`.

A "robot" with 30 markers coupled two ways to a 1-D "grid" of N cells: grid velocity gathered at
the markers (N → 30), marker forces scattered to the grid by the transpose (30 → N). The same
problem built three ways, same tolerance; the exact fixed point is a 30 × 30 linear solve.

| N | build, norm | passes (GS / Jacobi) | error of the marker forces at exit ÷ tolerance (GS / Jacobi) |
|---|---|---|---|
| 1e3 | edge-mapped, interface | 10 / 22 | 23 / 12 |
| 1e3 | marker-side, interface | 13 / 27 | 3.6 / 1.9 |
| 1e3 | edge-mapped, mixed | 11 / 21 | 12 / 12 |
| 1e4 | edge-mapped, interface | 9 / 18 | 39 / 39 |
| 1e4 | marker-side, interface | 13 / 26 | 3.3 / 3.3 |
| 1e4 | edge-mapped, mixed | 9 / 18 | 39 / 39 |
| 1e5 | edge-mapped, interface | 7 / 15 | 134 / 72 |
| 1e5 | marker-side, interface | 13 / 26 | 3.3 / 3.3 |
| 1e5 | edge-mapped, mixed | 7 / 15 | 134 / 72 |
| 1e6 | edge-mapped, interface | 5 / 11 | 459 / 248 |
| 1e6 | marker-side, interface | 13 / 26 | 3.3 / 3.3 |
| 1e6 | edge-mapped, mixed | 5 / 11 | 459 / 248 |

"edge-mapped": both mappings on the edges, so the interface norm reads what each edge DELIVERS
(30 numbers for the gather, N for the scatter). "marker-side": the scatter is applied inside the
grid node, so both internal edges carry 30 numbers.

## Reading

- With the scatter's delivered value in the reading, the interface norm stops earlier as the
  grid grows: the error at exit is 23, 39, 134 and 459 times the tolerance at N = 1e3 to 1e6
  (Gauss-Seidel), against a constant 3.3 when both readings are marker-sized (3.3 is the ordinary
  residual-to-error factor of this loop, 1 / (1 − gain) weighted). Relative to the marker-side
  build: 7, 12, 40 and 138 times looser.
- **It gives the same verdict as the mixed norm** (identical passes and error from N = 1e4 up):
  the grid-sized delivered value makes the interface norm behave like a whole-state norm.
- Cause, from `coupling_residual_interface`: each delivered value is scaled by its own magnitude
  and all entries of all delivered values are pooled into one RMS; a delivered field of N entries
  of which about 60 change adds N to the count and almost nothing to the sum.
- This is today's behaviour for STATIC mapped edges too (the measurement uses `matrix_mapping`).

## After the compact-side rule (stage R1b)

Re-measured 2026-10-08 with the same script on the branch that makes the interface norm read a
static mapped edge on its compact side (an edge whose mapping delivers more entries than its
source holds is read at its source value; a tie and a gather as delivered), jax 0.11.0, CPU,
float64.  Numbers: `result_compact_side.json` (`result.json` is the measurement above).

| N | build, norm | passes (GS / Jacobi) | error of the marker forces at exit ÷ tolerance (GS / Jacobi) |
|---|---|---|---|
| 1e3 | edge-mapped, interface | 13 / 27 | 3.6 / 1.9 |
| 1e3 | marker-side, interface | 13 / 27 | 3.6 / 1.9 |
| 1e3 | edge-mapped, mixed | 11 / 21 | 12 / 12 |
| 1e4 | edge-mapped, interface | 13 / 26 | 3.3 / 3.3 |
| 1e4 | marker-side, interface | 13 / 26 | 3.3 / 3.3 |
| 1e4 | edge-mapped, mixed | 9 / 18 | 39 / 39 |
| 1e5 | edge-mapped, interface | 13 / 26 | 3.3 / 3.3 |
| 1e5 | marker-side, interface | 13 / 26 | 3.3 / 3.3 |
| 1e5 | edge-mapped, mixed | 7 / 15 | 134 / 72 |
| 1e6 | edge-mapped, interface | 13 / 26 | 3.3 / 3.3 |
| 1e6 | marker-side, interface | 13 / 26 | 3.3 / 3.3 |
| 1e6 | edge-mapped, mixed | 5 / 11 | 459 / 248 |

Before and after, the error of the marker forces at a `converged=True` exit in tolerances
(Gauss-Seidel / Jacobi):

| N | edge-mapped, interface: read as delivered (before) | edge-mapped, interface: compact side (now) | marker-side, interface | edge-mapped, mixed |
|---|---|---|---|---|
| 1e3 | 23 / 12 | 3.6 / 1.9 | 3.6 / 1.9 | 12 / 12 |
| 1e4 | 39 / 39 | 3.3 / 3.3 | 3.3 / 3.3 | 39 / 39 |
| 1e5 | 134 / 72 | 3.3 / 3.3 | 3.3 / 3.3 | 134 / 72 |
| 1e6 | 459 / 248 | 3.3 / 3.3 | 3.3 / 3.3 | 459 / 248 |

- The edge-mapped build under the interface norm is now the marker-side build: the same passes,
  the same reported residual and the same error at every N, to the last digit `result_compact_side.json`
  holds (the scatter edge is read at the 30 marker forces it carries, which is what the
  marker-side build's plain edge delivers).
- The error at exit no longer grows with the grid: 3.3 tolerances at N = 1e4 to 1e6 (3.6 at
  1e3, Gauss-Seidel), where it was 39, 134 and 459.
- The mixed norm's rows and the marker-side rows are the numbers of the first measurement: the
  rule changes only what the interface norm reads on an edge that expands.
