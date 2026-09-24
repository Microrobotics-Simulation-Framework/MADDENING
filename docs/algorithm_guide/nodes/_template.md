---
bibliography: ../../bibliography.bib
---

# [Node Name]

**Module**: `maddening.nodes.[module]`
**Stability**: [experimental | provisional | stable | deprecated]
**Algorithm ID**: `MADD-NODE-[XXX]`
**Version**: [semantic version of this algorithm implementation]

<!-- For downstream nodes only (not MADDENING reference implementations): -->
<!-- **Verification Mode**: [Mode 1 (Wrapping) | Mode 2 (Independent)] -->
<!-- **Upstream Node**: [MADDENING node class, if Mode 1] -->
<!-- **MADDENING Version Pin**: [exact version, if Mode 1] -->

## Summary

[1-2 sentence description of what this node simulates.]

## Governing Equations

[Full mathematical formulation. Use LaTeX math blocks.]

$$
\frac{\partial T}{\partial t} = \alpha \nabla^2 T + S
$$

## Discretization

[How the continuous equations are discretized. Explicit/implicit,
finite difference/finite element/lattice Boltzmann, order of accuracy.]

## Implementation Mapping

[Trace every term in the governing equations and discretization to the
specific Python/JAX function that implements it. This is mandatory for
{term}`IEC 62304` Class C detailed design traceability (Clause 5.4).]

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\alpha \nabla^2 T$ (diffusion) | `maddening.nodes.heat.HeatNode.update` | 2nd-order central FD stencil |
| $S$ (source term) | `maddening.nodes.heat.HeatNode.update` | Added after diffusion |
| Time integration ($\partial T / \partial t$) | Forward Euler in `maddening.nodes.heat.HeatNode.update` | 1st-order explicit |
| Boundary conditions | `state.at[0].set(left_T)` | JAX primitive: `jax.numpy.ndarray.at[].set()` |

[If a term is handled by a JAX primitive or third-party function for
which no explicit MADDENING function exists, document the primitive
and its calling convention. Every governing equation term must appear
in this table — silent omissions are not acceptable.]

[Every code span in the Implementation column is a fully qualified
`maddening.*` name, which `scripts/check_impl_mapping.py` resolves; the
only exception is a row whose Notes cell begins `JAX primitive` or
`Third-party`, as in the last row above. A symbol the class inherits
rather than defines needs a Notes cell that begins
``Inherited from `BaseClass` ``.]

[Code-driven generation of this table is recommended but not mandated.
The table may be maintained manually provided it is complete and reviewed
at each release. See `scripts/check_impl_mapping.py` for CI verification.]

## Assumptions and Simplifications

[Numbered list of every physical and mathematical assumption.]

1. [e.g., "Incompressible flow (Mach number << 1)"]

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| | | |

## Known Limitations and Failure Modes

[Specific conditions where this model is known to produce incorrect or
unreliable results. This section feeds directly into IEC 62304 {term}`SOUP`
anomaly assessment.]

1. [e.g., "CFL > 1 causes numerical instability"]

## Stability Conditions

[Analytical or empirical stability bounds for the numerical scheme.]

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| | | | |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| | | | | |

## Boundary Inputs

| Field | Shape | Default | Description |
|-------|-------|---------|-------------|
| | | | |

## References

[Cite using Pandoc-style `[@Key]` syntax, where `Key` matches an entry
in `docs/bibliography.bib`.  Multiple citations: `[@Key1; @Key2]`.
CI validates all cited keys exist via `scripts/check_citations.py`.]

- [@ExampleKey] Author, A. (Year). *Title*. Publisher. — Brief note on relevance to this node.

## Verification Evidence

[Link to verification report and test files.]

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | YYYY-MM-DD | Initial implementation |
