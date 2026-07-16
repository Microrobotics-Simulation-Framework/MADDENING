"""Matrix-free wavelet operator apply — the route past the dense-assembly ceiling.

Dense assembly (``operator.assemble_wave_dense``) materialises ``Wn`` and
``A_phys`` as ``(N, N)`` arrays, so peak memory is O(N²) and 3D dies at ~16³
(see ``operator.py`` module docstring).  This module applies the same operator
``A_wave = Wnᵀ A_phys Wn`` **without materialising anything**, so memory is O(N)
and the reachable grid is bounded by the solve, not the assembly.

Building blocks:

* ``make_wn_ops`` — the L²-normalised synthesis ``Wn v`` and its exact transpose
  ``Wnᵀ u``.  ``Wn = W diag(1/norms)``, so ``Wn v = synthesis(v / norms)`` and
  ``Wnᵀ u = synthesis_transpose(u) / norms``.  The transpose is
  ``jax.linear_transpose`` of synthesis — **never ``analysis``** (D3: DD wavelets
  are non-orthogonal, ``W⁻¹ ≠ Wᵀ``; ``analysis`` is the inverse and is wrong by
  order unity in the adjoint path).

* ``make_wave_apply`` — the full scaled matvec ``v → Â v`` used by CDD selection
  and the masked CG solve (M16): ``Â = D⁻¹ Wnᵀ A_phys Wn D⁻¹``.

The ``norms`` vector is a fixed O(N) array computed once at construction.  M14
supplies it in O(log N) via structural-block representatives; this module only
consumes it.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from maddening.nodes.adaptive.wavelets import transform as T

__all__ = ["make_wn_ops", "make_wave_apply",
           "make_laplacian_apply", "make_varcoeff_apply",
           "make_masked_operator_fn", "masked_cg_solve",
           "wave_diagonal_fast", "grid_gradient"]


def grid_gradient(u_flat: jax.Array, side: int, dim: int, h: float):
    """Central-difference gradient on the periodic grid, for any scalar field.

    Returns a list ``[∂₀u, …, ∂_{dim-1}u]`` of flat length-``side**dim`` arrays.
    A general derivative operator (``transform.py`` had none).  Linear in ``u``,
    so it composes with the (linear) synthesis to give a differentiable ``∇u`` —
    e.g. for a gradient-at-points sensor.
    """
    u = u_flat.reshape((side,) * dim)
    return [((jnp.roll(u, -1, axis=d) - jnp.roll(u, 1, axis=d)) / (2 * h)).reshape(-1)
            for d in range(dim)]


def wave_diagonal_fast(n_levels: int, n_coarse: int, order: int, dim: int,
                       norms: jax.Array,
                       a_phys_apply: Callable[[jax.Array], jax.Array]
                       ) -> jax.Array:
    """``diag(A_wave)`` in **O(N log N)** — the preconditioner diagonal without
    assembling ``A_wave``.

    ``diag(A_wave)_j = ⟨Wn·e_j, A_phys Wn·e_j⟩ = ⟨s_j, A_phys s_j⟩ / norms_j²``
    with ``s_j = synthesis(e_j)``.  For a **translation-invariant (constant-
    coefficient)** ``A_phys`` this is constant within each structural block
    (derisk D4), so one representative per block suffices — the same O(log N)
    trick as :func:`operator.column_norms_fast`.

    .. warning::
       Block-invariance holds only for constant-coefficient ``A_phys``.  For a
       variable coefficient the true diagonal is not block-constant; the lagged
       preconditioner (M19) evaluates this at a **reference** ``a₀`` and freezes
       ``D`` (measured: a saturating ~2.2× conditioning cost, FINDINGS_D5 /
       measurement 1), which is why lagging is the design.
    """
    synth = T._SYNTH[dim]
    N = T.n_dofs(n_levels, n_coarse, dim)
    block_ids_np, reps_np = T.structural_blocks(n_levels, n_coarse, dim)
    reps = jnp.asarray(reps_np)

    def rep_num(j):
        s = synth(jax.nn.one_hot(j, N, dtype=jnp.float64), n_levels, n_coarse, order)
        return jnp.sum(s * a_phys_apply(s))          # ⟨s_j, A_phys s_j⟩

    block_num = jax.vmap(rep_num)(reps)
    block_diag = block_num / (norms[reps] ** 2)
    return block_diag[jnp.asarray(block_ids_np)]


def make_laplacian_apply(side: int, dim: int, h: float, mass: float = 1.0
                         ) -> Callable[[jax.Array], jax.Array]:
    """Matrix-free ``(-Δ + mass·I)`` matching :func:`operator.physical_laplacian`.

    That dense operator is the Galerkin tensor form ``Σ_d S⊗M…`` with 1D
    stiffness ``S`` and lumped mass ``M = h·I``, so the matvec is a ``jnp.roll``
    stencil: ``Σ_d h^{dim-2}(2u − u₊ − u₋) + mass·h^dim·u``.  Never materialises
    ``(N, N)``.  Input/output are flat length-``side**dim`` vectors.
    """
    shape = (side,) * dim
    stiff_scale = h ** (dim - 2)
    mass_scale = mass * (h ** dim)

    def apply(u_flat: jax.Array) -> jax.Array:
        u = u_flat.reshape(shape)
        out = mass_scale * u
        for d in range(dim):
            out = out + stiff_scale * (
                2.0 * u - jnp.roll(u, -1, axis=d) - jnp.roll(u, 1, axis=d))
        return out.reshape(-1)

    return apply


def make_varcoeff_apply(a_grid: jax.Array, side: int, dim: int, h: float,
                        mass: float = 1.0) -> Callable[[jax.Array], jax.Array]:
    """Matrix-free ``(-∇·(a∇·) + mass·I)`` matching :func:`operator.physical_varcoeff`.

    Conservative, face-averaged, periodic.  For each axis the face coefficient is
    the mean of adjacent cell values; the matvec is
    ``mass·u + Σ_{d,±} a_face(u − u_neighbour)``.  Fully traceable in ``a_grid``,
    so ``dJ/da`` flows (the θ→A / application-1 path).  ``a_grid`` may be flat
    (``side**dim``) or shaped ``(side,)*dim`` -- the node stores it flat in state.
    Never materialises ``(N, N)``.
    """
    a = jnp.asarray(a_grid).reshape((side,) * dim)
    inv_h2 = 1.0 / h ** 2

    def apply(u_flat: jax.Array) -> jax.Array:
        u = u_flat.reshape((side,) * dim)
        out = mass * u
        for d in range(dim):
            a_plus = 0.5 * (a + jnp.roll(a, -1, axis=d)) * inv_h2   # face to +1
            a_minus = 0.5 * (a + jnp.roll(a, 1, axis=d)) * inv_h2   # face to -1
            out = out + a_plus * (u - jnp.roll(u, -1, axis=d))
            out = out + a_minus * (u - jnp.roll(u, 1, axis=d))
        return out.reshape(-1)

    return apply


def make_wn_ops(n_levels: int, n_coarse: int, order: int, dim: int,
                norms: jax.Array
                ) -> Tuple[Callable[[jax.Array], jax.Array],
                           Callable[[jax.Array], jax.Array]]:
    """Return ``(wn_apply, wn_transpose)`` for the L²-normalised synthesis.

    ``wn_apply(v) = Wn v`` (wavelet coeffs → grid values);
    ``wn_transpose(u) = Wnᵀ u`` (grid values → wavelet coeffs).
    """
    synth = T._SYNTH[dim]

    def wn_apply(v: jax.Array) -> jax.Array:
        return synth(v / norms, n_levels, n_coarse, order)

    def wn_transpose(u: jax.Array) -> jax.Array:
        return T.synthesis_transpose(u, n_levels, n_coarse, order, dim) / norms

    return wn_apply, wn_transpose


def make_wave_apply(n_levels: int, n_coarse: int, order: int, dim: int,
                    norms: jax.Array, a_phys_apply: Callable[[jax.Array], jax.Array],
                    D: jax.Array) -> Callable[[jax.Array], jax.Array]:
    """Return the scaled matvec ``v → Â v`` with ``Â = D⁻¹ Wnᵀ A_phys Wn D⁻¹``.

    ``a_phys_apply`` is the matrix-free physical operator (grid → grid; M15).
    ``D`` is the preconditioner's diagonal scaling.  Nothing is materialised.
    """
    wn_apply, wn_transpose = make_wn_ops(n_levels, n_coarse, order, dim, norms)

    def apply(v: jax.Array) -> jax.Array:
        w = v / D                      # D⁻¹ v
        grid = wn_apply(w)             # Wn D⁻¹ v  (coeffs → grid)
        grid = a_phys_apply(grid)      # A_phys Wn D⁻¹ v
        coeff = wn_transpose(grid)     # Wnᵀ A_phys Wn D⁻¹ v
        return coeff / D               # D⁻¹ (…)

    return apply


def make_masked_operator_fn(apply: Callable[[jax.Array], jax.Array],
                            mask: jax.Array) -> Callable[[jax.Array], jax.Array]:
    """Matrix-free frozen-active-set operator ``v → A_eff v``.

    The fn-based analogue of :func:`operator.make_masked_operator` (which needs a
    dense/BCOO ``A``): acts as ``apply`` on the active block and as the identity
    on inactive rows/cols, so the frozen solve returns ``0`` outside the mask::

        A_eff v = where(mask, apply(where(mask, v, 0)), v)

    ``apply`` is any matrix-free matvec (e.g. :func:`make_wave_apply`).  The
    active block inherits ``apply``'s symmetry/PSD, so CG is valid.
    """
    def operator_fn(v: jax.Array) -> jax.Array:
        vm = jnp.where(mask, v, 0.0)
        Av = apply(vm)
        return jnp.where(mask, Av, v)

    return operator_fn


def masked_cg_solve(apply: Callable[[jax.Array], jax.Array], mask: jax.Array,
                    rhs_scaled: jax.Array, *,
                    inner_precond: Callable[[jax.Array], jax.Array] | None = None,
                    rtol: float = 1e-10, atol: float = 1e-12) -> jax.Array:
    """Frozen active-set solve by masked CG — the matrix-free replacement for
    :func:`operator.gather_solve`.

    ``gather_solve`` does ``A[jnp.ix_(ix, ix)]``, which requires a dense ``A`` and
    is structurally incompatible with a matrix-free operator; this solves the same
    active-block system ``A_ΛΛ c_Λ = b_Λ`` iteratively via
    :func:`~maddening.core.solver_utils.ift_linear_solve` (CG) on the masked
    matvec, so it never materialises anything.  ``apply`` is the scaled matvec
    ``v → Â v``; under diagonal scaling ``Â`` is well-conditioned and
    ``inner_precond`` is ``None`` (the M5 diagonal-mode slot).  A contrast-robust
    preconditioner (R1) passes a masked ``v → M⁻¹v`` here.

    .. note::
       At high coefficient contrast the active-block system is ill-conditioned
       (κ ∝ contrast); CG then needs many iterations and, at inadequate budget,
       a small residual does not bound the solution error (D5 / FINDINGS_D5).
       That high-contrast regime is R1's concern (a contrast-robust preconditioner).
    """
    from maddening.core.solver_utils import ift_linear_solve  # lazy: lineax dep

    op = make_masked_operator_fn(apply, mask)
    b = jnp.where(mask, rhs_scaled, 0.0)
    precond = None
    if inner_precond is not None:
        precond = make_masked_operator_fn(inner_precond, mask)
    return ift_linear_solve(op, b, solver="cg", preconditioner=precond,
                            rtol=rtol, atol=atol)
