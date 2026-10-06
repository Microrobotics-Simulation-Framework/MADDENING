"""The coupling inventory's domain-lift claims, with a member sharded over four devices.

``tests/core/test_coupling_norm_edges_in_every_domain.py``,
``test_coupling_accelerations_in_every_domain.py`` and
``test_coupling_configuration_in_every_domain.py`` state their rows' claims
once, over :mod:`tests.core.coupling_domains`' memoryless pair, and run
them in every numeric domain but this one, which needs the CPU virtual
devices this directory's conftest provides.  Here the same test bodies run
with ``label="sharded"``: member ``b`` partitioned along its field by
``ShardedPointwiseNode`` over a four-device mesh, every field four copies of
the entries the test asks for, so the claim and its float64 oracle are
unchanged, entry by entry.  What the sharding reaches: the group's norm,
floor and spectrum read ``b``'s field through cross-device reductions
(``max|v|`` per field, the dot products of the accelerators and of the
Arnoldi process), and the edges carry a sharded array into a replicated
member.
"""

from __future__ import annotations

import jax
import pytest

from tests.core import coupling_domains as cd
from tests.core import test_coupling_accelerations_in_every_domain as accelerations
from tests.core import test_coupling_configuration_in_every_domain as configuration
from tests.core import test_coupling_mapped_edges_in_every_domain as mapped
from tests.core import test_coupling_norm_edges_in_every_domain as norms
from tests.core import test_coupling_spectral_bound_where_edges_share_a_field as shared

pytestmark = pytest.mark.skipif(len(jax.devices()) < cd.N_SHARD,
                                reason=f"needs {cd.N_SHARD} CPU-virtual devices")

#: Each row's claim, by the test bodies that state it.
CASES = {
    "CPL-011": [norms.test_a_group_whose_moving_fields_are_all_dead_banded_stops_after_one_pass],
    "CPL-043": [norms.test_a_group_at_its_dtypes_overflow_edge_converges_only_where_it_is_measured],
    "CPL-044": [norms.test_the_verdict_does_not_change_below_the_dtypes_normal_range],
    "CPL-090": [norms.test_a_dead_banded_field_on_the_loop_stays_in_the_spectrum],
    "CPL-049": [accelerations.test_the_estimate_is_the_distance_for_fixed_relaxation],
    "CPL-050": [accelerations.test_an_under_relaxed_first_pass_estimate_is_not_below_the_distance],
    "CPL-061": [accelerations.test_the_guard_holds_aitken_past_its_own_exit,
                accelerations.test_iqn_stops_on_its_first_sub_threshold_pass],
    "CPL-073": [accelerations.test_a_contraction_above_one_is_reported_unconverged,
                accelerations.test_iqn_converges_a_contraction_above_one],
    "CPL-074": [accelerations.test_iqn_on_a_single_accelerated_scalar_agrees_with_plain_iteration],
    "CPL-076": [accelerations.test_flux_reading_producers_step_under_jacobi_to_the_gauss_seidel_fixed_point],
    "CPL-014": [configuration.test_imvj_without_reuse_is_iqn_ils],
    "CPL-015": [configuration.test_relaxation_one_is_no_relaxation],
    "CPL-016": [configuration.test_one_pass_reads_the_iterate_its_mode_documents],
    "CPL-142": [configuration.test_the_initial_guess_gets_a_zero_derivative],
    # The interface norm and its bound on mapped internal edges: the mapping
    # is applied to a sharded field, and the norm reads what it delivers.
    "CPL-041": [mapped.test_the_interface_norm_measures_what_mapped_edges_deliver],
    # ... and on a field that two internal edges read, which the norm counts twice.
    "CPL-088": [mapped.test_a_usable_bound_covers_the_distance_in_what_mapped_edges_deliver,
                shared.test_a_usable_bound_covers_the_distance_where_a_node_reads_a_field_twice],
    "CPL-188": [mapped.test_the_reports_floor_is_the_one_the_step_measured_with_its_weights],
}


def test_the_member_is_partitioned():
    """The premise: ``b``'s field lives on four devices, one shard each."""
    d = cd.DOMAINS[cd.SHARDED]
    gm = cd.pair(d, g=(0.5, 0.5), tolerance=1e-5, max_iterations=10)
    field = gm._state["b"]["x"]
    assert len(field.sharding.device_set) == cd.N_SHARD
    assert field.addressable_shards[0].data.shape == (1,)
    assert gm._state["a"]["x"].shape == (cd.N_SHARD,)


# Slow: every row's graphs compiled with a member partitioned over four
# devices, a few seconds each on three cores.  A coupling group with a sharded
# member stays on every push, forward and adjoint, against an independent
# float64 model, and these rows' claims stay on every push in every other
# domain.
# Per push: tests/cloud/multigpu/test_coupling_group_with_sharded_and_replicated_members.py::test_the_forward_rollout_is_the_implicitly_coupled_step
# tests/core/test_coupling_norm_edges_in_every_domain.py::test_the_verdict_does_not_change_below_the_dtypes_normal_range
@pytest.mark.slow
@pytest.mark.parametrize("row", sorted(CASES))
def test_the_claim_holds_with_a_sharded_member(row, monkeypatch):
    """Each row's domain test, run with ``b`` sharded."""
    for fn in CASES[row]:
        if "monkeypatch" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            fn(cd.SHARDED, monkeypatch)
        else:
            fn(cd.SHARDED)
