"""``str()`` of ``FitResult`` and ``FIMReport`` is a short human summary;
``repr()`` and every field are unchanged.

``ProfileReport`` already had a ``__str__`` and is not touched.
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.profiler import ProfileReport
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import FIMReport, FitResult, fim, fit


def _fit_result(**overrides) -> FitResult:
    fields = dict(params={"nodes": {"s": {"k": jnp.float32(29.5), "w": jnp.zeros(3)}}},
                  losses=np.array([2.0, 0.5, 0.25]), converged=False, n_iter=2,
                  best_iteration=2, best_loss=0.25)
    fields.update(overrides)
    return FitResult(**fields)


def test_fit_result_str_summarises_the_fit():
    text = str(_fit_result(excited_rank=1, undetermined_drift=0.04))
    assert text.splitlines()[0] == "FitResult: not converged after 2 iterations"
    assert "loss: first 2, last 0.25, best 0.25 (iterate 2)" in text
    assert "identifiability guard: excited rank 1, undetermined drift 0.04" in text
    assert "params (2 leaves):" in text
    assert "['nodes']['s']['k'] = 29.5" in text and "['nodes']['s']['w'] = array(3,)" in text


def test_fit_result_str_handles_the_edge_cases():
    text = str(_fit_result(losses=np.array([]), n_iter=0, best_loss=None, best_iteration=None,
                           converged=True))
    assert "converged after 0 iterations" in text and "loss: none evaluated" in text
    assert "identifiability guard: not measured" in text
    declined = str(_fit_result(excited_rank=1, undetermined_drift=0.5, hold_declined=True))
    assert "hold declined: params are the raw iterate" in declined
    many = _fit_result(params={"nodes": {f"n{i:02d}": {"k": jnp.float32(i)} for i in range(13)}})
    assert "... and 3 more" in str(many)


def test_fit_result_repr_is_still_the_dataclass_repr():
    r = _fit_result()
    assert re.fullmatch(
        r"FitResult\(params=\{.*\}, losses=array\(\[2\.  , 0\.5 , 0\.25\]\), converged=False, "
        r"n_iter=2, excited_rank=None, undetermined_drift=None, best_iteration=2, "
        r"best_loss=0\.25, hold_declined=None\)", repr(r), flags=re.S), repr(r)
    assert str(r) != repr(r)


def _fim_report() -> FIMReport:
    return FIMReport(fim=jnp.eye(2), eigvals=jnp.array([0.0, 4.0]),
                     eigvecs=jnp.array([[0.6, 0.8], [0.8, -0.6]]), rank=1, cond=float("inf"),
                     crb=jnp.array([jnp.inf, 0.25]), param_names=("['a']", "['bb']"),
                     zero_scaled=("['a']",))


def test_fim_report_str_summarises_identifiability():
    text = str(_fim_report())
    assert text.splitlines()[0] == ("FIMReport: rank 1 of 2 parameters "
                                    "(1 undetermined direction); cond inf")
    assert "least identifiable: ['bb'] (weight 0.8 in the weakest direction)" in text
    assert "    ['a']   inf" in text and "    ['bb']  0.25" in text
    assert "zero_scaled: ['a']" in text


def test_fim_report_repr_is_still_the_dataclass_repr():
    r = _fim_report()
    assert repr(r).startswith("FIMReport(fim=Array(")
    assert "param_names=(\"['a']\", \"['bb']\")" in repr(r)
    assert str(r) != repr(r)


def test_str_works_on_reports_the_fitters_build():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0))
    gm.compile()
    result = fit(gm, lambda p: (p["nodes"]["s"]["stiffness"] - 25.0) ** 2, n_iter=3,
                 hold_undetermined=False)
    assert str(result).startswith("FitResult: ")
    assert "['nodes']['s']['stiffness'] = " in str(result)
    report = fim(lambda p: jnp.stack([p["k"] * 2.0, p["k"] + p["c"]]),
                 {"k": jnp.float32(3.0), "c": jnp.float32(1.0)}, scale=None)
    text = str(report)
    assert text.startswith("FIMReport: rank 2 of 2 parameters (all determined)")
    assert "['k']" in text and "['c']" in text


def test_profile_report_str_is_untouched():
    """It already had one; this change does not add another."""
    assert "__str__" in vars(ProfileReport)
    assert str(ProfileReport(graph_name="g")).startswith("=== Graph Profile: g ===")
