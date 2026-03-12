"""Tests for SimulationProvenance."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from maddening.core.provenance import SimulationProvenance


class TestSimulationProvenance:
    def test_default_construction(self):
        prov = SimulationProvenance()
        assert prov.python_version != ""
        assert prov.platform_info != ""
        assert prov.timestamp > 0

    def test_capture(self):
        prov = SimulationProvenance.capture()
        assert prov.jax_version != ""
        assert prov.jax_backend in ("cpu", "gpu", "tpu", "")
        assert prov.python_version != ""

    def test_capture_with_config(self):
        config = {"nodes": ["ball", "table"], "edges": 1}
        prov = SimulationProvenance.capture(graph_config=config)
        assert prov.graph_config == config

    def test_to_dict(self):
        prov = SimulationProvenance.capture()
        d = prov.to_dict()
        assert isinstance(d, dict)
        assert "jax_version" in d
        assert "python_version" in d
        assert "timestamp" in d

    def test_custom_fields(self):
        prov = SimulationProvenance.capture(random_seed=42)
        assert prov.random_seed == 42
