"""Tests for v0.2 #2 surrogates subpackage restructure.

The restructure splits ``maddening.surrogates`` into thematic
subpackages — ``primitives/``, ``weights/``, ``training/``, ``replace/``
— while preserving every legacy import path for at least one minor
version.  These tests pin both the new paths and the legacy paths so
neither regresses.
"""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Primitives subpackage scaffolding
# ---------------------------------------------------------------------------


class TestPrimitives:
    def test_subpackage_exists(self):
        mod = importlib.import_module("maddening.surrogates.primitives")
        assert mod is not None

    def test_dunder_all_is_list(self):
        from maddening.surrogates import primitives
        assert isinstance(primitives.__all__, list)

    def test_does_not_explode_when_imported_without_optional_deps(self):
        # The package itself must not pull in equinox/optax/MIME — it's
        # the *future-home* for primitives.  A bare ``import`` must work
        # even on a minimum install (CPU JAX only).
        mod = importlib.import_module("maddening.surrogates.primitives")
        # Re-import as a sanity check that no module-level side-effect ran
        importlib.reload(mod)


# ---------------------------------------------------------------------------
# Weights subpackage re-export
# ---------------------------------------------------------------------------


class TestWeightsSubpackage:
    def test_save_weights_available(self):
        from maddening.surrogates.weights import save_weights
        assert callable(save_weights)

    def test_load_weights_available(self):
        from maddening.surrogates.weights import load_weights
        assert callable(load_weights)

    def test_load_train_result_available(self):
        from maddening.surrogates.weights import load_train_result
        assert callable(load_train_result)

    def test_new_path_yields_same_object_as_old_path(self):
        from maddening.surrogates.weights import save_weights as new_save
        from maddening.surrogates.checkpoint import save_weights as old_save
        assert new_save is old_save

    def test_dunder_all(self):
        from maddening.surrogates import weights
        assert set(weights.__all__) == {"save_weights", "load_weights", "load_train_result"}


# ---------------------------------------------------------------------------
# Training subpackage re-export (lazy)
# ---------------------------------------------------------------------------


class TestTrainingSubpackage:
    def test_trainer_lazy_import(self):
        from maddening.surrogates.training import SurrogateTrainer
        from maddening.surrogates.trainer import SurrogateTrainer as direct
        assert SurrogateTrainer is direct

    def test_callbacks_lazy_import(self):
        from maddening.surrogates.training import EarlyStopping, ModelCheckpoint
        from maddening.surrogates.callbacks import EarlyStopping as direct_es
        assert EarlyStopping is direct_es

    def test_physics_losses_lazy_import(self):
        from maddening.surrogates.training import residual_loss, smoothness_loss
        from maddening.surrogates.physics_losses import residual_loss as direct
        assert residual_loss is direct

    def test_unknown_attribute_raises_attributeerror(self):
        from maddening.surrogates import training
        with pytest.raises(AttributeError, match="no attribute"):
            _ = training.NonExistentClass

    def test_dir_lists_expected_names(self):
        from maddening.surrogates import training
        names = set(dir(training))
        # Every public name in __all__ must appear in dir() for IDE
        # autocomplete to discover them.
        for n in training.__all__:
            assert n in names

    def test_repeat_access_uses_cached_attribute(self):
        from maddening.surrogates import training
        a = training.SurrogateTrainer
        b = training.SurrogateTrainer
        assert a is b


# ---------------------------------------------------------------------------
# Replace subpackage (was a leaf module in v0.1, now a subpackage)
# ---------------------------------------------------------------------------


class TestReplaceSubpackage:
    def test_replace_node_importable(self):
        from maddening.surrogates.replace import replace_node
        assert callable(replace_node)

    def test_legacy_top_level_import_still_works(self):
        # `from maddening.surrogates import replace_node` is the
        # path used by maddening.surrogates.__init__ and most callers.
        from maddening.surrogates import replace_node
        assert callable(replace_node)

    def test_module_path_resolves_to_subpackage(self):
        import maddening.surrogates.replace as repkg
        # In v0.1 this resolved to a .py module; in v0.2 it's a subpackage
        assert hasattr(repkg, "__path__")  # package marker

    def test_core_module_present(self):
        from maddening.surrogates.replace import _core
        assert hasattr(_core, "replace_node")


# ---------------------------------------------------------------------------
# Backwards compatibility: every v0.1 import path still works
# ---------------------------------------------------------------------------


class TestBackwardsCompat:
    @pytest.mark.parametrize("path,attr", [
        ("maddening.surrogates.architecture", "SurrogateArchitecture"),
        ("maddening.surrogates.node", "SurrogateNode"),
        ("maddening.surrogates.node", "euler_integrator"),
        ("maddening.surrogates.node", "rk4_integrator"),
        ("maddening.surrogates.dataset", "SurrogateDataset"),
        ("maddening.surrogates.dataset", "DatasetGenerator"),
        ("maddening.surrogates.replace", "replace_node"),
        ("maddening.surrogates.checkpoint", "save_weights"),
        ("maddening.surrogates.checkpoint", "load_weights"),
        ("maddening.surrogates.trainer", "SurrogateTrainer"),
        ("maddening.surrogates.trainer", "TrainResult"),
        ("maddening.surrogates.callbacks", "TrainingCallback"),
        ("maddening.surrogates.callbacks", "EarlyStopping"),
        ("maddening.surrogates.physics_losses", "residual_loss"),
        ("maddening.surrogates.physics_losses", "composite_loss"),
    ])
    def test_v01_path_still_works(self, path, attr):
        mod = importlib.import_module(path)
        assert hasattr(mod, attr), f"{path}.{attr} missing"


# ---------------------------------------------------------------------------
# Top-level surrogates package re-exports unchanged
# ---------------------------------------------------------------------------


class TestTopLevelReExports:
    def test_top_level_replace_node(self):
        from maddening.surrogates import replace_node
        assert callable(replace_node)

    def test_top_level_surrogate_architecture(self):
        from maddening.surrogates import SurrogateArchitecture
        assert SurrogateArchitecture is not None

    def test_top_level_surrogate_node(self):
        from maddening.surrogates import SurrogateNode
        assert SurrogateNode is not None

    def test_top_level_dataset(self):
        from maddening.surrogates import SurrogateDataset, DatasetGenerator
        assert SurrogateDataset is not None
        assert DatasetGenerator is not None

    def test_top_level_lazy_trainer(self):
        from maddening.surrogates import SurrogateTrainer  # lazy via __getattr__
        assert SurrogateTrainer is not None
