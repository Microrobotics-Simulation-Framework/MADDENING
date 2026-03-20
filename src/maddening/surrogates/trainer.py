"""
SurrogateTrainer -- Optax-based training loop for surrogate models.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.surrogates.architecture import SurrogateArchitecture
from maddening.surrogates.dataset import SurrogateDataset

try:
    import optax
except ImportError:
    optax = None

PyTree = Any


def _check_optax():
    if optax is None:
        raise ImportError(
            "SurrogateTrainer requires optax. "
            "Install with: pip install maddening[surrogates]"
        )


def mse_loss(pred: dict, target: dict) -> float:
    """Mean squared error across all state fields."""
    total = jnp.float32(0.0)
    count = 0
    for k in target:
        diff = pred[k] - target[k]
        total = total + jnp.sum(diff ** 2)
        count += diff.size
    return total / max(count, 1)


@dataclass
class TrainResult:
    """Result of surrogate training."""
    weights: Any
    architecture: SurrogateArchitecture
    train_losses: list
    val_losses: list
    state_spec: dict
    boundary_spec: dict

    def to_node(self, name, timestep, initial_values, integrator=None):
        """Create a SurrogateNode from this training result."""
        from maddening.surrogates.node import SurrogateNode
        return SurrogateNode(
            name=name,
            timestep=timestep,
            architecture=self.architecture,
            weights=self.weights,
            state_spec=self.state_spec,
            boundary_spec=self.boundary_spec,
            initial_values=initial_values,
            integrator=integrator,
        )

    def save(self, path: str, metadata: Optional[dict] = None) -> None:
        """Save weights and metadata to an NPZ file.

        Parameters
        ----------
        path : str
            Output file path.
        metadata : dict, optional
            Extra metadata to include.
        """
        from maddening.surrogates.checkpoint import save_weights
        meta = metadata or {}
        meta.setdefault("train_losses", self.train_losses[-5:])
        meta.setdefault("val_losses", self.val_losses[-5:])
        save_weights(
            path, self.weights,
            architecture=self.architecture,
            state_spec=self.state_spec,
            boundary_spec=self.boundary_spec,
            metadata=meta,
        )

    @staticmethod
    def load(path: str, architecture: SurrogateArchitecture, rng_key=None):
        """Load a TrainResult from a saved checkpoint.

        Parameters
        ----------
        path : str
            Input file path.
        architecture : SurrogateArchitecture
            Architecture instance of the same type used during saving.
        rng_key : PRNGKey, optional
            Random key for tree structure initialization.

        Returns
        -------
        TrainResult
        """
        from maddening.surrogates.checkpoint import load_train_result
        return load_train_result(path, architecture, rng_key)


@stability(StabilityLevel.EXPERIMENTAL)
class SurrogateTrainer:
    """Train a surrogate architecture on a SurrogateDataset.

    Parameters
    ----------
    architecture : SurrogateArchitecture
        The architecture to train.
    dataset : SurrogateDataset
        Training data.
    optimizer : optax optimizer, optional
        Defaults to ``optax.adam(1e-3)``.
    loss_fn : callable, optional
        ``(pred_dict, target_dict) -> scalar``.  Defaults to MSE.
    physics_loss_fn : callable, optional
        Additional physics-informed loss term.  Signature:
        ``(weights, state, boundary_inputs, pred, dt) -> scalar``.
    physics_loss_weight : float
        Weight for the physics loss (default 0.0).
    """

    def __init__(
        self,
        architecture: SurrogateArchitecture,
        dataset: SurrogateDataset,
        optimizer=None,
        loss_fn: Optional[Callable] = None,
        physics_loss_fn: Optional[Callable] = None,
        physics_loss_weight: float = 0.0,
    ):
        _check_optax()
        self.architecture = architecture
        self.dataset = dataset
        self.optimizer = optimizer or optax.adam(1e-3)
        self.loss_fn = loss_fn or mse_loss
        self.physics_loss_fn = physics_loss_fn
        self.physics_loss_weight = physics_loss_weight

    def train(
        self,
        n_epochs: int,
        batch_size: int = 32,
        rng_key=None,
        validation_split: float = 0.1,
        callback: Optional[Callable] = None,
        callbacks: Optional[list] = None,
    ) -> TrainResult:
        """Run the training loop.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.
        rng_key : PRNGKey, optional
            Random key for initialisation and shuffling.
        validation_split : float
            Fraction of data to hold out for validation.
        callback : callable, optional
            Called each epoch with ``(epoch, {"train_loss": ..., "val_loss": ...})``.
            Simple alternative to the ``callbacks`` list.
        callbacks : list[TrainingCallback], optional
            List of ``TrainingCallback`` instances for early stopping,
            checkpointing, LR scheduling, etc.

        Returns
        -------
        TrainResult
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        ds = self.dataset
        arch = self.architecture
        cbs = callbacks or []

        # Initialise weights
        init_key, shuffle_key = jax.random.split(rng_key)
        weights = arch.init_params(init_key, ds.state_spec, ds.boundary_spec)
        opt_state = self.optimizer.init(weights[0])  # optimise array part only

        # Split into train/val
        n_total = next(iter(ds.states.values())).shape[0]
        n_val = max(1, int(n_total * validation_split))
        n_train = n_total - n_val

        # Indices
        perm_key, shuffle_key = jax.random.split(shuffle_key)
        perm = jax.random.permutation(perm_key, n_total)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        def _index(d, idx):
            return {k: v[idx] for k, v in d.items()}

        train_states = _index(ds.states, train_idx)
        train_boundary = _index(ds.boundary_inputs, train_idx)
        train_targets = _index(ds.next_states, train_idx)
        val_states = _index(ds.states, val_idx)
        val_boundary = _index(ds.boundary_inputs, val_idx)
        val_targets = _index(ds.next_states, val_idx)

        dt = ds.dt
        loss_fn = self.loss_fn
        physics_loss_fn = self.physics_loss_fn
        physics_weight = self.physics_loss_weight

        arrays, static = weights

        # Check for LR schedule callbacks
        from maddening.surrogates.callbacks import LRSchedule
        lr_schedule_cbs = [cb for cb in cbs if isinstance(cb, LRSchedule)]

        # `static` is closed over (not passed through JIT) because it
        # contains non-array objects like activation functions.

        # Build per-sample loss
        def sample_loss(arrays, state, boundary, target):
            pred = arch.forward((arrays, static), state, boundary, dt)
            data_loss = loss_fn(pred, target)
            if physics_loss_fn is not None:
                return data_loss + physics_weight * physics_loss_fn(
                    (arrays, static), state, boundary, pred, dt
                )
            return data_loss

        # Batch loss: mean over batch
        def batch_loss(arrays, states_b, boundary_b, targets_b):
            # vmap over sample dimension
            losses = jax.vmap(
                lambda s, b, t: sample_loss(arrays, s, b, t)
            )(states_b, boundary_b, targets_b)
            return jnp.mean(losses)

        @jax.jit
        def train_step(arrays, opt_state, states_b, boundary_b, targets_b, lr_mult):
            loss, grads = jax.value_and_grad(batch_loss)(
                arrays, states_b, boundary_b, targets_b,
            )
            updates, new_opt_state = self.optimizer.update(grads, opt_state, arrays)
            # Apply LR scaling
            scaled_updates = jax.tree.map(lambda u: u * lr_mult, updates)
            new_arrays = optax.apply_updates(arrays, scaled_updates)
            return new_arrays, new_opt_state, loss

        @jax.jit
        def eval_loss(arrays, states_b, boundary_b, targets_b):
            return batch_loss(arrays, states_b, boundary_b, targets_b)

        train_losses = []
        val_losses = []

        # Callback state (mutable dict shared with callbacks)
        cb_state = {"weights": (arrays, static), "opt_state": opt_state}
        for cb in cbs:
            cb.on_train_begin(self, cb_state)

        for epoch in range(n_epochs):
            # Get current LR multiplier
            lr_mult = jnp.float32(1.0)
            for lr_cb in lr_schedule_cbs:
                lr_mult = lr_mult * lr_cb.lr_multiplier

            # Shuffle training data
            shuffle_key, epoch_key = jax.random.split(shuffle_key)
            epoch_perm = jax.random.permutation(epoch_key, n_train)
            shuffled_states = _index(train_states, epoch_perm)
            shuffled_boundary = _index(train_boundary, epoch_perm)
            shuffled_targets = _index(train_targets, epoch_perm)

            # Mini-batch training
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                idx = jnp.arange(start, end)
                s_b = _index(shuffled_states, idx)
                b_b = _index(shuffled_boundary, idx)
                t_b = _index(shuffled_targets, idx)

                arrays, opt_state, loss = train_step(
                    arrays, opt_state, s_b, b_b, t_b, lr_mult,
                )
                epoch_loss += float(loss)
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # Validation loss
            v_loss = float(eval_loss(arrays, val_states, val_boundary, val_targets))
            val_losses.append(v_loss)

            metrics = {"train_loss": avg_train_loss, "val_loss": v_loss}

            if callback is not None:
                callback(epoch, metrics)

            # Invoke training callbacks
            cb_state["weights"] = (arrays, static)
            cb_state["opt_state"] = opt_state
            for cb in cbs:
                cb.on_epoch_end(epoch, metrics, cb_state)
            # Callbacks may update weights (e.g. restore best)
            arrays, static = cb_state["weights"]

            # Check for early stopping
            if any(cb.should_stop for cb in cbs):
                break

        # Finalize callbacks
        final_metrics = {"train_loss": train_losses[-1], "val_loss": val_losses[-1]}
        cb_state["weights"] = (arrays, static)
        for cb in cbs:
            cb.on_train_end(final_metrics, cb_state)
        arrays, static = cb_state["weights"]

        return TrainResult(
            weights=(arrays, static),
            architecture=arch,
            train_losses=train_losses,
            val_losses=val_losses,
            state_spec=ds.state_spec,
            boundary_spec=ds.boundary_spec,
        )
