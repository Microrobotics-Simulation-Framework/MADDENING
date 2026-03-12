"""
Training callbacks for surrogate model training.

Provides a composable callback system for monitoring, early stopping,
model checkpointing, and learning rate scheduling during training.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax.numpy as jnp


class TrainingCallback:
    """Base class for training callbacks.

    Override any of the hook methods.  The ``trainer`` argument is the
    ``SurrogateTrainer`` instance, giving access to ``trainer.architecture``,
    ``trainer.dataset``, ``trainer.optimizer``, and (during training) the
    current weights.
    """

    should_stop: bool = False

    def on_train_begin(self, trainer, state: dict) -> None:
        """Called once before the first epoch."""

    def on_epoch_end(self, epoch: int, metrics: dict, state: dict) -> None:
        """Called at the end of each epoch.

        Parameters
        ----------
        epoch : int
            Zero-indexed epoch number.
        metrics : dict
            ``{"train_loss": float, "val_loss": float}``.
        state : dict
            Mutable training state with ``"weights"`` and ``"opt_state"``
            keys.  Callbacks may modify ``state["weights"]`` (e.g. to
            restore best weights).
        """

    def on_train_end(self, metrics: dict, state: dict) -> None:
        """Called once after the last epoch (or after early stop)."""


class EarlyStopping(TrainingCallback):
    """Stop training when a monitored metric has stopped improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    monitor : str
        Metric to monitor (default ``"val_loss"``).
    restore_best : bool
        If ``True``, restore the weights from the best epoch on stop.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        monitor: str = "val_loss",
        restore_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.restore_best = restore_best
        self.best_value = float("inf")
        self.best_epoch = 0
        self.wait = 0
        self._best_weights = None

    def on_train_begin(self, trainer, state):
        self.best_value = float("inf")
        self.best_epoch = 0
        self.wait = 0
        self.should_stop = False
        self._best_weights = None

    def on_epoch_end(self, epoch, metrics, state):
        current = metrics.get(self.monitor, metrics.get("val_loss"))
        if current < self.best_value - self.min_delta:
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best:
                import jax
                import jax.numpy as jnp
                self._best_weights = jax.tree.map(
                    lambda x: jnp.array(x) if hasattr(x, 'shape') else x,
                    state["weights"],
                )
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True

    def on_train_end(self, metrics, state):
        if self.restore_best and self._best_weights is not None:
            state["weights"] = self._best_weights


class ModelCheckpoint(TrainingCallback):
    """Save model weights periodically or when a metric improves.

    Parameters
    ----------
    path : str
        File path for the checkpoint.  Use ``"{epoch}"`` placeholder for
        per-epoch files.
    monitor : str
        Metric to monitor (default ``"val_loss"``).
    save_best_only : bool
        If ``True``, only save when the monitored metric improves.
    """

    def __init__(
        self,
        path: str = "surrogate_checkpoint.npz",
        monitor: str = "val_loss",
        save_best_only: bool = True,
    ):
        self.path = path
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_value = float("inf")
        self._architecture = None
        self._state_spec = None
        self._boundary_spec = None

    def on_train_begin(self, trainer, state):
        self.best_value = float("inf")
        self._architecture = trainer.architecture
        self._state_spec = trainer.dataset.state_spec
        self._boundary_spec = trainer.dataset.boundary_spec

    def on_epoch_end(self, epoch, metrics, state):
        current = metrics.get(self.monitor, metrics.get("val_loss"))
        if not self.save_best_only or current < self.best_value:
            self.best_value = current
            path = self.path.replace("{epoch}", str(epoch))
            from maddening.surrogates.checkpoint import save_weights
            save_weights(
                path, state["weights"],
                architecture=self._architecture,
                state_spec=self._state_spec,
                boundary_spec=self._boundary_spec,
                metadata={"epoch": epoch, **metrics},
            )


class LRSchedule(TrainingCallback):
    """Adjust the learning rate on a schedule.

    The schedule function receives the epoch number and returns a
    learning rate multiplier.  This works by scaling the optimizer's
    updates (not by replacing the optimizer).

    Parameters
    ----------
    schedule_fn : callable
        ``(epoch: int) -> float`` returning a learning rate multiplier.
        The optimizer's base LR is multiplied by this value.

    Example
    -------
    ::

        # Cosine decay from 1.0 to 0.01 over 100 epochs
        schedule = LRSchedule(
            lambda epoch: 0.01 + 0.99 * (1 + math.cos(math.pi * epoch / 100)) / 2
        )
    """

    def __init__(self, schedule_fn: Callable[[int], float]):
        self.schedule_fn = schedule_fn
        self._lr_multiplier = 1.0

    @property
    def lr_multiplier(self) -> float:
        return self._lr_multiplier

    def on_epoch_end(self, epoch, metrics, state):
        self._lr_multiplier = self.schedule_fn(epoch + 1)
