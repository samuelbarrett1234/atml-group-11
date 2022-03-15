from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class MultipleEarlyStopping(pl.callbacks.early_stopping.EarlyStopping):
    """Subclass of `pytorch_lightning.callbacks.early_stopping.EarlyStopping` that
    stops training early only if all of multiple conditions are met.

    The semantics are the obvious ones, see documentation of `EarlyStopping`.
    """
    def __init__(
        self,
        monitors: List[str],
        modes: List[str],
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        strict: bool = True,
        check_finite: bool = True,
        check_on_train_epoch_end: Optional[bool] = None,
    ):
        Callback.__init__(self)
        self.monitors = monitors
        self.patience = patience
        self.verbose = verbose
        self.modes = modes
        self.strict = strict
        self.check_finite = check_finite
        self.wait_count = 0
        self.stopped_epoch = 0
        self._check_on_train_epoch_end = check_on_train_epoch_end

        if any(mode not in self.mode_dict for mode in self.modes):
            raise MisconfigurationException(f"`modes` elements must be in {', '.join(self.mode_dict.keys())}")

        self.min_deltas = [min_delta if op == torch.gt else -min_delta for op in self.monitor_ops]
        torch_inf = torch.tensor(np.Inf)
        self.best_scores = [torch_inf if op == torch.lt else -torch_inf for op in self.monitor_ops]

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitors=self.monitors, modes=self.modes)

    def _validate_condition_metric(self, logs: Dict[str, float]) -> bool:
        for monitor in self.monitors:
            monitor_val = logs.get(monitor)
            error_msg = (
                f"Early stopping conditioned on metric `{monitor}` which is not available."
                " Pass in or modify your `EarlyStopping` callback to use any of the following:"
                f' `{"`, `".join(list(logs.keys()))}`'
            )
            if monitor_val is None:
                if self.strict:
                    raise RuntimeError(error_msg)
                if self.verbose > 0:
                    rank_zero_warn(error_msg, RuntimeWarning)
                return False
        return True

    @property
    def monitor_ops(self) -> List[Callable]:
        return [self.mode_dict[mode] for mode in self.modes]

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "best_scores": self.best_scores,
            "patience": self.patience,
        }

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]
    ) -> None:
        self.wait_count = callback_state["wait_count"]
        self.stopped_epoch = callback_state["stopped_epoch"]
        self.best_scores = callback_state["best_scores"]
        self.patience = callback_state["patience"]

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return

        currents = [logs[monitor].squeeze() for monitor in self.monitors]
        should_stop, reason = self._evaluate_stopping_criteria(currents)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason)

    def _evaluate_stopping_criteria(self, currents: List[torch.Tensor]) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = None

        for i, current in enumerate(currents):
            if self.check_finite and not torch.isfinite(current):
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitors[i]} = {current} is not finite."
                    f" Previous best value was {self.best_scores[i]:.3f}. Signaling Trainer to stop."
                )
                break
            if self.monitor_ops[i](current - self.min_deltas[i], self.best_scores[i].to(current.device)):
                should_stop = False
                reason = self._improvement_message(current, i)
                self.best_scores[i] = current
                self.wait_count = 0
                break
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metrics {self.monitors} did not improve in the last {self.wait_count} records."
                    f" Best scores: {self.best_scores}. Signaling Trainer to stop."
                )

        return should_stop, reason

    def _improvement_message(self, current: torch.Tensor, idx) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_scores[idx]):
            msg = (
                f"Metric {self.monitors[idx]} improved by {abs(self.best_scores[idx] - current):.3f} >="
                f" min_delta = {abs(self.min_deltas[idx])}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {self.monitors[idx]} improved. New best score: {current:.3f}"
        return msg


def sparse_dropout(x: torch.Tensor, p: float, training: bool = True):
    """Applies dropout to a sparse tensor.
    """
    assert x.is_sparse, "Input is not sparse"
    x = x.coalesce()
    new_values = F.dropout(x.values(), p=p, training=training)
    return torch.sparse_coo_tensor(values=new_values, 
                                   indices=x.indices(),
                                   size=x.size())