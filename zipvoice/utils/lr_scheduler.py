# start zipvoice/utils/lr_scheduler.py
"""Learning rate schedulers for ZipVoice model training."""

# Copyright      2022  Xiaomi Corp.        (authors: Daniel Povey)
#
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

import structlog
import torch
from torch.optim import Optimizer

log = structlog.get_logger()


class LRScheduler(ABC):
    """Base-class for learning rate schedulers where the learning-rate depends on both the batch and the epoch."""

    def __init__(self, optimizer: Optimizer, verbose: bool = False):
        """Initialize the LRScheduler.

        Args:
            optimizer: The optimizer to schedule.
            verbose: If True, log learning rate changes.
        """
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            msg = f"{type(optimizer).__name__} is not an Optimizer"
            raise TypeError(msg)
        self.optimizer = optimizer
        self.verbose = verbose

        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])

        self.base_lrs = [group["base_lr"] for group in optimizer.param_groups]

        self.epoch = 0
        self.batch = 0

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            # the user might try to override the base_lr, so don't include this in the
            # state. previously they were included.
            # "base_lrs": self.base_lrs,
            "epoch": self.epoch,
            "batch": self.batch,
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # the things with base_lrs are a work-around for a previous problem
        # where base_lrs were written with the state dict.
        base_lrs = self.base_lrs
        self.__dict__.update(state_dict)
        self.base_lrs = base_lrs

    def get_last_lr(self) -> list[float]:
        """Return last computed learning rate by current scheduler.

        Will be a list of float.
        """
        return self._last_lr

    @abstractmethod
    def get_lr(self):
        """Compute and return the list of learning rates for all param groups.

        Must be overridden by subclasses. Should use self.epoch, self.batch,
        and self.base_lrs to calculate the current learning rates.
        """
        ...

    def step_batch(self, batch: int | None = None) -> None:
        """Step the batch index, or set it to the given value.

        Args:
            batch: If provided, sets the batch counter to this value (must be
                the absolute batch index from the start of training). If None,
                increments the counter by one.
        """
        if batch is not None:
            self.batch = batch
        else:
            self.batch = self.batch + 1
        self._set_lrs()

    def step_epoch(self, epoch: int | None = None):
        """Step the epoch index, or set it to the given value.

        Args:
            epoch: If provided, sets the epoch counter to this value and should
                be called at the start of the epoch. If None, increments by one
                and should be called at the end of the epoch.
        """
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch = self.epoch + 1
        self._set_lrs()

    def _set_lrs(self):
        values = self.get_lr()
        if len(values) != len(self.optimizer.param_groups):
            msg = f"Expected {len(self.optimizer.param_groups)} LR values, got {len(values)}"
            raise RuntimeError(msg)

        for i, data in enumerate(zip(self.optimizer.param_groups, values, strict=False)):
            param_group, lr = data
            param_group["lr"] = lr
            self.print_lr(self.verbose, i, lr)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, is_verbose, group, lr):
        """Display the current learning rate."""
        if is_verbose:
            log.warning(
                "adjusting_lr",
                epoch=self.epoch,
                batch=self.batch,
                group=group,
                lr=f"{lr:.4e}",
            )


class Eden(LRScheduler):
    """Eden learning rate scheduler.

    The basic formula (before warmup) is:
      lr = base_lr * (((batch**2 + lr_batches**2) / lr_batches**2) ** -0.25 *
                     (((epoch**2 + lr_epochs**2) / lr_epochs**2) ** -0.25)) * warmup
    where `warmup` increases from linearly 0.5 to 1 over `warmup_batches` batches
    and then stays constant at 1.

    If you don't have the concept of epochs, or one epoch takes a very long time,
    you can replace the notion of 'epoch' with some measure of the amount of data
    processed, e.g. hours of data or frames of data, with 'lr_epochs' being set to
    some measure representing "quite a lot of data": say, one fifth or one third
    of an entire training run, but it doesn't matter much.  You could also use
    Eden2 which has only the notion of batches.

    We suggest base_lr = 0.04 (passed to optimizer) if used with ScaledAdam

    Args:
        optimizer: the optimizer to change the learning rates on
        lr_batches: the number of batches after which we start significantly
              decreasing the learning rate, suggest 5000.
        lr_epochs: the number of epochs after which we start significantly
              decreasing the learning rate, suggest 6 if you plan to do e.g.
              20 to 40 epochs, but may need smaller number if dataset is huge
              and you will do few epochs.
    """

    def __init__(  # noqa: PLR0913
        self,
        optimizer: Optimizer,
        lr_batches: int | float,
        lr_epochs: int | float,
        warmup_batches: int | float = 500.0,
        warmup_start: float = 0.5,
        verbose: bool = False,
    ):
        """Initialize the Eden scheduler.

        Args:
            optimizer: The optimizer to schedule.
            lr_batches: Number of batches after which LR starts significantly decreasing.
            lr_epochs: Number of epochs after which LR starts significantly decreasing.
            warmup_batches: Number of batches over which LR warms up from warmup_start to 1.0.
            warmup_start: Starting LR multiplier at the beginning of warmup.
            verbose: If True, log learning rate changes.
        """
        super().__init__(optimizer, verbose)
        self.lr_batches = lr_batches
        self.lr_epochs = lr_epochs
        self.warmup_batches = warmup_batches

        if not (0.0 <= warmup_start <= 1.0):
            msg = f"warmup_start must be in [0.0, 1.0], got {warmup_start}"
            raise ValueError(msg)
        self.warmup_start = warmup_start

    def get_lr(self):
        """Compute and return learning rates for all parameter groups using the Eden formula."""
        factor = ((self.batch**2 + self.lr_batches**2) / self.lr_batches**2) ** -0.25 * (
            ((self.epoch**2 + self.lr_epochs**2) / self.lr_epochs**2) ** -0.25
        )
        warmup_factor = (
            1.0
            if self.batch >= self.warmup_batches
            else self.warmup_start + (1.0 - self.warmup_start) * (self.batch / self.warmup_batches)
            # else 0.5 + 0.5 * (self.batch / self.warmup_batches)
        )

        return [x * factor * warmup_factor for x in self.base_lrs]


class FixedLRScheduler(LRScheduler):
    """Fixed learning rate scheduler.

    Args:
        optimizer: the optimizer to change the learning rates on
    """

    def __init__(
        self,
        optimizer: Optimizer,
        verbose: bool = False,
    ):
        """Initialize the FixedLRScheduler.

        Args:
            optimizer: The optimizer to schedule.
            verbose: If True, log learning rate changes.
        """
        super().__init__(optimizer, verbose)

    def get_lr(self):
        """Return the fixed base learning rates for all parameter groups."""
        return list(self.base_lrs)


def _test_eden():
    m = torch.nn.Linear(100, 100)
    from zipvoice.utils.optim import ScaledAdam

    optim = ScaledAdam(m.parameters(), lr=0.03)

    scheduler = Eden(optim, lr_batches=100, lr_epochs=2, verbose=True)

    for epoch in range(10):
        scheduler.step_epoch(epoch)  # sets epoch to `epoch`

        for _step in range(20):
            x = torch.randn(200, 100).detach()
            x.requires_grad = True
            y = m(x)
            dy = torch.randn(200, 100).detach()
            f = (y * dy).sum()
            f.backward()

            optim.step()
            scheduler.step_batch()
            optim.zero_grad()

    log.info("last_lr", lr=scheduler.get_last_lr())
    log.info("state_dict", state_dict=scheduler.state_dict())


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import subprocess

    s = subprocess.check_output(  # noqa: S602
        "git status -uno .; git log -1; git diff HEAD .",  # noqa: S607
        shell=True,
    )
    log.info("git_status", output=str(s))

    _test_eden()
# end zipvoice/utils/lr_scheduler.py
