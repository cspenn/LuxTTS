# start zipvoice/models/modules/scaling/_piecewise.py
# Copyright    2022-2025  Xiaomi Corp.        (authors: Daniel Povey
#                                                       Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

"""Piecewise-linear schedule utilities."""

import random

import structlog
import torch

log = structlog.get_logger()


class PiecewiseLinear:
    """Piecewise linear function, from float to float, specified as nonempty list of (x,y) pairs.

    x values in order.  x values <[initial x] or >[final x] are map to
    [initial y], [final y] respectively.
    """

    def __init__(self, *args):
        if len(args) < 1:
            msg = f"Expected at least 1 argument, got {len(args)}"
            raise ValueError(msg)
        if len(args) == 1 and isinstance(args[0], PiecewiseLinear):
            self.pairs = list(args[0].pairs)
        else:
            self.pairs = [(float(x), float(y)) for x, y in args]
        for x, y in self.pairs:
            if not isinstance(x, (float, int)):
                msg = f"Expected float or int for x, got {type(x)}"
                raise TypeError(msg)
            if not isinstance(y, (float, int)):
                msg = f"Expected float or int for y, got {type(y)}"
                raise TypeError(msg)

        for i in range(len(self.pairs) - 1):
            if not self.pairs[i + 1][0] > self.pairs[i][0]:
                msg = (
                    f"x values must be strictly increasing: "
                    f"index={i}, pairs[i]={self.pairs[i]}, pairs[i+1]={self.pairs[i + 1]}"
                )
                raise ValueError(msg)

    def __str__(self):
        # e.g. 'PiecewiseLinear((0., 10.), (100., 0.))'
        return f"PiecewiseLinear({str(self.pairs)[1:-1]})"

    def __call__(self, x):
        if x <= self.pairs[0][0]:
            return self.pairs[0][1]
        elif x >= self.pairs[-1][0]:
            return self.pairs[-1][1]
        else:
            cur_x, cur_y = self.pairs[0]
            for i in range(1, len(self.pairs)):
                next_x, next_y = self.pairs[i]
                if x >= cur_x and x <= next_x:
                    return cur_y + (next_y - cur_y) * (x - cur_x) / (next_x - cur_x)
                cur_x, cur_y = next_x, next_y
            msg = "Unreachable: x value not covered by piecewise linear pairs"
            raise RuntimeError(msg)

    def __mul__(self, alpha):
        return PiecewiseLinear(*[(x, y * alpha) for x, y in self.pairs])

    def __add__(self, x):
        if isinstance(x, (float, int)):
            return PiecewiseLinear(*[(p[0], p[1] + x) for p in self.pairs])
        s, x = self.get_common_basis(x)
        return PiecewiseLinear(*[(sp[0], sp[1] + xp[1]) for sp, xp in zip(s.pairs, x.pairs, strict=False)])

    def max(self, x):
        if isinstance(x, (float, int)):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(*[(sp[0], max(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs, strict=False)])

    def min(self, x):
        if isinstance(x, (float, int)):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(*[(sp[0], min(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs, strict=False)])

    def __eq__(self, other):
        return self.pairs == other.pairs

    def get_common_basis(self, p: "PiecewiseLinear", include_crossings: bool = False):
        """Returns (self_mod, p_mod) which are equivalent piecewise linear functions.

        The returned functions are equivalent to self and p, but with the same x values.

          p: the other piecewise linear function
          include_crossings: if true, include in the x values positions
              where the functions indicate by this and p crosss.
        """
        if not isinstance(p, PiecewiseLinear):
            msg = f"Expected PiecewiseLinear, got {type(p)}"
            raise TypeError(msg)

        # get sorted x-values without repetition.
        x_vals = sorted(set([x for x, _ in self.pairs] + [x for x, _ in p.pairs]))
        y_vals1 = [self(x) for x in x_vals]
        y_vals2 = [p(x) for x in x_vals]

        if include_crossings:
            extra_x_vals = []
            for i in range(len(x_vals) - 1):
                if (y_vals1[i] > y_vals2[i]) != (y_vals1[i + 1] > y_vals2[i + 1]):
                    # if the two lines in this subsegment potentially cross each other..
                    diff_cur = abs(y_vals1[i] - y_vals2[i])
                    diff_next = abs(y_vals1[i + 1] - y_vals2[i + 1])
                    # `pos`, between 0 and 1, gives the relative x position,
                    # with 0 being x_vals[i] and 1 being x_vals[i+1].
                    pos = diff_cur / (diff_cur + diff_next)
                    extra_x_val = x_vals[i] + pos * (x_vals[i + 1] - x_vals[i])
                    extra_x_vals.append(extra_x_val)
            if len(extra_x_vals) > 0:
                x_vals = sorted(set(x_vals + extra_x_vals))
        y_vals1 = [self(x) for x in x_vals]
        y_vals2 = [p(x) for x in x_vals]
        return (
            PiecewiseLinear(*zip(x_vals, y_vals1, strict=False)),
            PiecewiseLinear(*zip(x_vals, y_vals2, strict=False)),
        )


class ScheduledFloat(torch.nn.Module):
    """A float-like value that changes with training batch count.

    This object is a torch.nn.Module only because we want it to show up in
    [top_level module].modules(); it does not have a working forward() function.
    You are supposed to cast it to float, as in, float(parent_module.whatever), and use
    it as something like a dropout prob.

    It is a floating point value whose value changes depending on the batch count of the
    training loop.  It is a piecewise linear function where you specify the (x,y) pairs
    in sorted order on x; x corresponds to the batch index.  For batch-index values
    before the first x or after the last x, we just use the first or last y value.

    Example:
       self.dropout = ScheduledFloat((0.0, 0.2), (4000.0, 0.0), default=0.0)

    `default` is used when self.batch_count is not set or not in training mode or in
     torch.jit scripting mode.
    """

    def __init__(self, *args, default: float = 0.0):
        super().__init__()
        # self.batch_count and self.name will be written to in the training loop.
        self.batch_count = None
        self.name = None
        self.default = default
        self.schedule = PiecewiseLinear(*args)

    def extra_repr(self) -> str:
        return f"batch_count={self.batch_count}, schedule={str(self.schedule.pairs[1:-1])}"

    def __float__(self):
        batch_count = self.batch_count
        if batch_count is None or not self.training or torch.jit.is_scripting() or torch.jit.is_tracing():
            return float(self.default)
        else:
            ans = self.schedule(self.batch_count)
            if random.random() < 0.0002:  # noqa: S311
                log.debug(
                    "scheduled_float",
                    name=self.name,
                    batch_count=self.batch_count,
                    ans=ans,
                )
            return ans

    def __add__(self, x):
        if isinstance(x, (float, int)):
            return ScheduledFloat(self.schedule + x, default=self.default)
        else:
            return ScheduledFloat(self.schedule + x.schedule, default=self.default + x.default)

    def max(self, x):
        if isinstance(x, (float, int)):
            return ScheduledFloat(self.schedule.max(x), default=self.default)
        else:
            return ScheduledFloat(
                self.schedule.max(x.schedule),
                default=max(self.default, x.default),
            )


FloatLike = float | ScheduledFloat


class CutoffEstimator:
    """Estimates cutoffs so that a specified proportion of items will be above the cutoff.

    p is the proportion of items that should be above the cutoff.
    """

    def __init__(self, p: float):
        self.p = p
        # total count of items
        self.count = 0
        # total count of items that were above the cutoff
        self.count_above = 0
        # initial cutoff value
        self.cutoff = 0

    def __call__(self, x: float) -> bool:
        """Returns true if x is above the cutoff."""
        ans = x > self.cutoff
        self.count += 1
        if ans:
            self.count_above += 1
        cur_p = self.count_above / self.count
        delta_p = cur_p - self.p
        if (delta_p > 0) == ans:
            q = abs(delta_p)
            self.cutoff = x * q + self.cutoff * (1 - q)
        return ans


# end zipvoice/models/modules/scaling/_piecewise.py
