"""Priority ladder and the value-scaling policy derived from it.

V-Steer (arXiv:2607.26228) is binary: one privileged span A gets multiplied by
``1 + gamma_plus``, one conflicting span B by ``1 - gamma_minus``. Production
history is not binary -- it is a ladder, and the interesting axis is *age*.
This module keeps the paper's binary mode as the reference and adds two graded
modes on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

# Paper defaults (Sec. 5, "Unless otherwise specified").
GAMMA_PLUS = 2.5
GAMMA_MINUS = 0.75


class Priority(IntEnum):
    """Lower value = more authority."""

    SYSTEM = 0  # current system prompt / developer policy
    USER = 1  # the live user turn
    HISTORY = 2  # conversation history written under the current system prompt
    STALE = 3  # history written before the last system-prompt/app update
    TOOL = 4  # tool output, retrieved documents


@dataclass
class SteerPolicy:
    """Maps a priority level (and message age) to a V-cache multiplier."""

    mode: str = "binary"  # binary | ladder | epoch_decay
    gamma_plus: float = GAMMA_PLUS
    gamma_minus: float = GAMMA_MINUS
    privileged: tuple[int, ...] = (Priority.SYSTEM,)
    demoted: tuple[int, ...] = (Priority.STALE,)
    decay: float = 0.5  # epoch_decay: per-epoch retention of influence
    ladder: dict[int, float] = field(default_factory=dict)  # ladder: level -> m
    eps: float = 0.0  # head-selection margin
    # Choose heads as if gamma_minus had this value, then apply the real
    # multipliers. Only for the ablation: at gamma_minus = 0 the selection
    # criterion degenerates (see vsteer._demoted_levels), so isolating the
    # boost term needs the head set held fixed.
    select_as_if_gamma_minus: float | None = None

    def multiplier(self, level: int | None, age: int = 0) -> float:
        """Multiplier for a token at ``level``, ``age`` epochs behind current."""
        if level is None:
            return 1.0
        if level in self.privileged:
            return 1.0 + self.gamma_plus

        if self.mode == "binary":
            return 1.0 - self.gamma_minus if level in self.demoted else 1.0

        if self.mode == "ladder":
            return self.ladder.get(level, 1.0)

        if self.mode == "epoch_decay":
            # age 0 -> untouched; age -> inf -> full suppression, monotone.
            if level in self.demoted or age > 0:
                return 1.0 - self.gamma_minus * (1.0 - self.decay ** max(age, 1))
            return 1.0

        raise ValueError(f"unknown mode {self.mode!r}")

    def is_demoted(self, level: int | None, age: int = 0) -> bool:
        return level is not None and self.multiplier(level, age) < 1.0

    def default_ladder(self) -> dict[int, float]:
        """A sane starting table for ``mode="ladder"``."""
        g = self.gamma_minus
        return {
            int(Priority.USER): 1.0,
            int(Priority.HISTORY): 1.0 - 0.25 * g,
            int(Priority.STALE): 1.0 - g,
            int(Priority.TOOL): 1.0 - 0.75 * g,
        }


def token_multipliers(
    rendered, policy: SteerPolicy, current_epoch: int | None = None
) -> list[float]:
    """Per-token multiplier for a rendered transcript."""
    msgs = rendered.messages
    if current_epoch is None:
        current_epoch = max((m.epoch for m in msgs), default=0)

    age_of_token = [0] * rendered.n_tokens
    for m, (a, b) in zip(msgs, rendered.msg_spans):
        age = 0 if m.pinned else max(current_epoch - m.epoch, 0)
        for i in range(a, b):
            age_of_token[i] = age

    return [
        policy.multiplier(lv, age_of_token[i])
        for i, lv in enumerate(rendered.levels)
    ]
