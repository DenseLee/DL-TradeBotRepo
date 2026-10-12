"""Live-safe confirmation gate for the experimental C1 pyramiding arm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ConfirmedSpreeConfig:
    max_positions: int = 4
    minimum_open_r: float = 0.25
    minimum_minutes_between_adds: float = 5.0


@dataclass(frozen=True)
class OpenLeg:
    ticket: int
    direction: int
    current_r: float


@dataclass(frozen=True)
class ConfirmedSpreeDecision:
    allow_add: bool
    reason: str
    sequence: int
    minimum_leg_r: float | None


def evaluate_confirmed_add(
    legs: Iterable[OpenLeg],
    candidate_direction: int,
    minutes_since_last_add: float | None,
    config: ConfirmedSpreeConfig | None = None,
) -> ConfirmedSpreeDecision:
    """Require an already-profitable, same-direction pyramid before adding."""
    cfg = config or ConfirmedSpreeConfig()
    open_legs = list(legs)
    sequence = len(open_legs)
    if not open_legs:
        return ConfirmedSpreeDecision(True, "spree_seed", 0, None)
    if sequence >= int(cfg.max_positions):
        return ConfirmedSpreeDecision(False, "spree_max_positions", sequence, None)
    if any(int(leg.direction) != int(candidate_direction) for leg in open_legs):
        return ConfirmedSpreeDecision(False, "spree_direction_conflict", sequence, None)
    if (
        minutes_since_last_add is not None
        and minutes_since_last_add < float(cfg.minimum_minutes_between_adds)
    ):
        return ConfirmedSpreeDecision(False, "spree_add_cooldown", sequence, None)
    minimum_leg_r = min(float(leg.current_r) for leg in open_legs)
    if minimum_leg_r < float(cfg.minimum_open_r):
        return ConfirmedSpreeDecision(
            False, "spree_open_profit_unconfirmed", sequence, minimum_leg_r
        )
    return ConfirmedSpreeDecision(True, "spree_confirmed_add", sequence, minimum_leg_r)


def smoke() -> None:
    cfg = ConfirmedSpreeConfig()
    assert evaluate_confirmed_add([], 1, None, cfg).allow_add
    one = [OpenLeg(1, 1, 0.30)]
    assert evaluate_confirmed_add(one, 1, 5.0, cfg).allow_add
    assert evaluate_confirmed_add(one, -1, 5.0, cfg).reason == "spree_direction_conflict"
    assert evaluate_confirmed_add(one, 1, 4.9, cfg).reason == "spree_add_cooldown"
    assert evaluate_confirmed_add([OpenLeg(1, 1, 0.24)], 1, 5.0, cfg).reason == "spree_open_profit_unconfirmed"
    full = [OpenLeg(index, 1, 0.30) for index in range(4)]
    assert evaluate_confirmed_add(full, 1, 5.0, cfg).reason == "spree_max_positions"


if __name__ == "__main__":
    smoke()
    print("C1 confirmed spree smoke passed")
