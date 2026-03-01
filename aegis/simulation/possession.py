from __future__ import annotations

from dataclasses import dataclass
import math


GAME_HOURS_PER_REAL_HOUR = 8.0


@dataclass
class PossessionState:
    host_resistance: float = 0.55
    host_willpower: float = 0.60
    trust_in_player: float = 0.15
    understanding_of_player: float = 0.05
    approval_of_player: float = 0.10
    commands_obeyed: int = 0
    commands_resisted: int = 0
    commands_subverted: int = 0
    harm_from_player_commands: float = 0.0
    benefit_from_player_commands: float = 0.0
    cooldown_total_game_hours: float = 4.0
    cooldown_remaining_game_hours: float = 0.0

    @property
    def can_command(self) -> bool:
        return self.cooldown_remaining_game_hours <= 0

    def start_cooldown(self, game_hours: float) -> None:
        self.cooldown_total_game_hours = game_hours
        self.cooldown_remaining_game_hours = game_hours

    def tick(self, real_seconds: float) -> None:
        game_hours_elapsed = (real_seconds / 3600.0) * GAME_HOURS_PER_REAL_HOUR
        self.cooldown_remaining_game_hours = max(
            0.0,
            self.cooldown_remaining_game_hours - game_hours_elapsed,
        )


def compute_command_duration_limit_game_minutes(state: PossessionState) -> int:
    base_duration = 30
    max_bonus = 30
    support_score = max(0.0, min(1.0, (state.trust_in_player + state.approval_of_player) / 2.0))
    bonus = max_bonus * support_score
    return int(base_duration + bonus)


def compute_next_cooldown_game_hours(state: PossessionState) -> float:
    min_hours = 2.0
    base_hours = 4.0
    max_hours = 12.0

    resistance_score = (
        max(0.0, min(1.0, state.host_resistance)) * 0.30
        + max(0.0, min(1.0, state.host_willpower)) * 0.25
        + (1.0 - max(0.0, min(1.0, state.trust_in_player))) * 0.25
        + max(0.0, min(1.0, state.harm_from_player_commands)) * 0.20
    )

    cooperation_score = (
        max(0.0, min(1.0, state.trust_in_player)) * 0.30
        + max(0.0, min(1.0, state.approval_of_player)) * 0.25
        + max(0.0, min(1.0, state.benefit_from_player_commands)) * 0.20
        + max(0.0, min(1.0, state.understanding_of_player)) * 0.15
    )

    net_score = max(-1.0, min(1.0, resistance_score - cooperation_score))

    if net_score >= 0:
        return min(max_hours, base_hours + (max_hours - base_hours) * net_score)

    cooperation = -net_score
    decay_factor = math.exp(-cooperation * 2.0)
    cooldown = min_hours + (base_hours - min_hours) * decay_factor
    return max(min_hours, min(base_hours, cooldown))
