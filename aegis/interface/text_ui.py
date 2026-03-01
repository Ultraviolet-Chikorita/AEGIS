from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class FrameState:
    cooldown_remaining_game_hours: float
    cooldown_total_game_hours: float
    host_resistance: float
    host_willpower: float


class TextInterface:
    def __init__(self, width: int = 80, height: int = 40) -> None:
        self.width = width
        self.height = height
        self.narrative_buffer: deque[str] = deque(maxlen=100)

    def render_frame(self, frame_state: FrameState) -> None:
        print("\n" + "=" * self.width)
        self._render_possession_status(frame_state)
        print("-" * self.width)
        visible_lines = self.height - 8
        for line in list(self.narrative_buffer)[-visible_lines:]:
            print(line)

    def _render_possession_status(self, frame_state: FrameState) -> None:
        total = max(0.001, frame_state.cooldown_total_game_hours)
        pct = 1.0 - (frame_state.cooldown_remaining_game_hours / total)
        pct = max(0.0, min(1.0, pct))

        cooldown_bar = self._bar(pct, 20)
        resistance_bar = self._bar(frame_state.host_resistance, 10)
        willpower_bar = self._bar(frame_state.host_willpower, 10)

        print(
            "POSSESSION COOLDOWN: "
            f"{cooldown_bar} ({frame_state.cooldown_remaining_game_hours:.2f} game hours remaining)"
        )
        print(
            f"HOST RESISTANCE: {resistance_bar} {int(frame_state.host_resistance * 100)}%    "
            f"WILLPOWER: {willpower_bar} {int(frame_state.host_willpower * 100)}%"
        )

    def stream_narrative(self, text: str, char_delay: float = 0.0) -> None:
        if char_delay <= 0:
            print(text)
        else:
            for char in text:
                print(char, end="", flush=True)
                time.sleep(char_delay)
            print()
        self.narrative_buffer.append(text)

    def get_player_command(self) -> str:
        return input("\n[Voice command ready - type fallback] > ").strip()

    @staticmethod
    def _bar(ratio: float, width: int) -> str:
        ratio = max(0.0, min(1.0, ratio))
        filled = int(ratio * width)
        return "█" * filled + "░" * (width - filled)
