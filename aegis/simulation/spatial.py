from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class EntityPosition:
    entity_id: str
    x: float
    y: float
    entity_type: str = "npc"


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def nearby(
    entities: Iterable[EntityPosition],
    origin_x: float,
    origin_y: float,
    radius: float,
) -> list[EntityPosition]:
    out: list[EntityPosition] = []
    for entity in entities:
        if distance((entity.x, entity.y), (origin_x, origin_y)) <= radius:
            out.append(entity)
    return out
