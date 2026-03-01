from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import random
import re
from typing import Any
from uuid import uuid4

from aegis.interface.text_ui import FrameState
from aegis.prompts import AUTONOMOUS_BEHAVIOR_SYSTEM_PROMPT, HOST_RESPONSE_SYSTEM_PROMPT
from aegis.simulation.possession import (
    PossessionState,
    compute_command_duration_limit_game_minutes,
    compute_next_cooldown_game_hours,
)


REASONING_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "compliance_style": {"type": "string"},
        "interpretation": {"type": "string"},
        "likely_success": {"type": "number"},
        "effort_level": {"type": "number"},
        "internal_reaction": {"type": "string"},
        "spoken_words": {"type": "string"},
        "pre_actions": {"type": "array", "items": {"type": "string"}},
        "during_actions": {"type": "array", "items": {"type": "string"}},
        "post_actions": {"type": "array", "items": {"type": "string"}},
        "relationship_changes": {
            "type": "object",
            "properties": {
                "trust_delta": {"type": "number"},
                "resistance_delta": {"type": "number"},
                "understanding_delta": {"type": "number"},
            },
        },
        "explored_trait_ids": {"type": "array", "items": {"type": "string"}},
        "goals": {"type": "array", "items": {"type": "string"}},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "tone_progression": {"type": "array"},
        "voice_cluster": {"type": "string"},
    },
}

COMBAT_BEAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "left_action": {"type": "string"},
        "right_action": {"type": "string"},
        "tone": {"type": "string"},
        "momentum": {"type": "string"},
    },
}

EXTREME_RESISTANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "should_attempt_self_harm": {"type": "boolean"},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
        "desperation_evidence": {"type": "array", "items": {"type": "string"}},
        "protective_evidence": {"type": "array", "items": {"type": "string"}},
    },
}

DESTINATION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "should_set_destination": {"type": "boolean"},
        "location_node_id": {"type": "string"},
        "source_tool": {"type": "string"},
        "location_hint": {"type": "string"},
        "rationale": {"type": "string"},
        "confidence": {"type": "number"},
    },
}

EPISODE_RELATIONSHIP_DELTA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "friendship_delta": {"type": "number"},
        "trust_delta": {"type": "number"},
        "respect_delta": {"type": "number"},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
}

INTERACTION_EPISODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "should_use_generated_episode": {"type": "boolean"},
        "summary_override": {"type": "string"},
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker_id": {"type": "string"},
                    "spoken_words": {"type": "string"},
                    "action": {"type": "string"},
                    "tone": {"type": "string"},
                    "emotional_shift": {"type": "string"},
                },
            },
        },
        "graph_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "owner_id": {"type": "string"},
                    "source_tool": {"type": "string"},
                    "note": {"type": "string"},
                },
            },
        },
    },
}

INTERACTION_VARIANT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "should_use_dynamic": {"type": "boolean"},
        "kind": {"type": "string"},
        "tone": {"type": "string"},
        "summary": {"type": "string"},
        "details": {"type": "array", "items": {"type": "string"}},
        "friendship_delta": {"type": "number"},
        "trust_delta": {"type": "number"},
        "respect_delta": {"type": "number"},
        "emotional_tags": {"type": "array", "items": {"type": "string"}},
        "intensity": {"type": "number"},
    },
}

AUTONOMOUS_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_summary": {"type": "string"},
        "next_goal": {"type": "string"},
        "target_entity_id": {"type": "string"},
        "target_region": {"type": "string"},
        "intent_alignment": {"type": "string"},
    },
}


logger = logging.getLogger(__name__)
POSSESSION_NODE_ID = "possession-001"
FORBIDDEN_SPEECH_PATTERNS = [
    r"voice in (my|the) head",
    r"(controlling|commanding|forcing) me",
    r"being (possessed|controlled|commanded)",
    r"must obey",
    r"can't (resist|refuse|stop)",
    r"entity|presence|inside me",
]


class SimulationEngine:
    def __init__(
        self,
        graph,
        reasoning,
        speech,
        ui,
        voice,
        graph_tools,
        tick_hz: int = 10,
    ) -> None:
        self.graph = graph
        self.reasoning = reasoning
        self.speech = speech
        self.ui = ui
        self.voice = voice
        self.graph_tools = graph_tools
        self.tick_hz = tick_hz
        self._tick_seconds = 1.0 / max(1, tick_hz)
        self._host_id = "host"
        self.possession = PossessionState()
        self._was_command_ready = False
        self._world_time_seconds = 0.0
        self._tick_index = 0
        self._graph_sync_timer = 0.0
        self._autonomy_tick_accumulator = 0.0
        self._autonomy_interval_seconds = 7.5
        self._interaction_last_seen: dict[str, float] = {}
        self._interaction_cooldown_seconds = 18.0
        self._social_memory_cooldown_seconds = 26.0
        self._active_social_interactions: dict[str, dict[str, Any]] = {}
        self._max_active_social_interactions = 6
        self._max_new_social_interactions_per_tick = 3
        self._pending_social_interaction_tasks: dict[str, asyncio.Task[dict[str, Any] | None]] = {}
        self._pending_social_interaction_meta: dict[str, dict[str, Any]] = {}
        self._max_pending_social_plans = 4
        self._social_planning_timeout_seconds = 5.0
        self._social_planning_semaphore = asyncio.Semaphore(2)
        self._motion_locked_entity_ids: set[str] = set()
        self._llm_interaction_last_time = -9999.0
        self._llm_interaction_min_interval_seconds = 8.0
        self._last_autonomy_summary = ""
        self._autonomy_repeat_count = 0
        self._region_width = 1400.0
        self._region_height = 980.0
        self._map_km_width = 10.0
        self._map_km_height = 10.0
        self._regions = self._build_regions()
        self._buildings = self._build_buildings(self._regions)
        self._roads = self._build_roads(self._regions, self._buildings)
        self._settlements = self._build_settlements(self._regions)
        self._markets = self._build_markets(self._buildings)
        self._black_markets = self._build_black_markets(self._buildings)
        self._hunting_areas = self._build_hunting_areas(self._regions)
        self._economy_tick_accumulator = 0.0
        self._economy_interval_seconds = 6.0
        self._combat_tick_accumulator = 0.0
        self._combat_interval_seconds = 0.05
        self._command_task: asyncio.Task[None] | None = None
        self._social_emit_task: asyncio.Task[None] | None = None
        self._active_command_text = ""
        self._pending_combat_narration_tasks: dict[str, asyncio.Task[dict[str, Any] | None]] = {}
        self._active_combat_encounters: dict[str, dict[str, Any]] = {}
        self._world_entities = self._build_world_entities()
        logger.info("SimulationEngine initialized (tick_hz=%s, tick_seconds=%.3f)", self.tick_hz, self._tick_seconds)

    def _build_regions(self) -> list[dict[str, Any]]:
        region_specs = [
            {
                "id": "ironmarket",
                "label": "Ironmarket",
                "biome": "urban_market",
                "seed_x": 220.0,
                "seed_y": 220.0,
                "color": "#2f4052",
                "vertices": [[40.0, 70.0], [430.0, 40.0], [500.0, 265.0], [360.0, 420.0], [70.0, 360.0]],
            },
            {
                "id": "charter-heights",
                "label": "Charter Heights",
                "biome": "civic",
                "seed_x": 690.0,
                "seed_y": 200.0,
                "color": "#3c4f6b",
                "vertices": [[430.0, 40.0], [900.0, 30.0], [980.0, 210.0], [760.0, 390.0], [500.0, 265.0]],
            },
            {
                "id": "shipward",
                "label": "Shipward",
                "biome": "docks",
                "seed_x": 1170.0,
                "seed_y": 210.0,
                "color": "#274051",
                "vertices": [[900.0, 30.0], [1360.0, 70.0], [1365.0, 360.0], [1030.0, 430.0], [980.0, 210.0]],
            },
            {
                "id": "moss-cross",
                "label": "Moss Cross",
                "biome": "residential",
                "seed_x": 300.0,
                "seed_y": 610.0,
                "color": "#3d3a54",
                "vertices": [[70.0, 360.0], [360.0, 420.0], [470.0, 640.0], [300.0, 900.0], [30.0, 920.0], [20.0, 520.0]],
            },
            {
                "id": "sanctuary-rise",
                "label": "Sanctuary Rise",
                "biome": "temple",
                "seed_x": 760.0,
                "seed_y": 590.0,
                "color": "#2f4d41",
                "vertices": [[360.0, 420.0], [760.0, 390.0], [1030.0, 430.0], [980.0, 840.0], [640.0, 960.0], [470.0, 640.0]],
            },
            {
                "id": "green-basin",
                "label": "Green Basin",
                "biome": "garden",
                "seed_x": 1190.0,
                "seed_y": 700.0,
                "color": "#31594d",
                "vertices": [[1030.0, 430.0], [1365.0, 360.0], [1380.0, 920.0], [980.0, 840.0]],
            },
        ]

        regions: list[dict[str, Any]] = []
        for spec in region_specs:
            vertices = spec["vertices"]
            xs = [float(point[0]) for point in vertices]
            ys = [float(point[1]) for point in vertices]
            regions.append(
                {
                    **spec,
                    "vertices": [{"x": float(point[0]), "y": float(point[1])} for point in vertices],
                    "x": min(xs),
                    "y": min(ys),
                    "w": max(xs) - min(xs),
                    "h": max(ys) - min(ys),
                }
            )

        return regions

    async def _plan_command_destination_from_graph(
        self, command: str, reasoning_output: dict[str, Any]
    ) -> dict[str, Any] | None:
        try:
            plan = await self.reasoning.run_with_graph_access(
                system_prompt=(
                    "Identify whether the host should move toward a specific world location for this command. "
                    "Use graph tools and only propose a location that exists in graph nodes "
                    "(Building, Settlement, Region). Return JSON only. "
                    "If setting a destination, you must provide location_node_id and source_tool."
                ),
                task_prompt=(
                    "Determine a destination hint if this command/action sequence implies travel. "
                    "If not, set should_set_destination=false. "
                    "Do not invent places."
                ),
                seed_context={
                    "command": command,
                    "interpretation": str(reasoning_output.get("interpretation", "")),
                    "pre_actions": reasoning_output.get("pre_actions", []),
                    "during_actions": reasoning_output.get("during_actions", []),
                    "post_actions": reasoning_output.get("post_actions", []),
                    "goals": reasoning_output.get("goals", []),
                },
                output_schema=DESTINATION_PLAN_SCHEMA,
                graph_tools=self.graph_tools,
                max_rounds=4,
            )
        except Exception:
            logger.debug("Destination planning failed; skipping graph destination update", exc_info=True)
            return None

        if not isinstance(plan, dict) or not bool(plan.get("should_set_destination", False)):
            return None

        confidence = float(plan.get("confidence", 0.5) or 0.5)
        if confidence < 0.2:
            return None

        location_node_id = str(plan.get("location_node_id", "")).strip()
        source_tool = str(plan.get("source_tool", "")).strip().lower()
        if not location_node_id:
            return None
        if source_tool not in {"inspect", "edges_out", "edges_in", "traverse", "semantic_search", "find_nodes"}:
            return None

        resolved_by_id = await self._resolve_graph_location_node_id(location_node_id)
        if resolved_by_id is not None:
            return resolved_by_id

        location_hint = str(plan.get("location_hint", "")).strip()
        if location_hint:
            return await self._resolve_graph_location_hint(location_hint)
        return None

    async def _resolve_graph_location_node_id(self, node_id: str) -> dict[str, Any] | None:
        node_key = node_id.strip()
        if not node_key:
            return None

        rows = await self.graph.query(
            """
            MATCH (loc {id: $node_id})
            WHERE loc:Building OR loc:Settlement OR loc:Region
            RETURN labels(loc) AS labels, loc.id AS id, coalesce(loc.name, loc.label, loc.id) AS name
            LIMIT 1
            """,
            {"node_id": node_key},
        )
        if not rows:
            return None

        row = rows[0]
        labels = [str(item) for item in row.get("labels", [])]
        resolved_id = str(row.get("id", node_key))
        name = str(row.get("name", resolved_id))

        def _region_position(region_id: str) -> tuple[float, float] | None:
            region = next((item for item in self._regions if str(item.get("id", "")) == region_id), None)
            if region is None:
                return None
            return (
                float(region.get("seed_x", region.get("x", self._region_width / 2.0))),
                float(region.get("seed_y", region.get("y", self._region_height / 2.0))),
            )

        def _settlement_position(settlement_id: str) -> tuple[float, float] | None:
            settlement = next((item for item in self._settlements if str(item.get("id", "")) == settlement_id), None)
            if settlement is None:
                return None
            return (float(settlement.get("center_x", 0.0)), float(settlement.get("center_y", 0.0)))

        def _building_position(building_id: str) -> tuple[float, float] | None:
            building = next((item for item in self._buildings if str(item.get("id", "")) == building_id), None)
            if building is None:
                return None
            footprint = building.get("footprint", [])
            if not isinstance(footprint, list) or not footprint:
                return None
            xs = [float(point.get("x", 0.0)) for point in footprint if isinstance(point, dict)]
            ys = [float(point.get("y", 0.0)) for point in footprint if isinstance(point, dict)]
            if not xs or not ys:
                return None
            return (sum(xs) / len(xs), sum(ys) / len(ys))

        if "Building" in labels:
            pos = _building_position(resolved_id)
            if pos is not None:
                return {"id": resolved_id, "name": name, "kind": "building", "x": pos[0], "y": pos[1]}
        if "Settlement" in labels:
            pos = _settlement_position(resolved_id)
            if pos is not None:
                return {"id": resolved_id, "name": name, "kind": "settlement", "x": pos[0], "y": pos[1]}
        if "Region" in labels:
            pos = _region_position(resolved_id)
            if pos is not None:
                return {"id": resolved_id, "name": name, "kind": "region", "x": pos[0], "y": pos[1]}

        return None

    async def _resolve_graph_location_hint(self, location_hint: str) -> dict[str, Any] | None:
        hint = location_hint.strip().lower()
        if not hint:
            return None

        rows = await self.graph.query(
            """
            MATCH (loc)
            WHERE loc:Building OR loc:Settlement OR loc:Region
            WITH
                loc,
                toLower($hint) AS hint,
                toLower(coalesce(loc.name, loc.label, loc.id, '')) AS canonical
            WHERE canonical CONTAINS hint OR hint CONTAINS canonical
            RETURN
                labels(loc) AS labels,
                loc.id AS id,
                coalesce(loc.name, loc.label, loc.id) AS name,
                canonical
            ORDER BY
                CASE
                    WHEN canonical = hint THEN 0
                    WHEN canonical STARTS WITH hint THEN 1
                    ELSE 2
                END,
                size(canonical) ASC
            LIMIT 6
            """,
            {"hint": hint},
        )
        if not rows:
            return None

        def _region_position(region_id: str) -> tuple[float, float] | None:
            region = next((item for item in self._regions if str(item.get("id", "")) == region_id), None)
            if region is None:
                return None
            return (
                float(region.get("seed_x", region.get("x", self._region_width / 2.0))),
                float(region.get("seed_y", region.get("y", self._region_height / 2.0))),
            )

        def _settlement_position(settlement_id: str) -> tuple[float, float] | None:
            settlement = next((item for item in self._settlements if str(item.get("id", "")) == settlement_id), None)
            if settlement is None:
                return None
            return (float(settlement.get("center_x", 0.0)), float(settlement.get("center_y", 0.0)))

        def _building_position(building_id: str) -> tuple[float, float] | None:
            building = next((item for item in self._buildings if str(item.get("id", "")) == building_id), None)
            if building is None:
                return None
            footprint = building.get("footprint", [])
            if not isinstance(footprint, list) or not footprint:
                return None
            xs = [float(point.get("x", 0.0)) for point in footprint if isinstance(point, dict)]
            ys = [float(point.get("y", 0.0)) for point in footprint if isinstance(point, dict)]
            if not xs or not ys:
                return None
            return (sum(xs) / len(xs), sum(ys) / len(ys))

        for row in rows:
            labels = [str(item) for item in row.get("labels", [])]
            node_id = str(row.get("id", ""))
            name = str(row.get("name", node_id))

            if "Building" in labels:
                pos = _building_position(node_id)
                if pos is not None:
                    return {"id": node_id, "name": name, "kind": "building", "x": pos[0], "y": pos[1]}
            if "Settlement" in labels:
                pos = _settlement_position(node_id)
                if pos is not None:
                    return {"id": node_id, "name": name, "kind": "settlement", "x": pos[0], "y": pos[1]}
            if "Region" in labels:
                pos = _region_position(node_id)
                if pos is not None:
                    return {"id": node_id, "name": name, "kind": "region", "x": pos[0], "y": pos[1]}

        return None

    def _build_roads(self, regions: list[dict[str, Any]], buildings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        region_by_id = {str(region["id"]): region for region in regions}
        road_specs = [
            ("road-spine-north", "North Spine", ["ironmarket", "charter-heights", "shipward"]),
            ("road-spine-south", "South Spine", ["moss-cross", "sanctuary-rise", "green-basin"]),
            ("road-artery-west", "West Artery", ["ironmarket", "moss-cross"]),
            ("road-artery-east", "East Artery", ["shipward", "green-basin"]),
            ("road-midline", "Midline", ["charter-heights", "sanctuary-rise"]),
        ]
        roads: list[dict[str, Any]] = []
        for road_id, label, region_chain in road_specs:
            waypoints: list[dict[str, float]] = []
            for region_id in region_chain:
                region = region_by_id.get(region_id)
                if region is None:
                    continue
                waypoints.append({"x": float(region["seed_x"]), "y": float(region["seed_y"])})
            roads.append(
                {
                    "id": road_id,
                    "label": label,
                    "surface_type": "cobble",
                    "waypoints": waypoints,
                    "connects_region_ids": [str(item) for item in region_chain],
                    "connects_building_ids": [],
                }
            )

        for building in buildings:
            region_id = str(building.get("region_id", ""))
            region = region_by_id.get(region_id)
            if region is None:
                continue
            centroid = self._building_centroid(building)
            if centroid is None:
                continue
            roads.append(
                {
                    "id": f"road-access-{building['id']}",
                    "label": f"Access: {building.get('name', building['id'])}",
                    "surface_type": "stone-path",
                    "waypoints": [
                        {"x": float(region["seed_x"]), "y": float(region["seed_y"])},
                        {"x": float(centroid[0]), "y": float(centroid[1])},
                    ],
                    "connects_region_ids": [region_id],
                    "connects_building_ids": [str(building.get("id", ""))],
                }
            )
        return roads

    def _build_settlements(self, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        settlements: list[dict[str, Any]] = []
        for region in regions:
            settlements.append(
                {
                    "id": f"settlement-{region['id']}",
                    "name": f"{region['label']} Ward",
                    "region_id": region["id"],
                    "center_x": float(region["seed_x"]),
                    "center_y": float(region["seed_y"]),
                }
            )
        return settlements

    def _build_buildings(self, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        building_specs = [
            ("bld-inkhall", "Inkhall Exchange", "guildhall", "ironmarket"),
            ("bld-watchtower-east", "East Watchtower", "watchtower", "charter-heights"),
            ("bld-tide-yards", "Tide Yards", "warehouse", "shipward"),
            ("bld-moss-tenements", "Moss Tenements", "residence", "moss-cross"),
            ("bld-sanctum", "Sanctum of Ash", "temple", "sanctuary-rise"),
            ("bld-greenhouse", "Green Basin Conservatory", "garden-hall", "green-basin"),
        ]
        region_by_id = {str(region["id"]): region for region in regions}
        buildings: list[dict[str, Any]] = []
        for building_id, name, building_type, region_id in building_specs:
            region = region_by_id.get(region_id)
            if region is None:
                continue
            cx = float(region["seed_x"])
            cy = float(region["seed_y"])
            buildings.append(
                {
                    "id": building_id,
                    "name": name,
                    "type": building_type,
                    "region_id": region_id,
                    "footprint": [
                        {"x": cx - 24.0, "y": cy - 18.0},
                        {"x": cx + 24.0, "y": cy - 18.0},
                        {"x": cx + 24.0, "y": cy + 18.0},
                        {"x": cx - 24.0, "y": cy + 18.0},
                    ],
                }
            )
        return buildings

    def _building_centroid(self, building: dict[str, Any]) -> tuple[float, float] | None:
        footprint = building.get("footprint", [])
        if not isinstance(footprint, list) or not footprint:
            return None
        xs = [float(point.get("x", 0.0)) for point in footprint if isinstance(point, dict)]
        ys = [float(point.get("y", 0.0)) for point in footprint if isinstance(point, dict)]
        if not xs or not ys:
            return None
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _build_markets(self, buildings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        type_priority = {"guildhall", "warehouse", "garden-hall"}
        markets: list[dict[str, Any]] = []
        for building in buildings:
            if str(building.get("type", "")) not in type_priority:
                continue
            centroid = self._building_centroid(building)
            if centroid is None:
                continue
            markets.append(
                {
                    "id": f"market-{building['id']}",
                    "name": f"{building.get('name', 'Market')} Bazaar",
                    "building_id": str(building["id"]),
                    "region_id": str(building.get("region_id", "")),
                    "x": float(centroid[0]),
                    "y": float(centroid[1]),
                    "tax_rate": 0.12,
                    "price_index": 1.0,
                    "supply": {"monster_hide": 8.0, "monster_meat": 10.0, "bone_shard": 7.0},
                    "demand": {"monster_hide": 11.0, "monster_meat": 9.0, "bone_shard": 8.0},
                    "tax_collected": 0.0,
                    "gross_volume": 0.0,
                    "active": True,
                }
            )
        return markets[:3]

    def _build_black_markets(self, buildings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dens = [
            building
            for building in buildings
            if str(building.get("type", "")) in {"residence", "warehouse", "watchtower"}
        ]
        if not dens:
            return []
        chosen = dens[0]
        centroid = self._building_centroid(chosen)
        if centroid is None:
            return []
        return [
            {
                "id": f"black-market-{chosen['id']}",
                "name": "Shadow Exchange",
                "building_id": str(chosen["id"]),
                "region_id": str(chosen.get("region_id", "")),
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "tax_rate": 0.0,
                "price_index": 1.08,
                "risk": 0.24,
                "supply": {"monster_hide": 3.0, "monster_meat": 2.0, "bone_shard": 4.0},
                "demand": {"monster_hide": 14.0, "monster_meat": 8.0, "bone_shard": 12.0},
                "gross_volume": 0.0,
                "active": True,
            }
        ]

    def _build_hunting_areas(self, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        target_region = next((item for item in regions if str(item.get("id")) == "green-basin"), None)
        if target_region is None:
            target_region = regions[-1] if regions else None
        if target_region is None:
            return []

        return [
            {
                "id": "hunt-outer-glen",
                "name": "Outer Glen Hunting Grounds",
                "region_id": str(target_region.get("id", "")),
                "x": float(target_region.get("seed_x", self._region_width * 0.8)),
                "y": float(target_region.get("seed_y", self._region_height * 0.78)),
                "radius": 115.0,
                "success_bonus": 0.12,
                "active": True,
            }
        ]

    def _serialize_region_for_graph(self, region: dict[str, Any]) -> dict[str, Any]:
        row = dict(region)
        row["vertices_json"] = json.dumps(region.get("vertices", []))
        row.pop("vertices", None)
        return row

    def _serialize_road_for_graph(self, road: dict[str, Any]) -> dict[str, Any]:
        row = dict(road)
        row["waypoints_json"] = json.dumps(road.get("waypoints", []))
        row.pop("waypoints", None)
        return row

    def _serialize_building_for_graph(self, building: dict[str, Any]) -> dict[str, Any]:
        row = dict(building)
        row["footprint_json"] = json.dumps(building.get("footprint", []))
        row.pop("footprint", None)
        return row

    def _normalize_region(self, props: dict[str, Any]) -> dict[str, Any]:
        row = dict(props)
        vertices_value = row.get("vertices")
        if not isinstance(vertices_value, list):
            vertices_json = row.get("vertices_json")
            if isinstance(vertices_json, str) and vertices_json.strip():
                try:
                    vertices_value = json.loads(vertices_json)
                except json.JSONDecodeError:
                    vertices_value = []
            else:
                vertices_value = []

        vertices: list[dict[str, float]] = []
        for point in vertices_value:
            if isinstance(point, dict):
                vertices.append(
                    {
                        "x": float(point.get("x", 0.0)),
                        "y": float(point.get("y", 0.0)),
                    }
                )
        row["vertices"] = vertices
        return row

    def _normalize_road(self, props: dict[str, Any]) -> dict[str, Any]:
        row = dict(props)
        waypoints_value = row.get("waypoints")
        if not isinstance(waypoints_value, list):
            waypoints_json = row.get("waypoints_json")
            if isinstance(waypoints_json, str) and waypoints_json.strip():
                try:
                    waypoints_value = json.loads(waypoints_json)
                except json.JSONDecodeError:
                    waypoints_value = []
            else:
                waypoints_value = []

        waypoints: list[dict[str, float]] = []
        for point in waypoints_value:
            if isinstance(point, dict):
                waypoints.append(
                    {
                        "x": float(point.get("x", 0.0)),
                        "y": float(point.get("y", 0.0)),
                    }
                )
        row["waypoints"] = waypoints
        return row

    def _build_world_entities(self) -> list[dict[str, Any]]:
        rng = random.Random(37)
        base_item_pool = [
            "bread",
            "water flask",
            "lockpick",
            "healing salve",
            "ledger scrap",
            "coin pouch",
            "rope",
            "chalk",
            "seal ring",
            "worn map",
        ]
        entities: list[dict[str, Any]] = [
            {
                "id": "host",
                "label": "Mira Vale",
                "kind": "host",
                "x": 705.0,
                "y": 570.0,
                "vx": 0.0,
                "vy": 0.0,
                "patrol_target_x": 705.0,
                "patrol_target_y": 570.0,
                "speed": 34.0,
                "faction": "locals",
                "coins": 14,
                "inventory": ["journal", "silver pin", "worn map", "water flask"],
            }
        ]

        npc_names = [
            "Mara Quill", "Captain Rho", "Brother Cael", "Ilya Fen", "Doran Pike",
            "Nessa Thorn", "Kest Arlow", "Vera Mott", "Jalen Pryce", "Torin Bale",
            "Sera Moss", "Bram Coil", "Lysa Venn", "Quen Hart", "Tamsin Reed",
            "Orin Voss", "Neri Cask", "Hollis Wren", "Pella Night", "Rook Dar",
            "Mikra Dune", "Sable Kerr", "Edric Lume", "Yara Flint", "Corin Vale",
        ]
        factions = ["guild", "watch", "locals"]

        for index, name in enumerate(npc_names, start=1):
            faction = factions[(index - 1) % len(factions)]
            x = rng.uniform(60.0, self._region_width - 60.0)
            y = rng.uniform(60.0, self._region_height - 60.0)
            target_x = rng.uniform(60.0, self._region_width - 60.0)
            target_y = rng.uniform(60.0, self._region_height - 60.0)
            entities.append(
                {
                    "id": f"npc-{index:03d}",
                    "label": name,
                    "kind": "npc",
                    "x": x,
                    "y": y,
                    "vx": 0.0,
                    "vy": 0.0,
                    "patrol_target_x": target_x,
                    "patrol_target_y": target_y,
                    "speed": rng.uniform(39.0, 56.0),
                    "faction": faction,
                    "coins": int(rng.uniform(4, 28)),
                    "inventory": [
                        base_item_pool[int(rng.uniform(0, len(base_item_pool)))],
                        base_item_pool[int(rng.uniform(0, len(base_item_pool)))],
                    ],
                }
            )
        return entities

    def _build_psychology_rows(self) -> dict[str, list[dict[str, Any]]]:
        faction_archetypes: dict[str, dict[str, Any]] = {
            "guild": {
                "trait": ("calculating", "Pragmatic and transactional under pressure"),
                "goal": "Secure leverage through trade intelligence",
                "belief": "Information is worth more than coin.",
                "capability": ("commerce", "Negotiation"),
            },
            "watch": {
                "trait": ("duty-bound", "Prioritizes order and chain of command"),
                "goal": "Keep district unrest contained",
                "belief": "Order prevents catastrophe.",
                "capability": ("security", "Tactical Awareness"),
            },
            "locals": {
                "trait": ("community-rooted", "Protective of neighbors and routines"),
                "goal": "Protect local community ties",
                "belief": "People survive together, not alone.",
                "capability": ("social", "Local Insight"),
            },
        }

        trait_rows: list[dict[str, Any]] = []
        capability_rows: list[dict[str, Any]] = []
        goal_rows: list[dict[str, Any]] = []
        belief_rows: list[dict[str, Any]] = []

        base_tick = int(self._world_time_seconds * self.tick_hz)
        for entity in self._world_entities:
            owner_id = str(entity["id"])
            faction = str(entity.get("faction", "locals"))
            archetype = faction_archetypes.get(faction, faction_archetypes["locals"])
            is_host = owner_id == self._host_id

            primary_trait_label, primary_trait_desc = archetype["trait"]
            host_trait_label = "eldritch-friction" if is_host else primary_trait_label
            host_trait_desc = (
                "Persistent cognitive dissonance from external compulsion"
                if is_host
                else primary_trait_desc
            )

            trait_rows.append(
                {
                    "id": f"trait-{owner_id}-core",
                    "owner_id": owner_id,
                    "label": host_trait_label,
                    "description": host_trait_desc,
                    "triggers": ["threat", "compulsion", "conflict"],
                    "soothers": ["safety", "allies", "control"],
                    "felt_as": "tension",
                    "expressed_as": "guarded speech",
                    "promotes": ["self-preservation"],
                    "inhibits": ["reckless trust"],
                    "embedding": [],
                    "intensity": 0.62 if is_host else 0.48,
                    "baseline_intensity": 0.45,
                    "formed_at_tick": base_tick,
                    "last_activated_tick": base_tick,
                    "text": f"{host_trait_label} {host_trait_desc}",
                    "sim_managed": True,
                }
            )

            capability_domain, capability_label = archetype["capability"]
            capability_rows.append(
                {
                    "id": f"capability-{owner_id}-primary",
                    "owner_id": owner_id,
                    "domain": capability_domain,
                    "label": capability_label,
                    "current_level": 0.64 if is_host else 0.58,
                    "name": capability_label,
                    "text": f"{capability_domain} {capability_label}",
                    "sim_managed": True,
                }
            )

            goal_rows.append(
                {
                    "id": f"goal-{owner_id}-core",
                    "owner_id": owner_id,
                    "description": (
                        "Maintain autonomy while surviving possession windows"
                        if is_host
                        else archetype["goal"]
                    ),
                    "priority": 0.82 if is_host else 0.66,
                    "status": "active",
                    "name": "Core Goal",
                    "text": archetype["goal"],
                    "sim_managed": True,
                }
            )

            belief_rows.append(
                {
                    "id": f"belief-{owner_id}-core",
                    "owner_id": owner_id,
                    "content": (
                        "The controlling presence has a hidden agenda."
                        if is_host
                        else archetype["belief"]
                    ),
                    "confidence": 0.69 if is_host else 0.63,
                    "text": archetype["belief"],
                    "sim_managed": True,
                }
            )

        return {
            "traits": trait_rows,
            "capabilities": capability_rows,
            "goals": goal_rows,
            "beliefs": belief_rows,
        }

    async def _sync_psychology_graph(self) -> None:
        rows = self._build_psychology_rows()

        await self.graph.query(
            """
            UNWIND $traits AS trait
            MERGE (t:Trait {id: trait.id})
            SET t += trait
            """,
            {"traits": rows["traits"]},
        )
        await self.graph.query(
            """
            MATCH (t:Trait {sim_managed: true})
            WHERE NOT t.id IN $trait_ids
            DETACH DELETE t
            """,
            {"trait_ids": [str(item["id"]) for item in rows["traits"]]},
        )
        await self.graph.query(
            """
            UNWIND $traits AS trait
            MATCH (n:NPC {id: trait.owner_id}), (t:Trait {id: trait.id})
            MERGE (n)-[:HAS_TRAIT]->(t)
            """,
            {"traits": rows["traits"]},
        )

        await self.graph.query(
            """
            UNWIND $capabilities AS capability
            MERGE (c:Capability {id: capability.id})
            SET c += capability
            """,
            {"capabilities": rows["capabilities"]},
        )
        await self.graph.query(
            """
            MATCH (c:Capability {sim_managed: true})
            WHERE NOT c.id IN $capability_ids
            DETACH DELETE c
            """,
            {"capability_ids": [str(item["id"]) for item in rows["capabilities"]]},
        )
        await self.graph.query(
            """
            UNWIND $capabilities AS capability
            MATCH (n:NPC {id: capability.owner_id}), (c:Capability {id: capability.id})
            MERGE (n)-[:HAS_CAPABILITY]->(c)
            """,
            {"capabilities": rows["capabilities"]},
        )

        await self.graph.query(
            """
            UNWIND $goals AS goal
            MERGE (g:Goal {id: goal.id})
            SET g += goal
            """,
            {"goals": rows["goals"]},
        )
        await self.graph.query(
            """
            MATCH (g:Goal {sim_managed: true})
            WHERE NOT g.id IN $goal_ids
            DETACH DELETE g
            """,
            {"goal_ids": [str(item["id"]) for item in rows["goals"]]},
        )
        await self.graph.query(
            """
            UNWIND $goals AS goal
            MATCH (n:NPC {id: goal.owner_id}), (g:Goal {id: goal.id})
            MERGE (n)-[:HAS_GOAL]->(g)
            """,
            {"goals": rows["goals"]},
        )

        await self.graph.query(
            """
            UNWIND $beliefs AS belief
            MERGE (b:Belief {id: belief.id})
            SET b += belief
            """,
            {"beliefs": rows["beliefs"]},
        )
        await self.graph.query(
            """
            MATCH (b:Belief {sim_managed: true})
            WHERE NOT b.id IN $belief_ids
            DETACH DELETE b
            """,
            {"belief_ids": [str(item["id"]) for item in rows["beliefs"]]},
        )
        await self.graph.query(
            """
            UNWIND $beliefs AS belief
            MATCH (n:NPC {id: belief.owner_id}), (b:Belief {id: belief.id})
            MERGE (n)-[:HAS_BELIEF]->(b)
            """,
            {"beliefs": rows["beliefs"]},
        )

        experience_rows = [
            {
                "id": f"exp-{entity['id']}-recent",
                "owner_id": str(entity["id"]),
                "description": (
                    "Recent possession aftershock and attempted self-stabilization"
                    if str(entity["id"]) == self._host_id
                    else f"Routine district pressure response for {entity.get('faction', 'locals')}"
                ),
                "outcome": "ongoing",
                "text": "recent lived experience",
                "sim_managed": True,
            }
            for entity in self._world_entities
        ]
        await self.graph.query(
            """
            UNWIND $experiences AS experience
            MERGE (e:Experience {id: experience.id})
            SET e += experience
            """,
            {"experiences": experience_rows},
        )
        await self.graph.query(
            """
            MATCH (e:Experience {sim_managed: true})
            WHERE NOT e.id IN $experience_ids
            DETACH DELETE e
            """,
            {"experience_ids": [str(item["id"]) for item in experience_rows]},
        )
        await self.graph.query(
            """
            UNWIND $experiences AS experience
            MATCH (n:NPC {id: experience.owner_id}), (e:Experience {id: experience.id})
            MERGE (n)-[:HAS_EXPERIENCE]->(e)
            """,
            {"experiences": experience_rows},
        )

        gossip_rows = [
            {
                "id": f"gossip-{entity['id']}",
                "content": f"Rumors swirl around {entity.get('label', entity.get('id'))} in {entity.get('faction', 'locals')} circles.",
                "importance": 0.35,
                "text": str(entity.get("label", "unknown")),
                "sim_managed": True,
            }
            for entity in self._world_entities
            if str(entity.get("id")) != self._host_id
        ]
        await self.graph.query(
            """
            UNWIND $gossip AS gossip
            MERGE (g:GossipItem {id: gossip.id})
            SET g += gossip
            """,
            {"gossip": gossip_rows},
        )
        await self.graph.query(
            """
            MATCH (g:GossipItem {sim_managed: true})
            WHERE NOT g.id IN $gossip_ids
            DETACH DELETE g
            """,
            {"gossip_ids": [str(item["id"]) for item in gossip_rows]},
        )

        await self.graph.query(
            """
            MATCH (h:NPC {id: $host_id}), (p:PossessionState {id: $possession_id})
            OPTIONAL MATCH (h)-[:HAS_TRAIT]->(t:Trait)
            WITH h, p, collect(t) AS traits
            FOREACH (trait IN traits | MERGE (trait)-[:ACTIVATED_BY_POSSESSION]->(p))
            """,
            {"host_id": self._host_id, "possession_id": POSSESSION_NODE_ID},
        )

        await self.graph.query(
            """
            MATCH (a:Trait {sim_managed: true}), (b:Trait {sim_managed: true})
            WHERE a.id < b.id AND a.owner_id = b.owner_id
            MERGE (a)-[:TENSIONS_WITH]->(b)
            """
        )
        await self.graph.query(
            """
            MATCH (t:Trait {sim_managed: true})
            WITH collect(t) AS traits
            FOREACH (t IN traits | FOREACH (u IN traits |
                FOREACH (_ IN CASE WHEN t.id <> u.id AND t.owner_id = u.owner_id THEN [1] ELSE [] END |
                    MERGE (t)-[:AMPLIFIES]->(u)
                )
            ))
            """
        )

        speech_rows = [
            {
                "id": f"speech-state-{entity['id']}",
                "text": "",
                "voice_cluster": "guarded_suspicious" if str(entity["id"]) == self._host_id else "casual_friendly",
                "owner_id": str(entity["id"]),
                "sim_managed": True,
            }
            for entity in self._world_entities
        ]
        await self.graph.query(
            """
            UNWIND $speech AS speech
            MERGE (s:SpeechContent {id: speech.id})
            SET s += speech
            """,
            {"speech": speech_rows},
        )
        await self.graph.query(
            """
            MATCH (s:SpeechContent {sim_managed: true})
            WHERE NOT s.id IN $speech_ids
            DETACH DELETE s
            """,
            {"speech_ids": [str(item["id"]) for item in speech_rows]},
        )
        await self.graph.query(
            """
            UNWIND $speech AS speech
            MATCH (n:NPC {id: speech.owner_id}), (s:SpeechContent {id: speech.id})
            MERGE (n)-[:USES_VOICE_PROFILE]->(s)
            """,
            {"speech": speech_rows},
        )

        await self._sync_inventory_graph()
        await self._sync_economy_graph()

    async def _sync_inventory_graph(self) -> None:
        inventory_rows: list[dict[str, Any]] = []
        for entity in self._world_entities:
            owner_id = str(entity.get("id"))
            for index, item_name in enumerate(entity.get("inventory", [])):
                label = str(item_name).strip()
                if not label:
                    continue
                inventory_rows.append(
                    {
                        "id": f"inv-{owner_id}-{index}-{self._safe_inventory_token(label)}",
                        "owner_id": owner_id,
                        "name": label,
                        "quantity": 1,
                        "rarity": "common",
                        "sim_managed": True,
                    }
                )

        await self.graph.query(
            """
            UNWIND $items AS item
            MERGE (i:InventoryItem {id: item.id})
            SET i += item
            """,
            {"items": inventory_rows},
        )
        await self.graph.query(
            """
            MATCH (i:InventoryItem {sim_managed: true})
            WHERE NOT i.id IN $item_ids
            DETACH DELETE i
            """,
            {"item_ids": [str(item["id"]) for item in inventory_rows]},
        )
        await self.graph.query(
            """
            UNWIND $items AS item
            MATCH (n:NPC {id: item.owner_id}), (i:InventoryItem {id: item.id})
            MERGE (n)-[:HAS_ITEM]->(i)
            """,
            {"items": inventory_rows},
        )

    def _update_market_pressure(self) -> None:
        venues = [*self._markets, *[market for market in self._black_markets if bool(market.get("active", False))]]
        for venue in venues:
            supply = venue.setdefault("supply", {})
            demand = venue.setdefault("demand", {})
            for commodity in ("monster_hide", "monster_meat", "bone_shard"):
                supply_level = float(supply.get(commodity, 4.0) or 4.0)
                demand_level = float(demand.get(commodity, 8.0) or 8.0)
                drift = random.uniform(-0.2, 0.6)
                demand_level = max(0.0, demand_level * 0.94 + 0.45 + drift)
                supply_level = max(0.0, supply_level * 0.985)
                supply[commodity] = round(supply_level, 3)
                demand[commodity] = round(demand_level, 3)

            total_supply = sum(float(value) for value in supply.values())
            total_demand = sum(float(value) for value in demand.values())
            ratio = total_demand / max(1.0, total_supply)
            venue["price_index"] = max(0.65, min(1.8, 0.82 + ratio * 0.28))

    async def _run_economy_tick(self) -> None:
        self._update_market_pressure()
        self._simulate_hunting_runs()
        await self._simulate_market_sales()

    def _entity_region_id(self, entity: dict[str, Any]) -> str:
        x = float(entity.get("x", 0.0))
        y = float(entity.get("y", 0.0))
        for region in self._regions:
            vertices = region.get("vertices")
            if isinstance(vertices, list) and vertices and self._point_in_polygon(x, y, vertices):
                return str(region.get("id", ""))
        return ""

    def _simulate_hunting_runs(self) -> None:
        if not self._hunting_areas:
            return
        area = self._hunting_areas[0]
        if not bool(area.get("active", False)):
            return

        center_x = float(area.get("x", self._region_width * 0.8))
        center_y = float(area.get("y", self._region_height * 0.8))
        radius = float(area.get("radius", 110.0))
        success_bonus = float(area.get("success_bonus", 0.0))
        candidates = [
            entity
            for entity in self._world_entities
            if str(entity.get("kind")) == "npc"
            and math.hypot(float(entity.get("x", 0.0)) - center_x, float(entity.get("y", 0.0)) - center_y)
            <= radius * 1.8
        ]
        if not candidates:
            candidates = [entity for entity in self._world_entities if str(entity.get("kind")) == "npc"]
        if not candidates:
            return

        sample_count = min(4, len(candidates))
        for hunter in random.sample(candidates, sample_count):
            stamina = float(hunter.get("stamina", 100.0) or 100.0)
            success_chance = max(0.1, min(0.82, 0.23 + (stamina / 420.0) + success_bonus + random.uniform(-0.08, 0.12)))
            if random.random() > success_chance:
                continue

            commodity = random.choice(["monster_hide", "monster_meat", "bone_shard"])
            quantity = max(1, int(random.uniform(1, 4)))
            goods = hunter.setdefault("gathered_goods", {})
            goods[commodity] = int(goods.get(commodity, 0) or 0) + quantity
            inventory = hunter.setdefault("inventory", [])
            for _ in range(quantity):
                inventory.append(commodity)
            hunter["stamina"] = max(0.0, stamina - random.uniform(2.5, 7.5))

            self._log_event(
                summary="Hunting success",
                details=[
                    f"Hunter: {hunter.get('label', hunter.get('id', 'npc'))}",
                    f"Area: {area.get('name', 'Hunting Grounds')}",
                    f"Yield: {quantity}x {commodity}",
                    f"Success chance: {success_chance:.2f}",
                ],
                category="economy",
            )

    def _choose_sale_venue(self, seller: dict[str, Any]) -> dict[str, Any] | None:
        region_id = self._entity_region_id(seller)
        active_black = [item for item in self._black_markets if bool(item.get("active", False))]
        use_black = bool(active_black) and random.random() < 0.38
        venue_pool = active_black if use_black else self._markets
        if not venue_pool:
            venue_pool = self._markets or active_black
        if not venue_pool:
            return None

        same_region = [item for item in venue_pool if str(item.get("region_id", "")) == region_id]
        if same_region:
            return random.choice(same_region)
        return random.choice(venue_pool)

    async def _simulate_market_sales(self) -> None:
        base_prices = {"monster_hide": 12.0, "monster_meat": 8.0, "bone_shard": 7.0}
        sellers = [
            entity
            for entity in self._world_entities
            if isinstance(entity.get("gathered_goods"), dict) and any(int(v or 0) > 0 for v in entity["gathered_goods"].values())
        ]
        random.shuffle(sellers)

        for seller in sellers[:6]:
            goods = seller.setdefault("gathered_goods", {})
            sale_options = [item for item, qty in goods.items() if int(qty or 0) > 0]
            if not sale_options:
                continue
            commodity = random.choice(sale_options)
            quantity_available = int(goods.get(commodity, 0) or 0)
            if quantity_available <= 0:
                continue

            venue = self._choose_sale_venue(seller)
            if venue is None:
                continue

            quantity = min(quantity_available, max(1, int(random.uniform(1, 4))))
            supply = venue.setdefault("supply", {})
            demand = venue.setdefault("demand", {})
            supply_level = float(supply.get(commodity, 4.0) or 4.0)
            demand_level = float(demand.get(commodity, 9.0) or 9.0)
            price_index = float(venue.get("price_index", 1.0) or 1.0)

            scarcity = max(-0.45, min(1.15, (demand_level - supply_level) / max(4.0, demand_level + supply_level)))
            unit_price = base_prices.get(commodity, 6.0) * (1.0 + scarcity) * price_index
            unit_price = max(2.0, min(45.0, unit_price))

            gross = quantity * unit_price
            tax_rate = float(venue.get("tax_rate", 0.0) or 0.0)
            tax = gross * tax_rate
            net = gross - tax

            seller["coins"] = int(seller.get("coins", 0) or 0) + max(1, int(round(net)))
            goods[commodity] = max(0, quantity_available - quantity)
            supply[commodity] = round(supply_level + quantity, 3)
            demand[commodity] = round(max(0.0, demand_level - (quantity * 0.4)), 3)
            venue["tax_collected"] = round(float(venue.get("tax_collected", 0.0) or 0.0) + tax, 3)
            venue["gross_volume"] = round(float(venue.get("gross_volume", 0.0) or 0.0) + gross, 3)

            inventory = seller.setdefault("inventory", [])
            removed = 0
            while removed < quantity:
                try:
                    inventory.remove(commodity)
                    removed += 1
                except ValueError:
                    break

            venue_kind = "black market" if str(venue.get("id", "")).startswith("black-market-") else "market"
            self._log_event(
                summary="Goods sold",
                details=[
                    f"Seller: {seller.get('label', seller.get('id', 'npc'))}",
                    f"Venue: {venue.get('name', venue.get('id', 'market'))} ({venue_kind})",
                    f"Commodity: {commodity}",
                    f"Quantity: {quantity}",
                    f"Gross: {gross:.1f}",
                    f"Tax: {tax:.1f}",
                    f"Net: {net:.1f}",
                ],
                category="economy",
            )
            await self._record_market_transaction_to_graph(
                seller_id=str(seller.get("id", "")),
                venue_id=str(venue.get("id", "")),
                commodity=commodity,
                quantity=quantity,
                gross=gross,
                tax=tax,
                net=net,
            )

    async def _record_market_transaction_to_graph(
        self,
        seller_id: str,
        venue_id: str,
        commodity: str,
        quantity: int,
        gross: float,
        tax: float,
        net: float,
    ) -> None:
        if not seller_id or not venue_id:
            return
        transaction_id = f"txn-{uuid4().hex[:12]}"
        await self.graph.query(
            """
            MATCH (seller:NPC {id: $seller_id})
            OPTIONAL MATCH (m:Market {id: $venue_id})
            OPTIONAL MATCH (bm:BlackMarket {id: $venue_id})
            WITH seller, coalesce(m, bm) AS venue
            WHERE venue IS NOT NULL
            CREATE (t:Transaction {
                id: $transaction_id,
                commodity: $commodity,
                quantity: $quantity,
                gross_value: $gross,
                tax_value: $tax,
                net_value: $net,
                game_time_seconds: $game_time_seconds
            })
            MERGE (seller)-[:SOLD_GOODS]->(t)
            MERGE (venue)-[:RECORDED_TRANSACTION]->(t)
            """,
            {
                "seller_id": seller_id,
                "venue_id": venue_id,
                "transaction_id": transaction_id,
                "commodity": commodity,
                "quantity": int(quantity),
                "gross": float(gross),
                "tax": float(tax),
                "net": float(net),
                "game_time_seconds": self._world_time_seconds,
            },
        )

    async def _sync_economy_graph(self) -> None:
        market_rows = [self._serialize_market_for_graph(item) for item in self._markets]
        black_market_rows = [self._serialize_market_for_graph(item) for item in self._black_markets]

        await self.graph.query(
            """
            UNWIND $markets AS market
            MERGE (m:Market {id: market.id})
            SET m += market, m.sim_managed = true
            """,
            {"markets": market_rows},
        )
        await self.graph.query(
            """
            MATCH (m:Market {sim_managed: true})
            WHERE NOT m.id IN $market_ids
            DETACH DELETE m
            """,
            {"market_ids": [str(item["id"]) for item in self._markets]},
        )

        await self.graph.query(
            """
            UNWIND $markets AS market
            MERGE (m:BlackMarket {id: market.id})
            SET m += market, m.sim_managed = true
            """,
            {"markets": black_market_rows},
        )
        await self.graph.query(
            """
            MATCH (m:BlackMarket {sim_managed: true})
            WHERE NOT m.id IN $market_ids
            DETACH DELETE m
            """,
            {"market_ids": [str(item["id"]) for item in self._black_markets]},
        )

        await self.graph.query(
            """
            UNWIND $markets AS market
            MATCH (b:Building {id: market.building_id}), (m:Market {id: market.id})
            MERGE (b)-[:HOSTS_MARKET]->(m)
            """,
            {"markets": self._markets},
        )
        await self.graph.query(
            """
            UNWIND $markets AS market
            MATCH (b:Building {id: market.building_id}), (m:BlackMarket {id: market.id})
            MERGE (b)-[:HOSTS_BLACK_MARKET]->(m)
            """,
            {"markets": self._black_markets},
        )

        await self.graph.query(
            """
            UNWIND $areas AS area
            MERGE (h:HuntingArea {id: area.id})
            SET h += area, h.sim_managed = true
            """,
            {"areas": self._hunting_areas},
        )
        await self.graph.query(
            """
            MATCH (h:HuntingArea {sim_managed: true})
            WHERE NOT h.id IN $area_ids
            DETACH DELETE h
            """,
            {"area_ids": [str(area["id"]) for area in self._hunting_areas]},
        )
        await self.graph.query(
            """
            UNWIND $areas AS area
            MATCH (h:HuntingArea {id: area.id}), (r:Region {id: area.region_id})
            MERGE (r)-[:HAS_HUNTING_AREA]->(h)
            """,
            {"areas": self._hunting_areas},
        )

    def _serialize_market_for_graph(self, market: dict[str, Any]) -> dict[str, Any]:
        row = dict(market)
        row["supply_json"] = json.dumps(market.get("supply", {}))
        row["demand_json"] = json.dumps(market.get("demand", {}))
        row.pop("supply", None)
        row.pop("demand", None)
        return row

    def _safe_inventory_token(self, label: str) -> str:
        token = re.sub(r"[^a-zA-Z0-9]+", "-", label.strip().lower())
        return token.strip("-")[:20] or "item"

    async def _infer_player_intent_from_graph(self, host_id: str) -> dict[str, Any]:
        rows = await self.graph.query(
            """
            MATCH (p:PossessionState {id: $possession_id})-[:RECEIVED_COMMAND]->(c:PlayerCommand)
            RETURN c.raw_text AS raw_text, c.host_response_type AS response_type, c.tick AS tick
            ORDER BY c.tick DESC
            LIMIT 20
            """,
            {"possession_id": POSSESSION_NODE_ID},
        )

        helpful_terms = ("help", "heal", "protect", "save", "rescue", "shelter", "feed")
        harmful_terms = ("kill", "hurt", "assault", "burn", "torture", "poison")
        helpful_count = 0
        harmful_count = 0
        coercive_count = 0

        for row in rows:
            text = str(row.get("raw_text", "")).lower()
            response_type = str(row.get("response_type", "")).lower()
            if any(term in text for term in helpful_terms):
                helpful_count += 1
            if any(term in text for term in harmful_terms):
                harmful_count += 1
            if response_type in {"minimal", "malicious"}:
                coercive_count += 1

        sample_size = max(1, len(rows))
        polarity = (helpful_count - harmful_count) / sample_size
        pressure = coercive_count / sample_size

        if harmful_count > helpful_count and pressure > 0.35:
            inferred_goal = "Exploit host for violent leverage"
        elif helpful_count > harmful_count:
            inferred_goal = "Stabilize allies and gather cooperative influence"
        else:
            inferred_goal = "Pursue unclear agenda through opportunistic commands"

        confidence = min(0.95, 0.35 + (abs(polarity) * 0.45) + (pressure * 0.2))
        intent = {
            "host_id": host_id,
            "inferred_goal": inferred_goal,
            "confidence": round(confidence, 3),
            "helpful_command_ratio": round(helpful_count / sample_size, 3),
            "harmful_command_ratio": round(harmful_count / sample_size, 3),
            "coercive_response_ratio": round(pressure, 3),
            "sample_size": len(rows),
        }

        await self._record_intent_inference_to_graph(intent)
        return intent

    async def _record_intent_inference_to_graph(self, intent: dict[str, Any]) -> None:
        inference_id = f"intent-{uuid4().hex[:12]}"
        await self.graph.query(
            """
            MATCH (host:NPC {id: $host_id}), (p:PossessionState {id: $possession_id})
            CREATE (i:IntentInference {
                id: $inference_id,
                inferred_goal: $inferred_goal,
                confidence: $confidence,
                helpful_command_ratio: $helpful_ratio,
                harmful_command_ratio: $harmful_ratio,
                coercive_response_ratio: $coercive_ratio,
                sample_size: $sample_size,
                game_time_seconds: $game_time_seconds
            })
            MERGE (host)-[:INFERRED_PLAYER_INTENT]->(i)
            MERGE (p)-[:LATEST_INTENT_INFERENCE]->(i)
            """,
            {
                "host_id": str(intent.get("host_id", self._host_id)),
                "possession_id": POSSESSION_NODE_ID,
                "inference_id": inference_id,
                "inferred_goal": str(intent.get("inferred_goal", "unknown")),
                "confidence": float(intent.get("confidence", 0.0)),
                "helpful_ratio": float(intent.get("helpful_command_ratio", 0.0)),
                "harmful_ratio": float(intent.get("harmful_command_ratio", 0.0)),
                "coercive_ratio": float(intent.get("coercive_response_ratio", 0.0)),
                "sample_size": int(intent.get("sample_size", 0)),
                "game_time_seconds": self._world_time_seconds,
            },
        )

    async def _attempt_robbery(self, left_id: str, right_id: str) -> None:
        left = next((entity for entity in self._world_entities if str(entity.get("id")) == left_id), None)
        right = next((entity for entity in self._world_entities if str(entity.get("id")) == right_id), None)
        if left is None or right is None:
            return

        thief, victim = (left, right) if random.random() < 0.5 else (right, left)

        victim_inventory = victim.get("inventory")
        if not isinstance(victim_inventory, list):
            victim_inventory = []
            victim["inventory"] = victim_inventory

        thief_inventory = thief.get("inventory")
        if not isinstance(thief_inventory, list):
            thief_inventory = []
            thief["inventory"] = thief_inventory

        stolen_item: str | None = None
        if victim_inventory:
            stolen_item = str(victim_inventory.pop(0))
            thief_inventory.append(stolen_item)

        stolen_coins = 0
        victim_coins = int(victim.get("coins", 0) or 0)
        if victim_coins > 0:
            stolen_coins = max(1, int(victim_coins * 0.25))
            victim["coins"] = max(0, victim_coins - stolen_coins)
            thief["coins"] = int(thief.get("coins", 0) or 0) + stolen_coins

        if not stolen_item and stolen_coins <= 0:
            return

        details = [
            f"Thief: {thief.get('label', thief.get('id'))}",
            f"Victim: {victim.get('label', victim.get('id'))}",
        ]
        if stolen_item:
            details.append(f"Item stolen: {stolen_item}")
        if stolen_coins > 0:
            details.append(f"Coins stolen: {stolen_coins}")

        self._log_event(summary="Robbery incident", details=details, category="interaction")

        await self.graph.query(
            """
            MATCH (thief:NPC {id: $thief_id}), (victim:NPC {id: $victim_id})
            CREATE (m:Memory {
                id: $memory_id,
                kind: 'robbery',
                summary: $summary,
                details: $details,
                game_time_seconds: $game_time_seconds
            })
            MERGE (thief)-[:GENERATED_MEMORY]->(m)
            MERGE (victim)-[:GENERATED_MEMORY]->(m)
            MERGE (rel:Relationship {id: $relationship_id})
            ON CREATE SET
                rel.agent_a_id = $thief_id,
                rel.agent_b_id = $victim_id,
                rel.friendship = 0.2,
                rel.trust = 0.2,
                rel.respect = 0.2,
                rel.interaction_count = 0
            SET
                rel.interaction_count = coalesce(rel.interaction_count, 0) + 1,
                rel.trust = CASE
                    WHEN coalesce(rel.trust, 0.2) - 0.07 < 0.0 THEN 0.0
                    ELSE coalesce(rel.trust, 0.2) - 0.07
                END,
                rel.respect = CASE
                    WHEN coalesce(rel.respect, 0.2) - 0.05 < 0.0 THEN 0.0
                    ELSE coalesce(rel.respect, 0.2) - 0.05
                END,
                rel.last_interaction_game_seconds = $game_time_seconds
            """,
            {
                "thief_id": str(thief.get("id")),
                "victim_id": str(victim.get("id")),
                "memory_id": f"memory-rob-{uuid4().hex[:12]}",
                "relationship_id": f"rel-{'-'.join(sorted([str(thief.get('id')), str(victim.get('id'))]))}",
                "summary": "Robbery between nearby NPCs",
                "details": details,
                "game_time_seconds": self._world_time_seconds,
            },
        )

    async def run(self) -> None:
        logger.info("Simulation loop entering run state")
        await self._ensure_graph_backed_state()
        self._announce_host_context()

        while True:
            self._tick_index += 1
            self._poll_command_task()
            self._poll_social_emit_task()
            await self._collect_ready_social_interactions()
            await self._collect_ready_combat_narrations()
            await self._tick_world(self._tick_seconds)
            self._resolve_collisions()
            await self._advance_action_sequences()

            self._combat_tick_accumulator += self._tick_seconds
            while self._combat_tick_accumulator >= self._combat_interval_seconds:
                await self._run_combat_tick()
                self._combat_tick_accumulator -= self._combat_interval_seconds

            self._economy_tick_accumulator += self._tick_seconds
            if self._economy_tick_accumulator >= self._economy_interval_seconds:
                await self._run_economy_tick()
                self._economy_tick_accumulator = 0.0

            if self._tick_index % 5 == 0:
                await self._run_social_fabric_tick()
            if self._tick_index % 10 == 0:
                await self._run_information_diffusion_tick()
            if self._tick_index % 500 == 0:
                await self._run_trait_evolution_tick()

            self._render_frame()
            self._publish_world_state()

            if self.possession.can_command and not self._was_command_ready:
                logger.info("Cooldown completed; command entry now available")
                self.ui.stream_narrative("Cooldown complete. Hold # to push-to-talk and issue a command.")
                self._log_event(
                    summary="Compulsion ready",
                    details=["Cooldown finished", "Waiting for # command input"],
                    category="system",
                )

            command: str | None = None
            if self.possession.can_command:
                if self._command_task is None:
                    if hasattr(self.ui, "pop_command"):
                        command = self.ui.pop_command()
                    elif hasattr(self.ui, "wait_for_command"):
                        command = await self.ui.wait_for_command()
                    else:
                        command = self.voice.record_and_transcribe(duration=5.0)
                        if not command:
                            command = self.ui.get_player_command()
            else:
                self.possession.tick(self._tick_seconds)

            await self._maybe_run_autonomous_behavior(
                self._tick_seconds,
                command_issued=bool(command) or self._command_task is not None,
            )

            self._graph_sync_timer += self._tick_seconds
            if self._graph_sync_timer >= 1.0:
                await self._sync_graph_state()
                self._graph_sync_timer = 0.0

            self._was_command_ready = self.possession.can_command

            if command:
                if command.lower() in {"quit", "exit"}:
                    logger.info("Received termination command: %s", command)
                    self.ui.stream_narrative("The presence fades. Session ended.")
                    await self._cancel_background_tasks()
                    return
                logger.info("Dispatching command asynchronously: %s", command)
                self._active_command_text = command
                self._command_task = asyncio.create_task(self._process_command_async(command))

            await asyncio.sleep(self._tick_seconds)

    async def _process_command_async(self, command: str) -> None:
        try:
            await self._handle_command(command)
        except Exception as exc:
            logger.exception("Command handling failed for input: %s", command)
            self.ui.stream_narrative(
                "Command processing failed due to model output formatting. "
                "The simulation is still running; try rephrasing or retrying."
            )
            self._log_event(
                summary="Command processing failed",
                details=[f"Command: {command}", f"Error: {type(exc).__name__}: {exc}"],
                category="system",
            )
        finally:
            if self._active_command_text == command:
                self._active_command_text = ""

    def _poll_command_task(self) -> None:
        if self._command_task is None:
            return
        if not self._command_task.done():
            return

        try:
            self._command_task.result()
        except asyncio.CancelledError:
            logger.info("Background command task cancelled")
        except Exception:
            logger.exception("Background command task crashed")
        finally:
            self._command_task = None

    def _poll_social_emit_task(self) -> None:
        if self._social_emit_task is None:
            return
        if not self._social_emit_task.done():
            return

        try:
            self._social_emit_task.result()
        except asyncio.CancelledError:
            logger.info("Background social emission task cancelled")
        except Exception:
            logger.exception("Background social emission task crashed")
        finally:
            self._social_emit_task = None

    async def _cancel_background_tasks(self) -> None:
        if self._command_task is not None:
            self._command_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._command_task
            self._command_task = None

        if self._social_emit_task is not None:
            self._social_emit_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._social_emit_task
            self._social_emit_task = None

        for task in self._pending_combat_narration_tasks.values():
            task.cancel()
        for task in self._pending_combat_narration_tasks.values():
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._pending_combat_narration_tasks.clear()

        for task in self._pending_social_interaction_tasks.values():
            task.cancel()
        for task in self._pending_social_interaction_tasks.values():
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._pending_social_interaction_tasks.clear()
        self._pending_social_interaction_meta.clear()
        self._motion_locked_entity_ids.clear()

    async def _handle_command(self, command: str) -> None:
        if _is_mental_state_command(command):
            logger.warning("Rejected mental-state command: %s", command)
            self.possession.commands_resisted += 1
            self.possession.host_resistance = min(1.0, self.possession.host_resistance + 0.01)
            self.ui.stream_narrative(
                "That command targets internal mental state and is rejected by the compulsion."
            )
            self._log_event(
                summary="Command rejected",
                details=[f"Input: {command}", "Reason: Mental-state command blocked"],
                category="system",
            )
            return

        self.ui.stream_narrative(f"> You compel: {command}")
        self._log_event(
            summary="Command received",
            details=[f"Command: {command}", "Status: reasoning in progress"],
            category="system",
        )
        logger.info("Running reasoning pipeline for command: %s", command)
        event_details: list[str] = [f"Command: {command}"]

        command_duration_limit_game_minutes = compute_command_duration_limit_game_minutes(self.possession)
        seed_context = {
            "possession": {
                "host_resistance": self.possession.host_resistance,
                "host_willpower": self.possession.host_willpower,
                "trust_in_player": self.possession.trust_in_player,
                "understanding_of_player": self.possession.understanding_of_player,
                "approval_of_player": self.possession.approval_of_player,
                "harm_from_player_commands": self.possession.harm_from_player_commands,
                "benefit_from_player_commands": self.possession.benefit_from_player_commands,
            },
            "command_duration_limit_game_minutes": command_duration_limit_game_minutes,
        }

        reasoning_output = await self.reasoning.run_with_graph_access(
            system_prompt=HOST_RESPONSE_SYSTEM_PROMPT,
            task_prompt=(
                "Determine how the host complies with this command: "
                f"{command}. Return valid JSON only."
            ),
            seed_context=seed_context,
            output_schema=REASONING_OUTPUT_SCHEMA,
            graph_tools=self.graph_tools,
        )
        logger.debug("Reasoning output keys: %s", sorted(reasoning_output.keys()))

        likely_success_raw = reasoning_output.get("likely_success", reasoning_output.get("effort_level", 0.5))
        try:
            likely_success = float(likely_success_raw)
        except (TypeError, ValueError):
            likely_success = 0.5
        reasoning_output["likely_success"] = max(0.0, min(1.0, likely_success))

        spoken_words = _validate_speech(str(reasoning_output.get("spoken_words", "")))
        reasoning_output["spoken_words"] = spoken_words

        speech_output = self.speech.generate_speech(reasoning_output)
        logger.debug("Speech output keys: %s", sorted(speech_output.keys()))
        await self._stream_speech(speech_output)

        self.ui.stream_narrative(
            f"[Internal] {reasoning_output.get('internal_reaction', '...')}"
        )
        event_details.append(f"Internal reaction: {reasoning_output.get('internal_reaction', '...')}")
        event_details.append(f"Estimated success: {float(reasoning_output.get('likely_success', 0.5)):.2f}")

        for block_name in ("pre_actions", "during_actions", "post_actions"):
            for action in reasoning_output.get(block_name, []):
                event_details.append(f"Queued {block_name}: {action}")

        self._apply_post_command_drift(reasoning_output)
        self._update_command_outcome_metrics(command, reasoning_output)
        cooldown_hours = compute_next_cooldown_game_hours(self.possession)
        self.possession.start_cooldown(cooldown_hours)
        self.possession.commands_obeyed += 1
        logger.info("Cooldown started for %.2f game hours", cooldown_hours)
        self.ui.stream_narrative(
            f"Cooldown started: {cooldown_hours:.2f} game hours ({cooldown_hours * 7.5:.1f} real minutes)."
        )
        event_details.append(f"Cooldown started: {cooldown_hours:.2f} game hours")
        command_id = await self._record_command_to_graph(command=command, reasoning_output=reasoning_output)
        resolved_outcome = await self._resolve_command_outcome_from_capabilities(command, reasoning_output)
        await self._record_command_execution_outcome_to_graph(command_id=command_id, outcome=resolved_outcome)
        event_details.append(
            "Resolved outcome: "
            f"{'success' if bool(resolved_outcome.get('success', False)) else 'failure'} "
            f"(domain={resolved_outcome.get('required_domain', 'general')}, "
            f"capability={float(resolved_outcome.get('capability_level', 0.5)):.2f})"
        )
        await self._create_action_sequence_for_command(
            command_id=command_id,
            reasoning_output=reasoning_output,
            command_duration_limit_game_minutes=command_duration_limit_game_minutes,
        )

        grounded_destination = await self._plan_command_destination_from_graph(command, reasoning_output)
        if grounded_destination is not None:
            host = next((entity for entity in self._world_entities if entity.get("kind") == "host"), None)
            if host is not None:
                host["patrol_target_x"] = float(grounded_destination["x"])
                host["patrol_target_y"] = float(grounded_destination["y"])
                event_details.append(f"Grounded destination: {grounded_destination['name']}")

        self._log_event(
            summary=f"Compulsion executed: {command}",
            details=event_details,
            category="interaction",
        )

    def _render_frame(self) -> None:
        self.ui.render_frame(
            FrameState(
                cooldown_remaining_game_hours=self.possession.cooldown_remaining_game_hours,
                cooldown_total_game_hours=self.possession.cooldown_total_game_hours,
                host_resistance=self.possession.host_resistance,
                host_willpower=self.possession.host_willpower,
            )
        )

    def _announce_host_context(self) -> None:
        host = next((entity for entity in self._world_entities if str(entity.get("id", "")) == self._host_id), None)
        if host is None:
            return

        host_x = float(host.get("x", 0.0))
        host_y = float(host.get("y", 0.0))
        region = self._region_for_point(host_x, host_y)
        self.ui.stream_narrative(
            "Possession anchor established: "
            f"{host.get('label', self._host_id)} "
            f"at ({host_x:.1f}, {host_y:.1f}) in {region}."
        )
        self.ui.stream_narrative(
            "Host baseline | "
            f"Resistance {self.possession.host_resistance * 100:.0f}% | "
            f"Willpower {self.possession.host_willpower * 100:.0f}% | "
            f"Health {float(host.get('health', 0.0)):.0f} | "
            f"Stamina {float(host.get('stamina', 0.0)):.0f}"
        )

    async def _stream_speech(self, speech_output: dict[str, Any]) -> None:
        audio_file = speech_output.get("audio_file")
        if audio_file:
            logger.info("Generated ElevenLabs audio clip: %s", audio_file)
            self.ui.stream_narrative(f"[Audio] Generated ElevenLabs clip: {audio_file}")

        spoken_fragments: list[str] = []

        segments = speech_output.get("segments", [])
        if not segments:
            logger.debug("Speech output contained no segments")
            self.ui.stream_narrative('> "..."')
            return

        logger.info("Streaming %s speech segment(s)", len(segments))
        for segment in segments:
            text = segment.get("text", "...")
            tone = segment.get("tone", "neutral")
            pacing = segment.get("pacing", "medium")
            pause_after_ms = int(segment.get("pause_after_ms", 0))
            self.ui.stream_narrative(f'> [{tone}/{pacing}] "{text}"')
            spoken_fragments.append(f"[{tone}/{pacing}] {text}")
            if pause_after_ms > 0:
                await asyncio.sleep(pause_after_ms / 1000.0)

        if spoken_fragments:
            self._log_event(
                summary="Host speech emitted",
                details=spoken_fragments,
                category="speech",
            )

    def _apply_post_command_drift(self, reasoning_output: dict[str, Any]) -> None:
        relationship_changes = reasoning_output.get("relationship_changes")
        if isinstance(relationship_changes, dict):
            self.possession.trust_in_player = max(
                0.0,
                min(1.0, self.possession.trust_in_player + float(relationship_changes.get("trust_delta", 0.0))),
            )
            self.possession.host_resistance = max(
                0.0,
                min(1.0, self.possession.host_resistance + float(relationship_changes.get("resistance_delta", 0.0))),
            )
            self.possession.understanding_of_player = max(
                0.0,
                min(
                    1.0,
                    self.possession.understanding_of_player
                    + float(relationship_changes.get("understanding_delta", 0.0)),
                ),
            )
            return

        style = str(reasoning_output.get("compliance_style", "neutral")).lower()
        effort = float(reasoning_output.get("effort_level", 0.5) or 0.5)

        if style in {"malicious", "minimal"}:
            self.possession.host_resistance = min(1.0, self.possession.host_resistance + 0.03)
            self.possession.trust_in_player = max(0.0, self.possession.trust_in_player - 0.02)
        elif style in {"cooperative", "enthusiastic"}:
            self.possession.host_resistance = max(0.0, self.possession.host_resistance - 0.02)
            self.possession.trust_in_player = min(1.0, self.possession.trust_in_player + 0.03)

        if effort < 0.3:
            self.possession.approval_of_player = max(0.0, self.possession.approval_of_player - 0.01)
        elif effort > 0.7:
            self.possession.approval_of_player = min(1.0, self.possession.approval_of_player + 0.01)

    def _update_command_outcome_metrics(self, command: str, reasoning_output: dict[str, Any]) -> None:
        lowered = command.lower()
        harmful_terms = ("kill", "hurt", "torture", "maim", "assault", "burn", "poison")
        helpful_terms = ("help", "heal", "save", "protect", "rescue", "feed", "shelter")

        if any(term in lowered for term in harmful_terms):
            self.possession.harm_from_player_commands = min(
                1.0, self.possession.harm_from_player_commands + 0.06
            )
            self.possession.benefit_from_player_commands = max(
                0.0, self.possession.benefit_from_player_commands - 0.02
            )
        elif any(term in lowered for term in helpful_terms):
            self.possession.benefit_from_player_commands = min(
                1.0, self.possession.benefit_from_player_commands + 0.05
            )
            self.possession.harm_from_player_commands = max(
                0.0, self.possession.harm_from_player_commands - 0.02
            )

        if str(reasoning_output.get("compliance_style", "")).lower() == "malicious":
            self.possession.commands_subverted += 1

    async def _resolve_command_outcome_from_capabilities(
        self, command: str, reasoning_output: dict[str, Any]
    ) -> dict[str, Any]:
        command_lower = command.lower()
        interpretation = str(reasoning_output.get("interpretation", "")).lower()
        text = f"{command_lower} {interpretation}"

        domain_keywords: dict[str, tuple[str, ...]] = {
            "combat": ("fight", "attack", "duel", "kill", "guard", "defend"),
            "commerce": ("buy", "sell", "trade", "bargain", "market"),
            "security": ("patrol", "arrest", "watch", "secure", "investigate"),
            "social": ("talk", "ask", "convince", "charm", "negotiate", "persuade"),
            "stealth": ("sneak", "steal", "pickpocket", "lockpick", "infiltrate"),
        }

        required_domain = "general"
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                required_domain = domain
                break

        rows = await self.graph.query(
            """
            MATCH (h:NPC {id: $host_id})-[:HAS_CAPABILITY]->(c:Capability)
            RETURN coalesce(c.domain, 'general') AS domain, coalesce(c.current_level, 0.5) AS level
            """,
            {"host_id": self._host_id},
        )

        levels: list[float] = []
        domain_level = 0.5
        for row in rows:
            level = float(row.get("level", 0.5) or 0.5)
            levels.append(level)
            if str(row.get("domain", "general")).lower() == required_domain:
                domain_level = max(domain_level, level)

        if required_domain == "general" and levels:
            domain_level = sum(levels) / len(levels)

        likely_success = float(reasoning_output.get("likely_success", 0.5) or 0.5)
        effort_level = float(reasoning_output.get("effort_level", 0.5) or 0.5)

        challenge = 0.0
        if any(token in text for token in ("hard", "dangerous", "elite", "heavily", "fortified")):
            challenge += 0.12
        if any(token in text for token in ("quickly", "urgent", "immediately")):
            challenge += 0.06

        resolved_score = (likely_success * 0.5) + (effort_level * 0.15) + (domain_level * 0.35) - challenge
        resolved_score = max(0.03, min(0.97, resolved_score))
        success = random.random() <= resolved_score

        return {
            "required_domain": required_domain,
            "capability_level": round(domain_level, 3),
            "resolved_score": round(resolved_score, 3),
            "success": success,
            "summary": (
                f"Host {'succeeds' if success else 'struggles'} while executing command "
                f"(domain={required_domain}, score={resolved_score:.2f})."
            ),
        }

    async def _record_command_execution_outcome_to_graph(self, command_id: str, outcome: dict[str, Any]) -> None:
        if not command_id:
            return
        await self.graph.query(
            """
            MATCH (c:PlayerCommand {id: $command_id})
            SET
                c.required_capability_domain = $required_domain,
                c.capability_level = $capability_level,
                c.resolved_success_score = $resolved_score,
                c.resolved_success = $resolved_success,
                c.execution_outcome_summary = $summary
            """,
            {
                "command_id": command_id,
                "required_domain": str(outcome.get("required_domain", "general")),
                "capability_level": float(outcome.get("capability_level", 0.5) or 0.5),
                "resolved_score": float(outcome.get("resolved_score", 0.5) or 0.5),
                "resolved_success": bool(outcome.get("success", False)),
                "summary": str(outcome.get("summary", "")),
            },
        )

    async def _maybe_run_autonomous_behavior(self, dt: float, command_issued: bool = False) -> None:
        if command_issued:
            return

        self._autonomy_tick_accumulator += dt
        if self._autonomy_tick_accumulator < self._autonomy_interval_seconds:
            return

        self._autonomy_tick_accumulator = 0.0
        host = next((entity for entity in self._world_entities if entity.get("kind") == "host"), None)
        if host is None:
            return

        active_sequence_rows = await self.graph.query(
            """
            MATCH (:NPC {id: $host_id})-[:HAS_ACTIVE_SEQUENCE]->(seq:ActionSequence {status: 'active'})
            RETURN count(seq) AS active_count
            """,
            {"host_id": self._host_id},
        )
        if int((active_sequence_rows[0] if active_sequence_rows else {}).get("active_count", 0) or 0) > 0:
            return

        if await self._maybe_trigger_extreme_resistance_event(host):
            return

        try:
            intent_inference = await self._infer_player_intent_from_graph(str(host.get("id", self._host_id)))
            action = await self.reasoning.run_with_graph_access(
                system_prompt=AUTONOMOUS_BEHAVIOR_SYSTEM_PROMPT,
                task_prompt="Determine the host's autonomous action during cooldown. Return valid JSON only.",
                seed_context={
                    "host_id": str(host.get("id", self._host_id)),
                    "possession_state_id": POSSESSION_NODE_ID,
                    "world_time_seconds": self._world_time_seconds,
                    "intent_inference": intent_inference,
                    "possession": {
                        "trust_in_player": self.possession.trust_in_player,
                        "approval_of_player": self.possession.approval_of_player,
                        "host_resistance": self.possession.host_resistance,
                        "understanding_of_player": self.possession.understanding_of_player,
                    },
                },
                output_schema=AUTONOMOUS_ACTION_SCHEMA,
                graph_tools=self.graph_tools,
            )
        except Exception:
            logger.exception("Autonomous behavior reasoning failed")
            return

        next_goal = str(action.get("next_goal", "")).strip()
        intent_alignment = str(action.get("intent_alignment", "neutral")).strip().lower()
        target_entity_id = str(action.get("target_entity_id", "")).strip()
        target_region = str(action.get("target_region", "")).strip()

        action_summary = str(action.get("action_summary", "")).strip()
        lowered_summary = action_summary.lower()
        if (not action_summary) or ("takes stock" in lowered_summary and "surroundings" in lowered_summary):
            if next_goal:
                action_summary = f"Host refocuses on: {next_goal}."
            elif target_region:
                action_summary = f"Host starts moving toward {target_region}."
            elif target_entity_id:
                action_summary = f"Host shifts attention toward {target_entity_id}."
            else:
                return

        normalized_summary = re.sub(r"\s+", " ", action_summary).strip().lower()
        generic_autonomy_markers = (
            "takes stock",
            "surroundings",
            "assesses the situation",
            "scans the area",
            "keeps watch",
            "maintains vigilance",
            "regroups",
        )
        is_generic_summary = any(marker in normalized_summary for marker in generic_autonomy_markers)

        if normalized_summary == self._last_autonomy_summary:
            self._autonomy_repeat_count += 1
        else:
            self._autonomy_repeat_count = 0

        if is_generic_summary or self._autonomy_repeat_count >= 1:
            target_label = target_entity_id
            if target_entity_id:
                target = next((entity for entity in self._world_entities if str(entity.get("id", "")) == target_entity_id), None)
                if target is not None:
                    target_label = str(target.get("label", target_entity_id))

            candidates: list[str] = []
            if next_goal:
                candidates.append(f"Host pivots toward this objective: {next_goal}.")
                candidates.append(f"Host reorients around a concrete plan: {next_goal}.")
            if target_region:
                candidates.append(f"Host advances toward {target_region} to act on current priorities.")
            if target_label:
                candidates.append(f"Host closes attention on {target_label} and prepares a response.")

            if candidates:
                action_summary = random.choice(candidates)
                normalized_summary = re.sub(r"\s+", " ", action_summary).strip().lower()
            elif is_generic_summary:
                return

        self._last_autonomy_summary = normalized_summary

        self.ui.stream_narrative(f"[Autonomy] {action_summary}")
        details = [
            f"Action: {action_summary}",
            f"Intent alignment: {intent_alignment or 'neutral'}",
        ]
        if next_goal:
            details.append(f"Next goal: {next_goal}")
        if target_region:
            details.append(f"Target region: {target_region}")
        if target_entity_id:
            details.append(f"Target entity: {target_entity_id}")

        if target_entity_id:
            target = next((entity for entity in self._world_entities if str(entity.get("id")) == target_entity_id), None)
            if target is not None:
                host["patrol_target_x"] = float(target.get("x", host["x"]))
                host["patrol_target_y"] = float(target.get("y", host["y"]))
        elif target_region:
            resolved = await self._resolve_graph_location_hint(target_region)
            if resolved is not None:
                host["patrol_target_x"] = float(resolved["x"])
                host["patrol_target_y"] = float(resolved["y"])
                target_region = str(resolved["name"])
            else:
                details.append(f"Unresolved location hint ignored: {target_region}")
                target_region = ""

        self._log_event(summary="Host autonomous action", details=details, category="autonomy")

        await self._record_autonomous_action_to_graph(
            host_id=str(host.get("id", "host")),
            action_summary=action_summary,
            next_goal=next_goal,
            target_entity_id=target_entity_id,
            target_region=target_region,
            intent_alignment=intent_alignment or "neutral",
        )

    async def _maybe_trigger_extreme_resistance_event(self, host: dict[str, Any]) -> bool:
        resistance = float(self.possession.host_resistance)
        trust = float(self.possession.trust_in_player)
        approval = float(self.possession.approval_of_player)
        willpower = float(self.possession.host_willpower)
        harm_ratio = float(self.possession.harm_from_player_commands)

        is_extreme_resistance = (
            resistance >= 0.94
            and trust <= 0.16
            and approval <= 0.14
            and willpower >= 0.65
            and harm_ratio >= 0.45
        )
        if not is_extreme_resistance:
            return False

        evaluation = await self._evaluate_extreme_resistance_psychology(str(host.get("id", self._host_id)))
        if not evaluation:
            return False

        if not bool(evaluation.get("should_attempt_self_harm", False)):
            return False

        confidence = float(evaluation.get("confidence", 0.0) or 0.0)
        desperation_evidence = [
            str(item) for item in evaluation.get("desperation_evidence", []) if str(item).strip()
        ]
        protective_evidence = [
            str(item) for item in evaluation.get("protective_evidence", []) if str(item).strip()
        ]
        if confidence < 0.55 or not desperation_evidence or len(protective_evidence) > len(desperation_evidence):
            return False

        probability = min(
            0.82,
            0.18
            + ((resistance - 0.94) * 2.2)
            + (harm_ratio * 0.34)
            + (max(0.0, willpower - 0.65) * 0.25),
        )
        probability = max(0.1, min(0.95, probability * (0.7 + confidence * 0.5)))
        if random.random() > probability:
            return False

        health = float(host.get("health", 100.0) or 100.0)
        if health <= 8.0:
            return False

        damage = max(4.0, min(20.0, 5.0 + (harm_ratio * 8.0) + (willpower * 6.0)))
        host["health"] = max(0.0, health - damage)
        host["stamina"] = max(0.0, float(host.get("stamina", 100.0) or 100.0) - damage * 0.7)

        self.possession.commands_resisted += 1
        self.possession.trust_in_player = max(0.0, self.possession.trust_in_player - 0.05)
        self.possession.approval_of_player = max(0.0, self.possession.approval_of_player - 0.04)

        host_name = str(host.get("label", host.get("id", "host")))
        summary = f"{host_name} spirals into self-harm under extreme resistance."
        details = [
            f"Host resistance: {resistance:.2f}",
            f"Trust in player: {trust:.2f}",
            f"Approval of player: {approval:.2f}",
            f"Damage taken: {damage:.1f}",
            f"Psych eval confidence: {confidence:.2f}",
            f"Psych eval rationale: {str(evaluation.get('rationale', 'n/a'))}",
        ]
        details.extend([f"Desperation evidence: {item}" for item in desperation_evidence[:3]])
        details.extend([f"Protective evidence: {item}" for item in protective_evidence[:2]])
        self.ui.stream_narrative(f"[Critical] {summary}")
        self._log_event(summary="Extreme resistance event", details=details, category="autonomy")

        await self.graph.query(
            """
            MATCH (host:NPC {id: $host_id}), (p:PossessionState {id: $possession_id})
            CREATE (m:Memory {
                id: $memory_id,
                kind: 'extreme_resistance',
                summary: $summary,
                details: $details,
                game_time_seconds: $game_time_seconds
            })
            MERGE (host)-[:GENERATED_MEMORY]->(m)
            MERGE (p)-[:GENERATED_MEMORY]->(m)
            """,
            {
                "host_id": str(host.get("id", self._host_id)),
                "possession_id": POSSESSION_NODE_ID,
                "memory_id": f"memory-resistance-{uuid4().hex[:12]}",
                "summary": summary,
                "details": details,
                "game_time_seconds": self._world_time_seconds,
            },
        )
        await self._record_autonomous_action_to_graph(
            host_id=str(host.get("id", self._host_id)),
            action_summary="Extreme resistance self-harm episode",
            next_goal="Regain agency by disrupting compulsion",
            target_entity_id="",
            target_region="",
            intent_alignment="hostile",
        )
        return True

    async def _evaluate_extreme_resistance_psychology(self, host_id: str) -> dict[str, Any] | None:
        try:
            result = await self.reasoning.run_with_graph_access(
                system_prompt=(
                    "Evaluate whether the host is psychologically likely to attempt self-harm as escape from possession. "
                    "Use graph tools to inspect host Trait, Belief, and Memory evidence before deciding. "
                    "Return JSON only."
                ),
                task_prompt=(
                    "Assess extreme resistance risk. Self-harm should only be true if desperation evidence clearly "
                    "outweighs protective attachments and confidence is substantial."
                ),
                seed_context={
                    "host_id": host_id,
                    "possession_state_id": POSSESSION_NODE_ID,
                    "host_resistance": self.possession.host_resistance,
                    "trust_in_player": self.possession.trust_in_player,
                    "approval_of_player": self.possession.approval_of_player,
                    "harm_from_player_commands": self.possession.harm_from_player_commands,
                },
                output_schema=EXTREME_RESISTANCE_SCHEMA,
                graph_tools=self.graph_tools,
                max_rounds=4,
            )
        except Exception:
            logger.debug("Extreme resistance psychology evaluation failed", exc_info=True)
            return None

        if not isinstance(result, dict):
            return None
        return result

    async def _run_combat_tick(self) -> None:
        await self._discover_combat_encounters()
        if not self._active_combat_encounters:
            return

        encounter_ids = list(self._active_combat_encounters.keys())
        all_participants = {
            participant_id
            for encounter in self._active_combat_encounters.values()
            for participant_id in (str(encounter.get("left_id", "")), str(encounter.get("right_id", "")))
            if participant_id
        }
        capability_rows = await self.graph.query(
            """
            UNWIND $ids AS npc_id
            MATCH (n:NPC {id: npc_id})
            OPTIONAL MATCH (n)-[:HAS_CAPABILITY]->(c:Capability)
            RETURN npc_id, collect({domain: coalesce(c.domain, 'general'), level: coalesce(c.current_level, 0.5)}) AS caps
            """,
            {"ids": sorted(all_participants)},
        )
        capability_index: dict[str, float] = {}
        for row in capability_rows:
            npc_id = str(row.get("npc_id", ""))
            caps = row.get("caps", [])
            levels: list[float] = []
            for cap in caps if isinstance(caps, list) else []:
                if not isinstance(cap, dict):
                    continue
                domain = str(cap.get("domain", "general")).lower()
                level = float(cap.get("level", 0.5) or 0.5)
                if domain in {"security", "combat", "martial", "tactics"}:
                    levels.append(level)
            capability_index[npc_id] = max(levels) if levels else 0.5

        for encounter_id in encounter_ids:
            encounter = self._active_combat_encounters.get(encounter_id)
            if encounter is None:
                continue

            left = next(
                (entity for entity in self._world_entities if str(entity.get("id")) == str(encounter.get("left_id", ""))),
                None,
            )
            right = next(
                (entity for entity in self._world_entities if str(entity.get("id")) == str(encounter.get("right_id", ""))),
                None,
            )
            if left is None or right is None:
                self._active_combat_encounters.pop(encounter_id, None)
                continue

            left_health = float(left.get("health", 100.0) or 100.0)
            right_health = float(right.get("health", 100.0) or 100.0)
            if left_health <= 0.0 or right_health <= 0.0:
                await self._end_combat_encounter(encounter_id, left, right, "incapacitation")
                continue

            dist = math.hypot(float(left.get("x", 0.0)) - float(right.get("x", 0.0)), float(left.get("y", 0.0)) - float(right.get("y", 0.0)))
            if dist > 70.0:
                encounter["disengage_ticks"] = int(encounter.get("disengage_ticks", 0) or 0) + 1
                if int(encounter["disengage_ticks"]) >= 4:
                    await self._end_combat_encounter(encounter_id, left, right, "disengaged")
                continue
            encounter["disengage_ticks"] = 0

            left_skill = float(capability_index.get(str(left.get("id", "")), 0.5))
            right_skill = float(capability_index.get(str(right.get("id", "")), 0.5))
            left_stamina = float(left.get("stamina", 100.0) or 100.0)
            right_stamina = float(right.get("stamina", 100.0) or 100.0)

            left_damage = max(0.4, 0.6 + (left_skill * 2.1) + (left_stamina / 180.0) + random.uniform(-0.35, 0.45))
            right_damage = max(0.4, 0.6 + (right_skill * 2.1) + (right_stamina / 180.0) + random.uniform(-0.35, 0.45))

            right["health"] = max(0.0, right_health - left_damage)
            left["health"] = max(0.0, left_health - right_damage)
            left["stamina"] = max(0.0, left_stamina - random.uniform(1.2, 3.4))
            right["stamina"] = max(0.0, right_stamina - random.uniform(1.2, 3.4))
            encounter["rounds"] = int(encounter.get("rounds", 0) or 0) + 1

            if int(encounter["rounds"]) % 6 == 0:
                self._schedule_combat_narration(
                    encounter_id=encounter_id,
                    round_index=int(encounter["rounds"]),
                    left=left,
                    right=right,
                    left_damage=left_damage,
                    right_damage=right_damage,
                )

            if float(left.get("health", 0.0)) <= 0.0 or float(right.get("health", 0.0)) <= 0.0:
                await self._end_combat_encounter(encounter_id, left, right, "incapacitation")

    async def _discover_combat_encounters(self) -> None:
        candidates = [
            entity
            for entity in self._world_entities
            if str(entity.get("kind", "npc")) in {"npc", "host"} and float(entity.get("health", 100.0) or 100.0) > 0.0
        ]
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                left_id = str(left.get("id", ""))
                right_id = str(right.get("id", ""))
                if not left_id or not right_id:
                    continue
                encounter_id = "encounter-" + "-".join(sorted([left_id, right_id]))
                if encounter_id in self._active_combat_encounters:
                    continue

                dist = math.hypot(float(left.get("x", 0.0)) - float(right.get("x", 0.0)), float(left.get("y", 0.0)) - float(right.get("y", 0.0)))
                if dist > 19.0:
                    continue

                hostility = 0.02
                if str(left.get("faction", "")) != str(right.get("faction", "")):
                    hostility += 0.1
                hostility += max(0.0, (0.35 - self.possession.trust_in_player) * 0.02)
                if random.random() > min(0.24, hostility):
                    continue

                self._active_combat_encounters[encounter_id] = {
                    "id": encounter_id,
                    "left_id": left_id,
                    "right_id": right_id,
                    "started_game_time_seconds": self._world_time_seconds,
                    "rounds": 0,
                    "disengage_ticks": 0,
                }
                self._log_event(
                    summary="Combat encounter started",
                    details=[
                        f"Encounter: {encounter_id}",
                        f"Participants: {left.get('label', left_id)} vs {right.get('label', right_id)}",
                    ],
                    category="combat",
                )

    async def _end_combat_encounter(
        self,
        encounter_id: str,
        left: dict[str, Any],
        right: dict[str, Any],
        reason: str,
    ) -> None:
        self._active_combat_encounters.pop(encounter_id, None)
        details = [
            f"Encounter: {encounter_id}",
            f"Reason: {reason}",
            f"{left.get('label', left.get('id', 'left'))} HP {float(left.get('health', 0.0)):.1f}",
            f"{right.get('label', right.get('id', 'right'))} HP {float(right.get('health', 0.0)):.1f}",
        ]
        self._log_event(summary="Combat encounter ended", details=details, category="combat")

        await self._record_interaction_to_graph(
            left_id=str(left.get("id", "")),
            right_id=str(right.get("id", "")),
            summary="Combat encounter concluded",
            details=details,
            kind="combat",
            emotional_tags=["hostility", "fear"],
            intensity=0.72,
            friendship_delta=-0.03,
            trust_delta=-0.04,
            respect_delta=-0.01,
        )

    def _schedule_combat_narration(
        self,
        encounter_id: str,
        round_index: int,
        left: dict[str, Any],
        right: dict[str, Any],
        left_damage: float,
        right_damage: float,
    ) -> None:
        if encounter_id in self._pending_combat_narration_tasks:
            return

        left_snapshot = {
            "id": str(left.get("id", "")),
            "label": str(left.get("label", left.get("id", "left"))),
            "health": float(left.get("health", 0.0)),
            "stamina": float(left.get("stamina", 0.0)),
            "damage_dealt": float(left_damage),
        }
        right_snapshot = {
            "id": str(right.get("id", "")),
            "label": str(right.get("label", right.get("id", "right"))),
            "health": float(right.get("health", 0.0)),
            "stamina": float(right.get("stamina", 0.0)),
            "damage_dealt": float(right_damage),
        }

        self._pending_combat_narration_tasks[encounter_id] = asyncio.create_task(
            self._plan_combat_narration_async(encounter_id, round_index, left_snapshot, right_snapshot)
        )

    async def _plan_combat_narration_async(
        self,
        encounter_id: str,
        round_index: int,
        left_snapshot: dict[str, Any],
        right_snapshot: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            result = await self.reasoning.run_with_graph_access(
                system_prompt=(
                    "Generate one grounded combat beat for a simulation log. "
                    "Use graph tools to pull participant context before writing. "
                    "Return JSON only."
                ),
                task_prompt=(
                    "Produce a concise combat exchange update for the current round. "
                    "No cinematic prose walls; keep it to a few tactical lines."
                ),
                seed_context={
                    "encounter_id": encounter_id,
                    "round": round_index,
                    "left": left_snapshot,
                    "right": right_snapshot,
                },
                output_schema=COMBAT_BEAT_SCHEMA,
                graph_tools=self.graph_tools,
                max_rounds=4,
            )
        except Exception:
            logger.debug("Combat narration generation failed for %s", encounter_id, exc_info=True)
            return None

        if not isinstance(result, dict):
            return None

        summary = str(result.get("summary", "")).strip()
        left_action = str(result.get("left_action", "")).strip()
        right_action = str(result.get("right_action", "")).strip()
        tone = str(result.get("tone", "tense")).strip() or "tense"
        momentum = str(result.get("momentum", "contested")).strip() or "contested"

        if not summary:
            return None

        details = [f"Encounter: {encounter_id}", f"Round: {round_index}", f"Tone: {tone}", f"Momentum: {momentum}"]
        if left_action:
            details.append(f"{left_snapshot['label']}: {left_action}")
        if right_action:
            details.append(f"{right_snapshot['label']}: {right_action}")
        details.append(f"{left_snapshot['label']} HP {float(left_snapshot['health']):.1f}")
        details.append(f"{right_snapshot['label']} HP {float(right_snapshot['health']):.1f}")

        return {"summary": summary, "details": details}

    async def _collect_ready_combat_narrations(self) -> None:
        if not self._pending_combat_narration_tasks:
            return

        for encounter_id, task in list(self._pending_combat_narration_tasks.items()):
            if not task.done():
                continue

            self._pending_combat_narration_tasks.pop(encounter_id, None)
            try:
                payload = task.result()
            except asyncio.CancelledError:
                continue
            except Exception:
                logger.exception("Combat narration task crashed for %s", encounter_id)
                continue

            if not payload:
                continue

            self._log_event(
                summary=str(payload.get("summary", "Combat exchange")),
                details=[str(item) for item in payload.get("details", []) if str(item).strip()],
                category="combat",
            )

    async def _tick_world(self, dt: float) -> None:
        self._world_time_seconds += dt

        for entity in self._world_entities:
            entity_id = str(entity.get("id", ""))
            if entity_id in self._motion_locked_entity_ids:
                entity["vx"] = 0.0
                entity["vy"] = 0.0
                continue

            kind = str(entity.get("kind", "npc"))
            host_autonomous = kind == "host"
            if kind != "npc" and not host_autonomous:
                entity["vx"] = 0.0
                entity["vy"] = 0.0
                continue

            target_x = float(entity.get("patrol_target_x", entity["x"]))
            target_y = float(entity.get("patrol_target_y", entity["y"]))
            dx = target_x - float(entity["x"])
            dy = target_y - float(entity["y"])
            distance = (dx * dx + dy * dy) ** 0.5

            if distance < 16.0:
                entity["patrol_target_x"] = random.uniform(30.0, self._region_width - 30.0)
                entity["patrol_target_y"] = random.uniform(30.0, self._region_height - 30.0)
                continue

            speed = float(entity.get("speed", 42.0))
            step = min(distance, speed * dt)
            nx = dx / max(distance, 0.001)
            ny = dy / max(distance, 0.001)
            entity["vx"] = nx * speed
            entity["vy"] = ny * speed
            entity["x"] = max(0.0, min(self._region_width, float(entity["x"]) + nx * step))
            entity["y"] = max(0.0, min(self._region_height, float(entity["y"]) + ny * step))

        # Interaction events are emitted from Neo4j query results during periodic graph sync.

    def _resolve_collisions(self) -> None:
        entities = self._world_entities
        min_separation = 7.5
        for idx, left in enumerate(entities):
            for right in entities[idx + 1 :]:
                dx = float(left.get("x", 0.0)) - float(right.get("x", 0.0))
                dy = float(left.get("y", 0.0)) - float(right.get("y", 0.0))
                dist = math.hypot(dx, dy)
                if dist <= 0.001 or dist >= min_separation:
                    continue
                overlap = (min_separation - dist) * 0.5
                nx = dx / dist
                ny = dy / dist
                left["x"] = max(0.0, min(self._region_width, float(left.get("x", 0.0)) + nx * overlap))
                left["y"] = max(0.0, min(self._region_height, float(left.get("y", 0.0)) + ny * overlap))
                right["x"] = max(0.0, min(self._region_width, float(right.get("x", 0.0)) - nx * overlap))
                right["y"] = max(0.0, min(self._region_height, float(right.get("y", 0.0)) - ny * overlap))

    async def _run_social_fabric_tick(self) -> None:
        await self._advance_active_social_interactions()
        await self._collect_ready_social_interactions()
        if self._social_emit_task is None:
            self._social_emit_task = asyncio.create_task(self._emit_world_interactions_from_graph_task())

    async def _emit_world_interactions_from_graph_task(self) -> None:
        try:
            await self._emit_world_interactions_from_graph()
        except Exception:
            logger.exception("Background social interaction emission failed")

    async def _run_information_diffusion_tick(self) -> None:
        await self.graph.query(
            """
            MATCH (source:NPC)-[:GENERATED_MEMORY]->(m:Memory)
            WHERE m.kind IN ['interaction', 'combat']
              AND coalesce(m.game_time_seconds, 0.0) >= $since_game_seconds
            MATCH (source)-[:PARTICIPATES_IN_RELATIONSHIP]->(rel:Relationship)<-[:PARTICIPATES_IN_RELATIONSHIP]-(listener:NPC)
            WHERE listener.id <> source.id
            WITH source, listener, rel, m,
                 (coalesce(rel.friendship, 0.2) * 0.4 + coalesce(rel.trust, 0.2) * 0.45 + coalesce(rel.respect, 0.2) * 0.15) AS spread_strength
            WHERE spread_strength >= 0.12
            MERGE (g:GossipItem {id: ('gossip-' + source.id + '-' + listener.id + '-' + m.id)})
            ON CREATE SET
                g.owner_id = listener.id,
                g.origin_npc_id = source.id,
                g.origin_memory_id = m.id,
                g.text = substring(coalesce(m.summary, ''), 0, 220),
                g.kind = 'social_rumor',
                g.spread_count = 0,
                g.sim_managed = true
            WITH g, spread_strength
            SET
                g.spread_count = coalesce(g.spread_count, 0) + 1,
                g.last_spread_game_seconds = $now_game_seconds,
                g.importance = CASE
                    WHEN (coalesce(g.importance, 0.2) * 0.72 + spread_strength * 0.38) < 0.0 THEN 0.0
                    WHEN (coalesce(g.importance, 0.2) * 0.72 + spread_strength * 0.38) > 1.0 THEN 1.0
                    ELSE (coalesce(g.importance, 0.2) * 0.72 + spread_strength * 0.38)
                END
            """,
            {
                "since_game_seconds": max(0.0, self._world_time_seconds - 120.0),
                "now_game_seconds": self._world_time_seconds,
            },
        )
        await self.graph.query(
            """
            MATCH (g:GossipItem {sim_managed: true})
            WITH g, (coalesce(g.importance, 0.2) * 0.95 + 0.01) AS candidate
            SET g.importance = CASE
                WHEN candidate < 0.0 THEN 0.0
                WHEN candidate > 1.0 THEN 1.0
                ELSE candidate
            END
            """
        )

    async def _run_trait_evolution_tick(self) -> None:
        await self.graph.query(
            """
            MATCH (t:Trait {sim_managed: true})
            OPTIONAL MATCH (t)-[:AMPLIFIES]->(amp:Trait {sim_managed: true})
            WITH t, coalesce(avg(coalesce(amp.intensity, 0.4)), 0.0) AS amp_avg
            OPTIONAL MATCH (t)-[:SUPPRESSES]->(sup:Trait {sim_managed: true})
            WITH t, amp_avg, coalesce(avg(coalesce(sup.intensity, 0.4)), 0.0) AS suppress_avg
            OPTIONAL MATCH (t)-[:TENSIONS_WITH]-(tense:Trait {sim_managed: true})
            WITH t, amp_avg, suppress_avg, coalesce(avg(coalesce(tense.intensity, 0.4)), 0.0) AS tension_avg
            WITH t,
                (coalesce(t.intensity, 0.4) * 0.93)
                + (amp_avg * 0.05)
                - (suppress_avg * 0.04)
                + (tension_avg * 0.02)
                + 0.006 AS intensity_candidate
            SET
                t.intensity = CASE
                    WHEN intensity_candidate < 0.1 THEN 0.1
                    WHEN intensity_candidate > 1.0 THEN 1.0
                    ELSE intensity_candidate
                END,
                t.last_activated_tick = $tick
            """,
            {"tick": self._tick_index},
        )

    async def _record_autonomous_action_to_graph(
        self,
        host_id: str,
        action_summary: str,
        next_goal: str,
        target_entity_id: str,
        target_region: str,
        intent_alignment: str,
    ) -> None:
        action_id = f"auto-{uuid4().hex[:12]}"
        await self.graph.query(
            """
            MATCH (host:NPC {id: $host_id}), (p:PossessionState {id: $possession_id})
            CREATE (a:AutonomousAction {
                id: $action_id,
                action_summary: $action_summary,
                next_goal: $next_goal,
                target_entity_id: $target_entity_id,
                target_region: $target_region,
                intent_alignment: $intent_alignment,
                game_time_seconds: $game_time_seconds
            })
            MERGE (host)-[:PERFORMED_AUTONOMOUS_ACTION]->(a)
            MERGE (p)-[:AUTONOMOUS_DURING_COOLDOWN]->(a)
            """,
            {
                "host_id": host_id,
                "possession_id": POSSESSION_NODE_ID,
                "action_id": action_id,
                "action_summary": action_summary,
                "next_goal": next_goal,
                "target_entity_id": target_entity_id,
                "target_region": target_region,
                "intent_alignment": intent_alignment,
                "game_time_seconds": self._world_time_seconds,
            },
        )

    async def _create_action_sequence_for_command(
        self,
        command_id: str,
        reasoning_output: dict[str, Any],
        command_duration_limit_game_minutes: int,
    ) -> None:
        sequence_id = f"seq-{uuid4().hex[:12]}"
        created_tick = int(self._world_time_seconds * self.tick_hz)
        expires_tick = created_tick + max(1, int(command_duration_limit_game_minutes * 75))

        steps: list[dict[str, Any]] = []
        step_index = 0
        for block_name, default_ms in (
            ("pre_actions", 1200),
            ("during_actions", 1800),
            ("post_actions", 1200),
        ):
            for content in reasoning_output.get(block_name, []):
                text = str(content).strip()
                if not text:
                    continue
                steps.append(
                    {
                        "id": f"step-{uuid4().hex[:12]}",
                        "step_index": step_index,
                        "type": block_name,
                        "duration_ms": default_ms,
                        "can_interrupt": True,
                        "interrupt_priority": 1,
                        "content": text,
                        "preconditions": [],
                        "abort_triggers": [],
                    }
                )
                step_index += 1

        if not steps:
            interpretation = str(reasoning_output.get("interpretation", "Carry out the compelled command.")).strip()
            if interpretation:
                steps.append(
                    {
                        "id": f"step-{uuid4().hex[:12]}",
                        "step_index": 0,
                        "type": "during_actions",
                        "duration_ms": 1800,
                        "can_interrupt": True,
                        "interrupt_priority": 1,
                        "content": interpretation,
                        "preconditions": [],
                        "abort_triggers": [],
                    }
                )

        await self.graph.query(
            """
            MATCH (host:NPC {id: $host_id}), (cmd:PlayerCommand {id: $command_id})
            CREATE (seq:ActionSequence {
                id: $sequence_id,
                owner_id: $host_id,
                status: 'active',
                created_tick: $created_tick,
                current_step_idx: 0,
                goals: $goals,
                triggered_by: 'player_command',
                player_command_id: $command_id,
                expires_tick: $expires_tick,
                next_step_due_game_seconds: $next_step_due_game_seconds
            })
            MERGE (host)-[:HAS_ACTIVE_SEQUENCE]->(seq)
            MERGE (cmd)-[:TRIGGERED]->(seq)
            """,
            {
                "host_id": self._host_id,
                "command_id": command_id,
                "sequence_id": sequence_id,
                "created_tick": created_tick,
                "expires_tick": expires_tick,
                "goals": [str(goal) for goal in reasoning_output.get("goals", []) if str(goal).strip()],
                "next_step_due_game_seconds": self._world_time_seconds + 0.2,
            },
        )

        await self.graph.query(
            """
            MATCH (seq:ActionSequence {id: $sequence_id})
            UNWIND $steps AS step
            CREATE (s:ActionStep {
                id: step.id,
                sequence_id: $sequence_id,
                step_index: step.step_index,
                type: step.type,
                duration_ms: step.duration_ms,
                can_interrupt: step.can_interrupt,
                interrupt_priority: step.interrupt_priority,
                content: step.content,
                preconditions: step.preconditions,
                abort_triggers: step.abort_triggers
            })
            MERGE (seq)-[:HAS_STEP]->(s)
            """,
            {"sequence_id": sequence_id, "steps": steps},
        )

        self._log_event(
            summary="Action sequence queued",
            details=[
                f"Sequence: {sequence_id}",
                f"Command: {command_id}",
                f"Steps: {len(steps)}",
                f"Expires at tick: {expires_tick}",
            ],
            category="system",
        )

    async def _advance_action_sequences(self) -> None:
        current_tick = int(self._world_time_seconds * self.tick_hz)
        rows = await self.graph.query(
            """
            MATCH (host:NPC {id: 'host'})-[:HAS_ACTIVE_SEQUENCE]->(seq:ActionSequence {status: 'active'})
            OPTIONAL MATCH (seq)-[:HAS_STEP]->(step:ActionStep {step_index: seq.current_step_idx})
            RETURN
                seq.id AS sequence_id,
                seq.current_step_idx AS current_step_idx,
                seq.expires_tick AS expires_tick,
                seq.next_step_due_game_seconds AS next_step_due_game_seconds,
                step.id AS step_id,
                step.content AS step_content,
                step.type AS step_type,
                step.duration_ms AS step_duration_ms,
                step.preconditions AS step_preconditions,
                step.abort_triggers AS step_abort_triggers,
                step.can_interrupt AS step_can_interrupt
            ORDER BY seq.created_tick ASC
            """
        )

        host = next((entity for entity in self._world_entities if str(entity.get("id")) == self._host_id), None)

        for row in rows:
            sequence_id = str(row.get("sequence_id", ""))
            if not sequence_id:
                continue

            expires_tick = int(row.get("expires_tick", current_tick + 1) or current_tick + 1)
            if current_tick > expires_tick:
                await self.graph.query(
                    """
                    MATCH (:NPC {id:'host'})-[r:HAS_ACTIVE_SEQUENCE]->(seq:ActionSequence {id: $sequence_id})
                    SET seq.status = 'interrupted', seq.interrupted_tick = $current_tick
                    DELETE r
                    """,
                    {"sequence_id": sequence_id, "current_tick": current_tick},
                )
                self._log_event(
                    summary="Action sequence interrupted",
                    details=[f"Sequence: {sequence_id}", "Reason: command duration limit reached"],
                    category="system",
                )
                continue

            next_due = float(row.get("next_step_due_game_seconds", 0.0) or 0.0)
            if self._world_time_seconds < next_due:
                continue

            should_abort, abort_reason = self._step_abort_triggered(row.get("step_abort_triggers"), host)
            if should_abort and bool(row.get("step_can_interrupt", True)):
                await self.graph.query(
                    """
                    MATCH (:NPC {id:'host'})-[r:HAS_ACTIVE_SEQUENCE]->(seq:ActionSequence {id: $sequence_id})
                    SET
                        seq.status = 'interrupted',
                        seq.interrupted_tick = $current_tick,
                        seq.interrupted_reason = $reason
                    DELETE r
                    """,
                    {"sequence_id": sequence_id, "current_tick": current_tick, "reason": abort_reason},
                )
                self._log_event(
                    summary="Action sequence interrupted",
                    details=[f"Sequence: {sequence_id}", f"Reason: {abort_reason}"],
                    category="system",
                )
                continue

            preconditions_met, precondition_reason = self._step_preconditions_met(
                row.get("step_preconditions"), host
            )
            if not preconditions_met:
                await self.graph.query(
                    """
                    MATCH (seq:ActionSequence {id: $sequence_id})
                    SET
                        seq.next_step_due_game_seconds = $next_due,
                        seq.waiting_on_precondition = $reason
                    """,
                    {
                        "sequence_id": sequence_id,
                        "next_due": self._world_time_seconds + 0.5,
                        "reason": precondition_reason,
                    },
                )
                self._log_event(
                    summary="Action step delayed",
                    details=[f"Sequence: {sequence_id}", f"Waiting for: {precondition_reason}"],
                    category="system",
                )
                continue

            step_id = row.get("step_id")
            step_content = str(row.get("step_content", "")).strip()
            if not step_id or not step_content:
                await self.graph.query(
                    """
                    MATCH (:NPC {id:'host'})-[r:HAS_ACTIVE_SEQUENCE]->(seq:ActionSequence {id: $sequence_id})
                    SET seq.status = 'completed', seq.completed_tick = $current_tick
                    DELETE r
                    """,
                    {"sequence_id": sequence_id, "current_tick": current_tick},
                )
                self._log_event(
                    summary="Action sequence completed",
                    details=[f"Sequence: {sequence_id}"],
                    category="interaction",
                )
                continue

            self.ui.stream_narrative(f"[Action] {step_content}")
            self._log_event(
                summary="Action step executed",
                details=[
                    f"Sequence: {sequence_id}",
                    f"Step: {row.get('current_step_idx', 0)}",
                    f"Type: {row.get('step_type', 'step')}",
                    f"Content: {step_content}",
                ],
                category="interaction",
            )

            step_duration_ms = int(row.get("step_duration_ms", 1200) or 1200)
            await self.graph.query(
                """
                MATCH (seq:ActionSequence {id: $sequence_id})
                SET
                    seq.current_step_idx = coalesce(seq.current_step_idx, 0) + 1,
                    seq.next_step_due_game_seconds = $next_due,
                    seq.waiting_on_precondition = null
                """,
                {
                    "sequence_id": sequence_id,
                    "next_due": self._world_time_seconds + max(0.1, step_duration_ms / 1000.0),
                },
            )

    def _normalize_step_conditions(self, raw: Any) -> list[str]:
        if raw is None:
            return []

        source = raw if isinstance(raw, list) else [raw]
        conditions: list[str] = []
        for item in source:
            candidate = ""
            if isinstance(item, dict):
                candidate = str(
                    item.get("name")
                    or item.get("condition")
                    or item.get("trigger")
                    or item.get("value")
                    or ""
                )
            else:
                candidate = str(item)
            token = candidate.strip().lower()
            if token:
                conditions.append(token)
        return conditions

    def _step_preconditions_met(self, preconditions_raw: Any, host: dict[str, Any] | None) -> tuple[bool, str]:
        conditions = self._normalize_step_conditions(preconditions_raw)
        if not conditions:
            return True, ""

        health = float((host or {}).get("health", 100.0) or 100.0)
        stamina = float((host or {}).get("stamina", 100.0) or 100.0)
        for condition in conditions:
            if ("alive" in condition or "conscious" in condition) and health <= 0.0:
                return False, "host must be alive"
            if ("stamina" in condition or "not_exhausted" in condition) and stamina <= 5.0:
                return False, "host must have stamina"
            if ("cooldown_active" in condition or "command_locked" in condition) and self.possession.can_command:
                return False, "cooldown must remain active"
            if ("command_ready" in condition or "cooldown_complete" in condition) and not self.possession.can_command:
                return False, "cooldown must be complete"
        return True, ""

    def _step_abort_triggered(self, abort_triggers_raw: Any, host: dict[str, Any] | None) -> tuple[bool, str]:
        triggers = self._normalize_step_conditions(abort_triggers_raw)
        if not triggers:
            return False, ""

        health = float((host or {}).get("health", 100.0) or 100.0)
        stamina = float((host or {}).get("stamina", 100.0) or 100.0)
        for trigger in triggers:
            if (
                "host_down" in trigger
                or "incapacitated" in trigger
                or "unconscious" in trigger
                or "self_harm" in trigger
            ) and (health <= 0.0 or stamina <= 0.0):
                return True, "host incapacitated"
            if (
                "new_command" in trigger
                or "player_interrupt" in trigger
                or "command_ready" in trigger
                or "cooldown_complete" in trigger
            ) and self.possession.can_command:
                return True, "player can issue new command"
            if "extreme_resistance" in trigger and self.possession.host_resistance >= 0.97:
                return True, "extreme host resistance"
        return False, ""

    async def _record_command_to_graph(self, command: str, reasoning_output: dict[str, Any]) -> str:
        command_id = f"cmd-{uuid4().hex[:12]}"
        command_tick = int(self._world_time_seconds * self.tick_hz)
        interpreted_action = str(reasoning_output.get("interpretation", ""))
        compliance_style = str(reasoning_output.get("compliance_style", "neutral"))
        outcome_summary = " | ".join(
            [
                interpreted_action,
                str(reasoning_output.get("internal_reaction", "")).strip(),
            ]
        ).strip(" |")
        await self.graph.query(
            """
            MATCH (p:PossessionState {id: $possession_id})
            CREATE (c:PlayerCommand {
                id: $command_id,
                text: $command_text,
                issued_at_game_seconds: $issued_at_game_seconds,
                compliance_style: $compliance_style,
                interpretation: $interpretation,
                likely_success: $likely_success,
                effort_level: $effort_level,
                spoken_words: $spoken_words,
                internal_reaction: $internal_reaction,
                tick: $tick,
                raw_text: $raw_text,
                interpreted_action: $interpreted_action,
                host_response_type: $host_response_type,
                outcome_summary: $outcome_summary,
                host_internal_reaction: $host_internal_reaction
            })
            MERGE (p)-[:RECEIVED_COMMAND {issued_at_game_seconds: $issued_at_game_seconds}]->(c)
            SET p.commands_obeyed = coalesce(p.commands_obeyed, 0) + 1
            """,
            {
                "possession_id": POSSESSION_NODE_ID,
                "command_id": command_id,
                "command_text": command,
                "issued_at_game_seconds": self._world_time_seconds,
                "compliance_style": compliance_style,
                "interpretation": interpreted_action,
                "likely_success": float(reasoning_output.get("likely_success", 0.5) or 0.5),
                "effort_level": float(reasoning_output.get("effort_level", 0.5) or 0.5),
                "spoken_words": str(reasoning_output.get("spoken_words", "")),
                "internal_reaction": str(reasoning_output.get("internal_reaction", "")),
                "tick": command_tick,
                "raw_text": command,
                "interpreted_action": interpreted_action,
                "host_response_type": compliance_style,
                "outcome_summary": outcome_summary,
                "host_internal_reaction": str(reasoning_output.get("internal_reaction", "")),
            },
        )
        return command_id

    async def _ensure_graph_backed_state(self) -> None:
        serialized_regions = [self._serialize_region_for_graph(region) for region in self._regions]
        serialized_roads = [self._serialize_road_for_graph(road) for road in self._roads]
        serialized_buildings = [self._serialize_building_for_graph(building) for building in self._buildings]

        await self.graph.query(
            """
            UNWIND $regions AS region
            MERGE (r:Region {id: region.id})
            SET r += region, r.sim_managed = true
            """,
            {"regions": serialized_regions},
        )
        await self.graph.query(
            """
            MATCH (r:Region {sim_managed: true})
            WHERE NOT r.id IN $region_ids
            DETACH DELETE r
            """,
            {"region_ids": [str(region["id"]) for region in self._regions]},
        )

        await self.graph.query(
            """
            UNWIND $roads AS road
            MERGE (rd:Road {id: road.id})
            SET rd += road, rd.sim_managed = true
            """,
            {"roads": serialized_roads},
        )
        await self.graph.query(
            """
            MATCH (rd:Road {sim_managed: true})
            WHERE NOT rd.id IN $road_ids
            DETACH DELETE rd
            """,
            {"road_ids": [str(road["id"]) for road in self._roads]},
        )

        await self.graph.query(
            """
            UNWIND $settlements AS settlement
            MERGE (s:Settlement {id: settlement.id})
            SET s += settlement, s.sim_managed = true
            """,
            {"settlements": self._settlements},
        )
        await self.graph.query(
            """
            MATCH (s:Settlement {sim_managed: true})
            WHERE NOT s.id IN $settlement_ids
            DETACH DELETE s
            """,
            {"settlement_ids": [str(settlement["id"]) for settlement in self._settlements]},
        )

        await self.graph.query(
            """
            UNWIND $buildings AS building
            MERGE (b:Building {id: building.id})
            SET b += building, b.sim_managed = true
            """,
            {"buildings": serialized_buildings},
        )
        await self.graph.query(
            """
            MATCH (b:Building {sim_managed: true})
            WHERE NOT b.id IN $building_ids
            DETACH DELETE b
            """,
            {"building_ids": [str(building["id"]) for building in self._buildings]},
        )

        await self.graph.query(
            """
            UNWIND $settlements AS settlement
            MATCH (s:Settlement {id: settlement.id}), (r:Region {id: settlement.region_id})
            MERGE (r)-[:HAS_SETTLEMENT]->(s)
            """,
            {"settlements": self._settlements},
        )

        await self.graph.query(
            """
            UNWIND $buildings AS building
            MATCH (b:Building {id: building.id}), (r:Region {id: building.region_id})
            MERGE (r)-[:HAS_BUILDING]->(b)
            """,
            {"buildings": self._buildings},
        )

        await self.graph.query(
            """
            UNWIND $roads AS road
            MATCH (rd:Road {id: road.id})
            UNWIND coalesce(road.connects_region_ids, []) AS region_id
            MATCH (r:Region {id: region_id})
            MERGE (rd)-[:CONNECTS_REGION]->(r)
            """,
            {"roads": self._roads},
        )
        await self.graph.query(
            """
            UNWIND $roads AS road
            MATCH (rd:Road {id: road.id})
            UNWIND coalesce(road.connects_building_ids, []) AS building_id
            MATCH (b:Building {id: building_id})
            MERGE (rd)-[:CONNECTS_BUILDING]->(b)
            """,
            {"roads": self._roads},
        )

        await self._sync_economy_graph()

        await self.graph.query(
            """
            UNWIND $entities AS entity
            MERGE (n:NPC {id: entity.id})
            SET n += entity,
                n.name = coalesce(entity.label, entity.id),
                n.pos_x = entity.x,
                n.pos_y = entity.y,
                n.is_host = coalesce(entity.kind, 'npc') = 'host',
                n.alive = coalesce(n.alive, true),
                n.health = coalesce(n.health, 100.0),
                n.stamina = coalesce(n.stamina, 100.0),
                n.occupation = coalesce(n.occupation, 'citizen'),
                n.sim_managed = true
            """,
            {"entities": self._world_entities},
        )
        await self.graph.query(
            """
            MATCH (n:NPC {sim_managed: true})
            WHERE NOT n.id IN $entity_ids
            DETACH DELETE n
            """,
            {"entity_ids": [str(entity["id"]) for entity in self._world_entities]},
        )

        await self.graph.query(
            """
            MATCH (host:NPC {id: $host_id})
            MERGE (p:PossessionState {id: $possession_id})
            SET p.host_id = $host_id
            MERGE (host)-[:IS_POSSESSED_BY]->(p)
            """,
            {"host_id": self._host_id, "possession_id": POSSESSION_NODE_ID},
        )

        await self._sync_psychology_graph()

        rows = await self.graph.query(
            """
            MERGE (p:PossessionState {id: $id})
            ON CREATE SET
                p.cooldown_total_game_hours = $cooldown_total_game_hours,
                p.cooldown_remaining_game_hours = $cooldown_remaining_game_hours,
                p.cooldown_total = $cooldown_total_game_hours,
                p.cooldown_remaining = $cooldown_remaining_game_hours,
                p.host_resistance = $host_resistance,
                p.host_willpower = $host_willpower,
                p.trust_in_player = $trust_in_player,
                p.understanding_of_player = $understanding_of_player,
                p.approval_of_player = $approval_of_player,
                p.commands_obeyed = $commands_obeyed,
                p.commands_resisted = $commands_resisted,
                p.commands_subverted = $commands_subverted,
                p.harm_from_player_commands = $harm_from_player_commands,
                p.benefit_from_player_commands = $benefit_from_player_commands,
                p.last_command_tick = 0
            RETURN properties(p) AS props
            """,
            {
                "id": POSSESSION_NODE_ID,
                "cooldown_total_game_hours": self.possession.cooldown_total_game_hours,
                "cooldown_remaining_game_hours": self.possession.cooldown_remaining_game_hours,
                "host_resistance": self.possession.host_resistance,
                "host_willpower": self.possession.host_willpower,
                "trust_in_player": self.possession.trust_in_player,
                "understanding_of_player": self.possession.understanding_of_player,
                "approval_of_player": self.possession.approval_of_player,
                "commands_obeyed": self.possession.commands_obeyed,
                "commands_resisted": self.possession.commands_resisted,
                "commands_subverted": self.possession.commands_subverted,
                "harm_from_player_commands": self.possession.harm_from_player_commands,
                "benefit_from_player_commands": self.possession.benefit_from_player_commands,
            },
        )

        if rows:
            props = rows[0]["props"]
            self.possession.cooldown_total_game_hours = float(
                props.get("cooldown_total_game_hours", self.possession.cooldown_total_game_hours)
            )
            self.possession.cooldown_remaining_game_hours = float(
                props.get("cooldown_remaining_game_hours", self.possession.cooldown_remaining_game_hours)
            )
            self.possession.host_resistance = float(
                props.get("host_resistance", self.possession.host_resistance)
            )
            self.possession.host_willpower = float(props.get("host_willpower", self.possession.host_willpower))
            self.possession.trust_in_player = float(props.get("trust_in_player", self.possession.trust_in_player))
            self.possession.understanding_of_player = float(
                props.get("understanding_of_player", self.possession.understanding_of_player)
            )
            self.possession.approval_of_player = float(
                props.get("approval_of_player", self.possession.approval_of_player)
            )
            self.possession.commands_obeyed = int(props.get("commands_obeyed", self.possession.commands_obeyed))
            self.possession.commands_resisted = int(
                props.get("commands_resisted", self.possession.commands_resisted)
            )
            self.possession.commands_subverted = int(
                props.get("commands_subverted", self.possession.commands_subverted)
            )
            self.possession.harm_from_player_commands = float(
                props.get("harm_from_player_commands", self.possession.harm_from_player_commands)
            )
            self.possession.benefit_from_player_commands = float(
                props.get("benefit_from_player_commands", self.possession.benefit_from_player_commands)
            )

        entity_rows = await self.graph.query(
            "MATCH (n:NPC {sim_managed: true}) RETURN properties(n) AS props ORDER BY n.id"
        )
        if entity_rows:
            self._world_entities = [self._normalize_entity(row["props"]) for row in entity_rows]

        region_rows = await self.graph.query(
            "MATCH (r:Region {sim_managed: true}) RETURN properties(r) AS props ORDER BY r.id"
        )
        if region_rows:
            self._regions = [self._normalize_region(row["props"]) for row in region_rows]

        road_rows = await self.graph.query(
            "MATCH (rd:Road {sim_managed: true}) RETURN properties(rd) AS props ORDER BY rd.id"
        )
        if road_rows:
            self._roads = [self._normalize_road(row["props"]) for row in road_rows]

        building_rows = await self.graph.query(
            "MATCH (b:Building {sim_managed: true}) RETURN properties(b) AS props ORDER BY b.id"
        )
        if building_rows:
            self._buildings = [self._normalize_building(row["props"]) for row in building_rows]

        logger.info("Loaded world state from Neo4j (%s entities, %s regions)", len(self._world_entities), len(self._regions))

    def _normalize_entity(self, props: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(props.get("id", "entity-unknown")),
            "label": str(props.get("label", props.get("id", "Unknown"))),
            "kind": str(props.get("kind", "npc")),
            "x": float(props.get("x", 0.0)),
            "y": float(props.get("y", 0.0)),
            "vx": float(props.get("vx", 0.0)),
            "vy": float(props.get("vy", 0.0)),
            "patrol_target_x": float(props.get("patrol_target_x", props.get("x", 0.0))),
            "patrol_target_y": float(props.get("patrol_target_y", props.get("y", 0.0))),
            "speed": float(props.get("speed", 40.0)),
            "faction": str(props.get("faction", "unknown")),
            "coins": int(props.get("coins", 0) or 0),
            "health": float(props.get("health", 100.0) or 100.0),
            "stamina": float(props.get("stamina", 100.0) or 100.0),
            "inventory": [str(item) for item in props.get("inventory", []) if str(item).strip()],
        }

    def _serialize_entity_for_graph(self, entity: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(entity.get("id", "entity-unknown")),
            "label": str(entity.get("label", entity.get("id", "Unknown"))),
            "kind": str(entity.get("kind", "npc")),
            "x": float(entity.get("x", 0.0)),
            "y": float(entity.get("y", 0.0)),
            "vx": float(entity.get("vx", 0.0)),
            "vy": float(entity.get("vy", 0.0)),
            "patrol_target_x": float(entity.get("patrol_target_x", entity.get("x", 0.0))),
            "patrol_target_y": float(entity.get("patrol_target_y", entity.get("y", 0.0))),
            "speed": float(entity.get("speed", 40.0)),
            "faction": str(entity.get("faction", "unknown")),
            "coins": int(entity.get("coins", 0) or 0),
            "health": float(entity.get("health", 100.0) or 100.0),
            "stamina": float(entity.get("stamina", 100.0) or 100.0),
            "occupation": str(entity.get("occupation", "citizen")),
            "alive": bool(entity.get("alive", True)),
            "inventory": [str(item) for item in entity.get("inventory", []) if str(item).strip()],
        }

    def _normalize_building(self, props: dict[str, Any]) -> dict[str, Any]:
        row = dict(props)
        footprint_value = row.get("footprint")
        if not isinstance(footprint_value, list):
            footprint_json = row.get("footprint_json")
            if isinstance(footprint_json, str) and footprint_json.strip():
                try:
                    footprint_value = json.loads(footprint_json)
                except json.JSONDecodeError:
                    footprint_value = []
            else:
                footprint_value = []

        footprint: list[dict[str, float]] = []
        for point in footprint_value:
            if isinstance(point, dict):
                footprint.append(
                    {
                        "x": float(point.get("x", 0.0)),
                        "y": float(point.get("y", 0.0)),
                    }
                )
        row["footprint"] = footprint
        return row

    async def _sync_graph_state(self) -> None:
        entities_for_graph = [self._serialize_entity_for_graph(entity) for entity in self._world_entities]
        await self.graph.query(
            """
            UNWIND $entities AS entity
            MATCH (n:NPC {id: entity.id})
            SET n += entity,
                n.name = coalesce(entity.label, entity.id),
                n.pos_x = entity.x,
                n.pos_y = entity.y,
                n.is_host = coalesce(entity.kind, 'npc') = 'host'
            """,
            {"entities": entities_for_graph},
        )

        await self._sync_inventory_graph()

        await self.graph.query(
            """
            MATCH (p:PossessionState {id: $id})
            SET
                p.cooldown_total_game_hours = $cooldown_total_game_hours,
                p.cooldown_remaining_game_hours = $cooldown_remaining_game_hours,
                p.cooldown_total = $cooldown_total_game_hours,
                p.cooldown_remaining = $cooldown_remaining_game_hours,
                p.host_resistance = $host_resistance,
                p.host_willpower = $host_willpower,
                p.trust_in_player = $trust_in_player,
                p.understanding_of_player = $understanding_of_player,
                p.approval_of_player = $approval_of_player,
                p.commands_obeyed = $commands_obeyed,
                p.commands_resisted = $commands_resisted,
                p.commands_subverted = $commands_subverted,
                p.harm_from_player_commands = $harm_from_player_commands,
                p.benefit_from_player_commands = $benefit_from_player_commands
            """,
            {
                "id": POSSESSION_NODE_ID,
                "cooldown_total_game_hours": self.possession.cooldown_total_game_hours,
                "cooldown_remaining_game_hours": self.possession.cooldown_remaining_game_hours,
                "host_resistance": self.possession.host_resistance,
                "host_willpower": self.possession.host_willpower,
                "trust_in_player": self.possession.trust_in_player,
                "understanding_of_player": self.possession.understanding_of_player,
                "approval_of_player": self.possession.approval_of_player,
                "commands_obeyed": self.possession.commands_obeyed,
                "commands_resisted": self.possession.commands_resisted,
                "commands_subverted": self.possession.commands_subverted,
                "harm_from_player_commands": self.possession.harm_from_player_commands,
                "benefit_from_player_commands": self.possession.benefit_from_player_commands,
            },
        )

    async def _emit_world_interactions_from_graph(self) -> None:
        rows = await self.graph.query(
            """
            MATCH (a:NPC {sim_managed: true}), (b:NPC {sim_managed: true})
            WHERE a.id < b.id
            OPTIONAL MATCH (rel:Relationship {id: ('rel-' + a.id + '-' + b.id)})
            WITH a, b, rel, sqrt((a.x - b.x)^2 + (a.y - b.y)^2) AS dist
            WHERE dist <= $interaction_distance
            RETURN
                a.id AS left_id,
                a.label AS left_label,
                coalesce(a.faction, 'unknown') AS left_faction,
                coalesce(a.is_host, false) AS left_is_host,
                b.id AS right_id,
                b.label AS right_label,
                coalesce(b.faction, 'unknown') AS right_faction,
                coalesce(b.is_host, false) AS right_is_host,
                ((a.x + b.x) / 2.0) AS mid_x,
                ((a.y + b.y) / 2.0) AS mid_y,
                dist AS distance,
                coalesce(rel.friendship, 0.2) AS friendship,
                coalesce(rel.trust, 0.2) AS trust,
                coalesce(rel.respect, 0.2) AS respect,
                coalesce(rel.interaction_count, 0) AS interaction_count
            """,
            {"interaction_distance": 42.0},
        )

        random.shuffle(rows)
        newly_started = 0

        for row in rows:
            if len(self._active_social_interactions) >= self._max_active_social_interactions:
                break
            if newly_started >= self._max_new_social_interactions_per_tick:
                break
            if len(self._pending_social_interaction_tasks) >= self._max_pending_social_plans:
                break

            pair_key = "::".join(sorted([str(row["left_id"]), str(row["right_id"])]))
            if pair_key in self._active_social_interactions:
                continue
            if pair_key in self._pending_social_interaction_tasks:
                continue

            left_id = str(row.get("left_id", ""))
            right_id = str(row.get("right_id", ""))
            if not left_id or not right_id:
                continue
            if left_id in self._motion_locked_entity_ids or right_id in self._motion_locked_entity_ids:
                continue

            last_seen = self._interaction_last_seen.get(pair_key, -9999.0)
            if self._world_time_seconds - last_seen < self._interaction_cooldown_seconds:
                continue

            self._interaction_last_seen[pair_key] = self._world_time_seconds
            region_name = self._region_for_point(float(row["mid_x"]), float(row["mid_y"]))
            self._lock_entities_for_interaction(left_id, right_id)
            row_copy = dict(row)
            row_copy["region_name"] = region_name
            planning_task = asyncio.create_task(self._plan_interaction_state_async(pair_key, row_copy))
            self._pending_social_interaction_tasks[pair_key] = planning_task
            self._pending_social_interaction_meta[pair_key] = {
                "left_id": left_id,
                "right_id": right_id,
                "left_label": str(row.get("left_label", left_id)),
                "right_label": str(row.get("right_label", right_id)),
                "region_name": region_name,
                "started_tick": self._tick_index,
            }

            self._log_event(
                summary="Interaction planning",
                details=[
                    f"Participants: {row_copy['left_label']} + {row_copy['right_label']}",
                    f"Region: {region_name}",
                    "Status: participants motion-locked while generating episode",
                ],
                category="interaction",
            )
            newly_started += 1

    async def _plan_interaction_state_async(
        self, pair_key: str, row: dict[str, Any]
    ) -> dict[str, Any] | None:
        try:
            async with self._social_planning_semaphore:
                interaction = await asyncio.wait_for(
                    self._select_interaction_variant(row),
                    timeout=self._social_planning_timeout_seconds,
                )
                episode_turns, summary_override = await asyncio.wait_for(
                    self._build_interaction_episode(row, interaction),
                    timeout=self._social_planning_timeout_seconds,
                )
        except (asyncio.TimeoutError, TimeoutError):
            logger.info("Interaction planning timed out for pair %s; interaction skipped", pair_key)
            return None
        except Exception:
            logger.exception("Interaction planning failed for pair %s; interaction skipped", pair_key)
            return None

        if not episode_turns:
            return None

        return {
            "pair_key": pair_key,
            "left_id": str(row.get("left_id", "")),
            "right_id": str(row.get("right_id", "")),
            "left_label": str(row.get("left_label", "")),
            "right_label": str(row.get("right_label", "")),
            "left_faction": str(row.get("left_faction", "unknown")),
            "right_faction": str(row.get("right_faction", "unknown")),
            "left_is_host": bool(row.get("left_is_host", False)),
            "right_is_host": bool(row.get("right_is_host", False)),
            "friendship": float(row.get("friendship", 0.2) or 0.2),
            "trust": float(row.get("trust", 0.2) or 0.2),
            "respect": float(row.get("respect", 0.2) or 0.2),
            "interaction_count": int(row.get("interaction_count", 0) or 0),
            "region_name": str(row.get("region_name", "unknown")),
            "distance": float(row.get("distance", 0.0) or 0.0),
            "interaction": interaction,
            "summary_override": summary_override,
            "turns": episode_turns,
            "turn_index": 0,
            "transcript": [],
            "executed_turns": [],
            "started_game_time_seconds": self._world_time_seconds,
        }

    async def _collect_ready_social_interactions(self) -> None:
        if not self._pending_social_interaction_tasks:
            return

        for pair_key, task in list(self._pending_social_interaction_tasks.items()):
            if not task.done():
                continue

            meta = self._pending_social_interaction_meta.pop(pair_key, {})
            self._pending_social_interaction_tasks.pop(pair_key, None)
            self._unlock_entities_for_interaction(
                str(meta.get("left_id", "")),
                str(meta.get("right_id", "")),
            )

            try:
                planned_state = task.result()
            except asyncio.CancelledError:
                continue
            except Exception:
                logger.exception("Planned interaction task crashed for pair %s", pair_key)
                continue

            if not planned_state:
                continue
            if pair_key in self._active_social_interactions:
                continue
            if not self._is_interaction_pair_still_valid(planned_state):
                self._log_event(
                    summary="Interaction cancelled",
                    details=[
                        f"Participants: {planned_state.get('left_label', 'left')} + {planned_state.get('right_label', 'right')}",
                        "Reason: participants no longer nearby/valid after planning delay",
                    ],
                    category="interaction",
                )
                continue

            self._active_social_interactions[pair_key] = planned_state
            interaction = planned_state.get("interaction", {})
            turns = planned_state.get("turns", [])
            self._log_event(
                summary=f"Interaction begins: {interaction.get('kind', 'interaction')}",
                details=[
                    f"Participants: {planned_state.get('left_label', 'left')} + {planned_state.get('right_label', 'right')}",
                    f"Region: {planned_state.get('region_name', 'unknown')}",
                    f"Planned turns: {len(turns) if isinstance(turns, list) else 0}",
                    f"Tone: {interaction.get('tone', 'neutral')}",
                ],
                category="interaction",
            )

    def _lock_entities_for_interaction(self, left_id: str, right_id: str) -> None:
        if left_id:
            self._motion_locked_entity_ids.add(left_id)
        if right_id:
            self._motion_locked_entity_ids.add(right_id)

    def _unlock_entities_for_interaction(self, left_id: str, right_id: str) -> None:
        if left_id:
            self._motion_locked_entity_ids.discard(left_id)
        if right_id:
            self._motion_locked_entity_ids.discard(right_id)

    def _is_interaction_pair_still_valid(self, state: dict[str, Any]) -> bool:
        left_id = str(state.get("left_id", ""))
        right_id = str(state.get("right_id", ""))
        left = next((entity for entity in self._world_entities if str(entity.get("id", "")) == left_id), None)
        right = next((entity for entity in self._world_entities if str(entity.get("id", "")) == right_id), None)
        if left is None or right is None:
            return False
        left_alive = bool(left.get("alive", True))
        right_alive = bool(right.get("alive", True))
        if not left_alive or not right_alive:
            return False
        dx = float(left.get("x", 0.0)) - float(right.get("x", 0.0))
        dy = float(left.get("y", 0.0)) - float(right.get("y", 0.0))
        return math.hypot(dx, dy) <= 72.0

    async def _advance_active_social_interactions(self) -> None:
        if not self._active_social_interactions:
            return

        for pair_key in list(self._active_social_interactions.keys()):
            state = self._active_social_interactions.get(pair_key)
            if state is None:
                continue

            turns = state.get("turns", [])
            if not isinstance(turns, list) or not turns:
                await self._finalize_social_interaction(state)
                self._active_social_interactions.pop(pair_key, None)
                continue

            turn_index = int(state.get("turn_index", 0))
            if turn_index >= len(turns):
                await self._finalize_social_interaction(state)
                self._active_social_interactions.pop(pair_key, None)
                continue

            turn = turns[turn_index]
            speaker_id = str(turn.get("speaker_id", state.get("left_id", "")))
            if speaker_id == str(state.get("right_id")):
                speaker_label = str(state.get("right_label", speaker_id))
            else:
                speaker_label = str(state.get("left_label", speaker_id))

            spoken_words = str(turn.get("spoken_words", "")).strip()
            action = str(turn.get("action", "")).strip()
            tone = str(turn.get("tone", "neutral")).strip() or "neutral"
            speaker_is_host = (
                (speaker_id == str(state.get("left_id", "")) and bool(state.get("left_is_host", False)))
                or (speaker_id == str(state.get("right_id", "")) and bool(state.get("right_is_host", False)))
            )

            if speaker_is_host and spoken_words:
                spoken_words = _validate_speech(spoken_words)

            host_involved = bool(state.get("left_is_host", False)) or bool(state.get("right_is_host", False))
            if host_involved and spoken_words:
                self.ui.stream_narrative(f"[{speaker_label} | {tone}] \"{spoken_words}\"")
            if host_involved and action:
                self.ui.stream_narrative(f"{speaker_label} {action}.")

            transcript_line = f"{speaker_label} ({tone}): {spoken_words}"
            if action:
                transcript_line += f" | action: {action}"
            state.setdefault("transcript", []).append(transcript_line)
            state.setdefault("executed_turns", []).append(
                {
                    "speaker_id": speaker_id,
                    "speaker_is_host": speaker_is_host,
                    "spoken_words": spoken_words,
                    "action": action,
                    "tone": tone,
                    "emotional_shift": str(turn.get("emotional_shift", "")).strip(),
                }
            )
            state["turn_index"] = turn_index + 1

            if int(state["turn_index"]) >= len(turns):
                await self._finalize_social_interaction(state)
                self._active_social_interactions.pop(pair_key, None)

    async def _finalize_social_interaction(self, state: dict[str, Any]) -> None:
        interaction = state.get("interaction", {})
        summary = str(state.get("summary_override") or interaction.get("summary") or "Interaction concluded")
        derived_friendship_delta, derived_trust_delta, derived_respect_delta = await self._derive_episode_relationship_deltas(
            state
        )

        base_friendship_delta = float(interaction.get("friendship_delta", 0.0))
        base_trust_delta = float(interaction.get("trust_delta", 0.0))
        base_respect_delta = float(interaction.get("respect_delta", 0.0))

        def _clamp_episode_delta(value: float) -> float:
            return max(-0.09, min(0.09, value))

        final_friendship_delta = _clamp_episode_delta(base_friendship_delta + derived_friendship_delta)
        final_trust_delta = _clamp_episode_delta(base_trust_delta + derived_trust_delta)
        final_respect_delta = _clamp_episode_delta(base_respect_delta + derived_respect_delta)

        details = [
            f"Region: {state.get('region_name', 'unknown')}",
            f"Distance: {float(state.get('distance', 0.0)):.1f} units",
            f"Factions: {state.get('left_faction', 'unknown')} + {state.get('right_faction', 'unknown')}",
            f"Type: {interaction.get('kind', 'interaction')}",
            f"Tone: {interaction.get('tone', 'neutral')}",
            (
                "Relationship deltas (base+episode): "
                f"friendship {base_friendship_delta:+.3f}+{derived_friendship_delta:+.3f}, "
                f"trust {base_trust_delta:+.3f}+{derived_trust_delta:+.3f}, "
                f"respect {base_respect_delta:+.3f}+{derived_respect_delta:+.3f}"
            ),
        ]
        details.extend([str(item) for item in interaction.get("details", [])])
        details.extend([f"Turn {index + 1}: {line}" for index, line in enumerate(state.get("transcript", []))])

        self._log_event(summary=summary, details=details, category="interaction")
        await self._record_interaction_to_graph(
            left_id=str(state.get("left_id", "")),
            right_id=str(state.get("right_id", "")),
            summary=summary,
            details=details,
            kind=str(interaction.get("kind", "interaction")),
            emotional_tags=[str(tag) for tag in interaction.get("emotional_tags", [])],
            intensity=float(interaction.get("intensity", 0.45)),
            friendship_delta=final_friendship_delta,
            trust_delta=final_trust_delta,
            respect_delta=final_respect_delta,
        )

        if str(interaction.get("kind", "")) == "robbery_attempt" or random.random() < 0.08:
            await self._attempt_robbery(str(state.get("left_id", "")), str(state.get("right_id", "")) )

    async def _build_interaction_episode(
        self, row: dict[str, Any], interaction: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str]:
        left_id = str(row.get("left_id", ""))
        right_id = str(row.get("right_id", ""))
        left_context = await self._load_social_context_for_npc(left_id)
        right_context = await self._load_social_context_for_npc(right_id)

        try:
            episode = await self.reasoning.run_with_graph_access(
                system_prompt=(
                    "Generate a short multi-turn social interaction episode for simulation. "
                    "You must use graph tools before finalizing (inspect, edges_out, edges_in, traverse, semantic_search). "
                    "Return graph_evidence citing real nodes found via tool results. "
                    "Include evidence for BOTH participants. "
                    "Each turn must include speaker_id, spoken_words, and action. "
                    "Do not use canned or generic template dialogue."
                ),
                task_prompt=(
                    "Simulate a realistic interaction over multiple turns, taking into account past experiences, "
                    "faction context, base stats, trait/belief signals, and relationship history."
                ),
                seed_context={
                    "selected_interaction": interaction,
                    "left": left_context,
                    "right": right_context,
                    "pair_snapshot": {
                        "friendship": float(row.get("friendship", 0.2) or 0.2),
                        "trust": float(row.get("trust", 0.2) or 0.2),
                        "respect": float(row.get("respect", 0.2) or 0.2),
                        "interaction_count": int(row.get("interaction_count", 0) or 0),
                    },
                },
                output_schema=INTERACTION_EPISODE_SCHEMA,
                graph_tools=self.graph_tools,
                max_rounds=5,
            )
        except Exception:
            logger.debug("Interaction episode generation failed for pair %s/%s", left_id, right_id, exc_info=True)
            return [], ""

        if not isinstance(episode, dict):
            return [], ""

        evidence_raw = episode.get("graph_evidence", [])
        if isinstance(evidence_raw, list) and evidence_raw:
            has_valid_evidence = await self._validate_episode_graph_evidence(evidence_raw, left_id, right_id)
            if not has_valid_evidence:
                logger.debug("Generated episode evidence weak/invalid for %s/%s; accepting turns without evidence gate", left_id, right_id)

        turns_raw = episode.get("turns", [])
        if not isinstance(turns_raw, list):
            return [], ""

        valid_speakers = {left_id, right_id}
        cleaned_turns: list[dict[str, Any]] = []
        for turn in turns_raw[:8]:
            if not isinstance(turn, dict):
                continue
            speaker_id = str(turn.get("speaker_id", "")).strip()
            if speaker_id not in valid_speakers:
                speaker_id = left_id if len(cleaned_turns) % 2 == 0 else right_id
            spoken_words = str(turn.get("spoken_words", "")).strip()
            action = str(turn.get("action", "")).strip()
            tone = str(turn.get("tone", "neutral")).strip() or "neutral"
            emotional_shift = str(turn.get("emotional_shift", "")).strip()
            if not spoken_words and not action:
                continue
            cleaned_turns.append(
                {
                    "speaker_id": speaker_id,
                    "spoken_words": spoken_words,
                    "action": action,
                    "tone": tone,
                    "emotional_shift": emotional_shift,
                }
            )

        if len(cleaned_turns) < 2:
            return [], ""

        return cleaned_turns, str(episode.get("summary_override", "")).strip()

    async def _validate_episode_graph_evidence(
        self, evidence_raw: list[Any], left_id: str, right_id: str
    ) -> bool:
        allowed_tools = {"inspect", "edges_out", "edges_in", "traverse", "semantic_search"}
        owner_targets = {left_id, right_id}

        node_ids: list[str] = []
        owners_found: set[str] = set()
        for item in evidence_raw[:10]:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("node_id", "")).strip()
            owner_id = str(item.get("owner_id", "")).strip()
            source_tool = str(item.get("source_tool", "")).strip().lower()
            if not node_id or source_tool not in allowed_tools:
                continue
            node_ids.append(node_id)
            if owner_id in owner_targets:
                owners_found.add(owner_id)

        if len(node_ids) < 2:
            return False
        if owners_found != owner_targets:
            return False

        rows = await self.graph.query(
            """
            UNWIND $node_ids AS node_id
            OPTIONAL MATCH (n {id: node_id})
            RETURN collect(n.id) AS found_ids
            """,
            {"node_ids": node_ids},
        )
        if not rows:
            return False

        found_ids = {str(item) for item in rows[0].get("found_ids", []) if item}
        valid_count = sum(1 for node_id in node_ids if node_id in found_ids)
        return valid_count >= 2

    async def _load_social_context_for_npc(self, npc_id: str) -> dict[str, Any]:
        rows = await self.graph.query(
            """
            MATCH (n:NPC {id: $npc_id})
            OPTIONAL MATCH (n)-[:HAS_EXPERIENCE]->(e:Experience)
            WITH n, collect(e.description)[0..6] AS experiences
            OPTIONAL MATCH (n)-[:GENERATED_MEMORY]->(m:Memory)
            WITH n, experiences, collect(m.summary)[0..8] AS memories
            OPTIONAL MATCH (n)-[:HAS_GOAL]->(g:Goal)
            WITH n, experiences, memories, collect(g.description)[0..5] AS goals
            OPTIONAL MATCH (n)-[:HAS_TRAIT]->(t:Trait)
            WITH n, experiences, memories, goals,
                collect({
                    label: coalesce(t.label, ''),
                    description: coalesce(t.description, ''),
                    intensity: coalesce(t.intensity, 0.0)
                })[0..8] AS traits
            OPTIONAL MATCH (n)-[:HAS_BELIEF]->(b:Belief)
            WITH n, experiences, memories, goals, traits,
                collect({
                    content: coalesce(b.content, ''),
                    confidence: coalesce(b.confidence, 0.0)
                })[0..6] AS beliefs
            OPTIONAL MATCH (n)-[:HAS_CAPABILITY]->(c:Capability)
            RETURN
                properties(n) AS npc,
                experiences,
                memories,
                goals,
                traits,
                beliefs,
                collect(c.label)[0..6] AS capabilities
            LIMIT 1
            """,
            {"npc_id": npc_id},
        )
        if not rows:
            return {"id": npc_id}
        row = rows[0]
        npc_props = row.get("npc", {}) if isinstance(row.get("npc"), dict) else {}
        return {
            "id": npc_id,
            "label": str(npc_props.get("label", npc_id)),
            "faction": str(npc_props.get("faction", "unknown")),
            "occupation": str(npc_props.get("occupation", "citizen")),
            "health": float(npc_props.get("health", 100.0)),
            "stamina": float(npc_props.get("stamina", 100.0)),
            "coins": int(npc_props.get("coins", 0) or 0),
            "memories": [str(item) for item in row.get("memories", []) if item],
            "experiences": [str(item) for item in row.get("experiences", []) if item],
            "goals": [str(item) for item in row.get("goals", []) if item],
            "capabilities": [str(item) for item in row.get("capabilities", []) if item],
            "traits": [item for item in row.get("traits", []) if isinstance(item, dict)],
            "beliefs": [item for item in row.get("beliefs", []) if isinstance(item, dict)],
        }

    async def _derive_episode_relationship_deltas(self, state: dict[str, Any]) -> tuple[float, float, float]:
        turns = state.get("executed_turns", [])
        if not isinstance(turns, list) or not turns:
            return (0.0, 0.0, 0.0)
        max_episode_shift = 0.028

        def _clamp(value: Any) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return 0.0
            return max(-max_episode_shift, min(max_episode_shift, number))

        transcript = [
            {
                "speaker_id": str(turn.get("speaker_id", "")),
                "speaker_is_host": bool(turn.get("speaker_is_host", False)),
                "spoken_words": str(turn.get("spoken_words", "")),
                "action": str(turn.get("action", "")),
                "tone": str(turn.get("tone", "neutral")),
                "emotional_shift": str(turn.get("emotional_shift", "")),
            }
            for turn in turns[:12]
            if isinstance(turn, dict)
        ]
        if not transcript:
            return (0.0, 0.0, 0.0)

        try:
            llm_result = await self.reasoning.run_with_graph_access(
                system_prompt=(
                    "Score social relationship changes from a completed interaction episode. "
                    "Use nuanced reasoning over dialogue, actions, tone shifts, faction context, and prior relationship. "
                    "Do not use simplistic keyword counting. Return JSON only."
                ),
                task_prompt=(
                    "Estimate friendship_delta, trust_delta, and respect_delta caused by this episode only. "
                    "Use small values in [-0.028, 0.028]."
                ),
                seed_context={
                    "left": {
                        "id": str(state.get("left_id", "")),
                        "label": str(state.get("left_label", "")),
                        "faction": str(state.get("left_faction", "unknown")),
                        "is_host": bool(state.get("left_is_host", False)),
                    },
                    "right": {
                        "id": str(state.get("right_id", "")),
                        "label": str(state.get("right_label", "")),
                        "faction": str(state.get("right_faction", "unknown")),
                        "is_host": bool(state.get("right_is_host", False)),
                    },
                    "relationship_before": {
                        "friendship": float(state.get("friendship", 0.2) or 0.2),
                        "trust": float(state.get("trust", 0.2) or 0.2),
                        "respect": float(state.get("respect", 0.2) or 0.2),
                        "interaction_count": int(state.get("interaction_count", 0) or 0),
                    },
                    "interaction_kind": str(state.get("interaction", {}).get("kind", "interaction")),
                    "interaction_tone": str(state.get("interaction", {}).get("tone", "neutral")),
                    "transcript": transcript,
                },
                output_schema=EPISODE_RELATIONSHIP_DELTA_SCHEMA,
                graph_tools=self.graph_tools,
                max_rounds=4,
            )
        except Exception:
            logger.debug("Episode delta LLM scoring failed; using neutral deltas", exc_info=True)
            return (0.0, 0.0, 0.0)

        if not isinstance(llm_result, dict):
            return (0.0, 0.0, 0.0)

        confidence = float(llm_result.get("confidence", 0.5) or 0.5)
        if confidence < 0.15:
            return (0.0, 0.0, 0.0)

        friendship_delta = _clamp(llm_result.get("friendship_delta", 0.0))
        trust_delta = _clamp(llm_result.get("trust_delta", 0.0))
        respect_delta = _clamp(llm_result.get("respect_delta", 0.0))
        return (friendship_delta, trust_delta, respect_delta)

    async def _select_interaction_variant(self, row: dict[str, Any]) -> dict[str, Any]:
        interaction = await self._generate_interaction_variant_llm(row)
        if interaction is None:
            raise RuntimeError("Interaction variant generation unavailable")
        self._llm_interaction_last_time = self._world_time_seconds
        return interaction

    async def _generate_interaction_variant_llm(self, row: dict[str, Any]) -> dict[str, Any] | None:
        left_label = str(row.get("left_label", row.get("left_id", "Unknown")))
        right_label = str(row.get("right_label", row.get("right_id", "Unknown")))

        try:
            dynamic = await self.reasoning.run_with_graph_access(
                system_prompt=(
                    "You generate social interaction variants for a systemic simulation. "
                    "Use graph tools before finalizing. Return JSON only. "
                    "Do not use canned template interaction names or stock dialogue phrasing."
                ),
                task_prompt=(
                    f"Generate one grounded interaction variant between {left_label} and {right_label}. "
                    "Return kind, tone, summary, details, emotional_tags, intensity, and relationship deltas."
                ),
                seed_context={
                    "left": {
                        "id": str(row.get("left_id", "")),
                        "label": left_label,
                        "faction": str(row.get("left_faction", "unknown")),
                        "is_host": bool(row.get("left_is_host", False)),
                    },
                    "right": {
                        "id": str(row.get("right_id", "")),
                        "label": right_label,
                        "faction": str(row.get("right_faction", "unknown")),
                        "is_host": bool(row.get("right_is_host", False)),
                    },
                    "relationship": {
                        "friendship": float(row.get("friendship", 0.2) or 0.2),
                        "trust": float(row.get("trust", 0.2) or 0.2),
                        "respect": float(row.get("respect", 0.2) or 0.2),
                        "interaction_count": int(row.get("interaction_count", 0) or 0),
                    },
                    "host_possession": {
                        "host_resistance": self.possession.host_resistance,
                        "trust_in_player": self.possession.trust_in_player,
                        "approval_of_player": self.possession.approval_of_player,
                    },
                },
                output_schema=INTERACTION_VARIANT_SCHEMA,
                graph_tools=self.graph_tools,
                max_rounds=4,
            )
        except Exception:
            logger.debug("Interaction variant generation failed", exc_info=True)
            return None

        if not isinstance(dynamic, dict):
            return None
        return self._normalize_interaction_variant(dynamic)

    def _normalize_interaction_variant(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        kind = str(payload.get("kind", "")).strip() or "social_exchange"
        summary = str(payload.get("summary", "")).strip() or "A social exchange unfolds between nearby NPCs."

        tone = str(payload.get("tone", "mixed")).strip() or "mixed"
        details_raw = payload.get("details", [])
        details = [str(item) for item in details_raw if str(item).strip()] if isinstance(details_raw, list) else []
        tags_raw = payload.get("emotional_tags", [])
        emotional_tags = [str(item) for item in tags_raw if str(item).strip()] if isinstance(tags_raw, list) else []

        def _cap_delta(value: Any) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return 0.0
            return max(-0.06, min(0.06, number))

        intensity = float(payload.get("intensity", 0.45) or 0.45)
        return {
            "kind": kind,
            "tone": tone,
            "summary": summary,
            "details": details,
            "friendship_delta": _cap_delta(payload.get("friendship_delta", 0.0)),
            "trust_delta": _cap_delta(payload.get("trust_delta", 0.0)),
            "respect_delta": _cap_delta(payload.get("respect_delta", 0.0)),
            "emotional_tags": emotional_tags,
            "intensity": max(0.05, min(0.98, intensity)),
            "weight": 1.0,
        }

    async def _record_interaction_to_graph(
        self,
        left_id: str,
        right_id: str,
        summary: str,
        details: list[str],
        kind: str,
        emotional_tags: list[str],
        intensity: float,
        friendship_delta: float,
        trust_delta: float,
        respect_delta: float,
    ) -> None:
        memory_id = f"memory-{uuid4().hex[:12]}"
        await self.graph.query(
            """
            MATCH (a:NPC {id: $left_id}), (b:NPC {id: $right_id})
            CREATE (m:Memory {
                id: $memory_id,
                owner_id: $left_id,
                kind: $kind,
                summary: $summary,
                details: $details,
                emotional_tags: $emotional_tags,
                intensity: $intensity,
                embedding: [],
                game_time_seconds: $game_time_seconds
            })
            MERGE (a)-[:GENERATED_MEMORY]->(m)
            MERGE (b)-[:GENERATED_MEMORY]->(m)
            MERGE (rel:Relationship {id: $relationship_id})
            ON CREATE SET
                rel.agent_a_id = $left_id,
                rel.agent_b_id = $right_id,
                rel.friendship = 0.2,
                rel.trust = 0.2,
                rel.respect = 0.2,
                rel.interaction_count = 0
            SET
                rel.interaction_count = coalesce(rel.interaction_count, 0) + 1,
                rel.friendship = CASE
                    WHEN coalesce(rel.friendship, 0.2) + $friendship_delta_param < 0.0 THEN 0.0
                    WHEN coalesce(rel.friendship, 0.2) + $friendship_delta_param > 1.0 THEN 1.0
                    ELSE coalesce(rel.friendship, 0.2) + $friendship_delta_param
                END,
                rel.trust = CASE
                    WHEN coalesce(rel.trust, 0.2) + $trust_delta_param < 0.0 THEN 0.0
                    WHEN coalesce(rel.trust, 0.2) + $trust_delta_param > 1.0 THEN 1.0
                    ELSE coalesce(rel.trust, 0.2) + $trust_delta_param
                END,
                rel.respect = CASE
                    WHEN coalesce(rel.respect, 0.2) + $respect_delta_param < 0.0 THEN 0.0
                    WHEN coalesce(rel.respect, 0.2) + $respect_delta_param > 1.0 THEN 1.0
                    ELSE coalesce(rel.respect, 0.2) + $respect_delta_param
                END,
                rel.last_interaction_game_seconds = $game_time_seconds
            MERGE (a)-[:PARTICIPATES_IN_RELATIONSHIP]->(rel)
            MERGE (b)-[:PARTICIPATES_IN_RELATIONSHIP]->(rel)
            """,
            {
                "left_id": left_id,
                "right_id": right_id,
                "memory_id": memory_id,
                "relationship_id": f"rel-{'-'.join(sorted([left_id, right_id]))}",
                "kind": kind,
                "summary": summary,
                "details": details,
                "emotional_tags": emotional_tags,
                "intensity": intensity,
                "game_time_seconds": self._world_time_seconds,
                "friendship_delta_param": friendship_delta,
                "trust_delta_param": trust_delta,
                "respect_delta_param": respect_delta,
            },
        )

    def _region_for_point(self, x: float, y: float) -> str:
        for region in self._regions:
            vertices = region.get("vertices")
            if isinstance(vertices, list) and vertices and self._point_in_polygon(x, y, vertices):
                return str(region["label"])
        return "Outer Reach"

    def _point_in_polygon(self, x: float, y: float, vertices: list[dict[str, Any]]) -> bool:
        inside = False
        vertex_count = len(vertices)
        if vertex_count < 3:
            return False
        j = vertex_count - 1
        for i in range(vertex_count):
            xi = float(vertices[i].get("x", 0.0))
            yi = float(vertices[i].get("y", 0.0))
            xj = float(vertices[j].get("x", 0.0))
            yj = float(vertices[j].get("y", 0.0))
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / max(yj - yi, 0.000001) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def _position_for_region(self, region_hint: str) -> tuple[float, float] | None:
        normalized = region_hint.strip().lower()
        if not normalized:
            return None
        for region in self._regions:
            region_id = str(region.get("id", "")).lower()
            label = str(region.get("label", "")).lower()
            if normalized in {region_id, label} or normalized in label:
                seed_x = float(region.get("seed_x", region.get("x", self._region_width / 2.0)))
                seed_y = float(region.get("seed_y", region.get("y", self._region_height / 2.0)))
                jitter_radius = 45.0
                angle = random.uniform(0.0, math.tau)
                radius = random.uniform(0.0, jitter_radius)
                px = max(10.0, min(self._region_width - 10.0, seed_x + math.cos(angle) * radius))
                py = max(10.0, min(self._region_height - 10.0, seed_y + math.sin(angle) * radius))
                return (px, py)
        return None

    def _publish_world_state(self) -> None:
        if not hasattr(self.ui, "update_minimap"):
            return

        host = next((entity for entity in self._world_entities if str(entity.get("id", "")) == self._host_id), None)
        host_payload: dict[str, Any] | None = None
        if host is not None:
            host_x = float(host.get("x", 0.0))
            host_y = float(host.get("y", 0.0))
            host_payload = {
                "id": str(host.get("id", self._host_id)),
                "label": str(host.get("label", self._host_id)),
                "x": host_x,
                "y": host_y,
                "region": self._region_for_point(host_x, host_y),
                "health": float(host.get("health", 0.0)),
                "stamina": float(host.get("stamina", 0.0)),
            }

        combat_markers: list[dict[str, Any]] = []
        for encounter in self._active_combat_encounters.values():
            left_id = str(encounter.get("left_id", ""))
            right_id = str(encounter.get("right_id", ""))
            if not left_id or not right_id:
                continue

            left = next((entity for entity in self._world_entities if str(entity.get("id")) == left_id), None)
            right = next((entity for entity in self._world_entities if str(entity.get("id")) == right_id), None)
            if left is None or right is None:
                continue

            center_x = (float(left.get("x", 0.0)) + float(right.get("x", 0.0))) / 2.0
            center_y = (float(left.get("y", 0.0)) + float(right.get("y", 0.0))) / 2.0
            combat_markers.append(
                {
                    "id": str(encounter.get("id", "")),
                    "x": center_x,
                    "y": center_y,
                    "left_id": left_id,
                    "right_id": right_id,
                    "left_label": str(left.get("label", left_id)),
                    "right_label": str(right.get("label", right_id)),
                    "rounds": int(encounter.get("rounds", 0) or 0),
                }
            )

        self.ui.update_minimap(
            {
                "width": self._region_width,
                "height": self._region_height,
                "km_width": self._map_km_width,
                "km_height": self._map_km_height,
                "regions": self._regions,
                "roads": self._roads,
                "buildings": self._buildings,
                "markets": self._markets,
                "black_markets": [item for item in self._black_markets if bool(item.get("active", False))],
                "hunting_areas": [item for item in self._hunting_areas if bool(item.get("active", False))],
                "combat_encounters": combat_markers,
                "host": host_payload,
                "tick": self._tick_index,
                "async_state": {
                    "command_processing": self._command_task is not None,
                    "active_command": self._active_command_text,
                    "social_scan_running": self._social_emit_task is not None,
                    "pending_social_plans": len(self._pending_social_interaction_tasks),
                    "pending_combat_narrations": len(self._pending_combat_narration_tasks),
                    "active_social_interactions": len(self._active_social_interactions),
                    "locked_entity_count": len(self._motion_locked_entity_ids),
                },
                "entities": [
                    {
                        "id": str(entity["id"]),
                        "label": str(entity["label"]),
                        "kind": str(entity["kind"]),
                        "x": float(entity["x"]),
                        "y": float(entity["y"]),
                        "faction": str(entity.get("faction", "unknown")),
                        "motion_locked": str(entity.get("id", "")) in self._motion_locked_entity_ids,
                    }
                    for entity in self._world_entities
                ],
            }
        )
        logger.debug("Published minimap state for %s entities", len(self._world_entities))

    def _log_event(self, summary: str, details: list[str], category: str) -> None:
        if hasattr(self.ui, "log_event"):
            self.ui.log_event(summary=summary, details=details, category=category)
        logger.debug("Event logged (%s): %s", category, summary)


def _is_mental_state_command(command: str) -> bool:
    lowered = command.lower()
    blocked_fragments = [
        "be happy",
        "be sad",
        "stop being",
        "feel",
        "trust me",
        "love me",
        "don't be afraid",
    ]
    return any(fragment in lowered for fragment in blocked_fragments)


def _validate_speech(spoken_words: str) -> str:
    for pattern in FORBIDDEN_SPEECH_PATTERNS:
        if re.search(pattern, spoken_words, re.IGNORECASE):
            return "[The words catch in the host's throat. They abruptly change the subject.]"
    return spoken_words
