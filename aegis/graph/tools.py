from __future__ import annotations

import re
from typing import Any

from aegis.graph.connection import Neo4jConnection


class GraphTools:
    def __init__(self, graph: Neo4jConnection) -> None:
        self.graph = graph

    async def execute(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "inspect": self.inspect,
            "edges_out": self.edges_out,
            "edges_in": self.edges_in,
            "find_nodes": self.find_nodes,
            "spatial_nearby": self.spatial_nearby,
            "semantic_search": self.semantic_search,
            "traverse": self.traverse,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        return await handler(payload)

    def _safe_name(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "", value)

    def _safe_names(self, values: list[str]) -> list[str]:
        out: list[str] = []
        for value in values:
            cleaned = self._safe_name(str(value))
            if cleaned:
                out.append(cleaned)
        return out

    async def inspect(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = "MATCH (n {id:$node_id}) RETURN properties(n) AS props LIMIT 1"
        rows = await self.graph.query(query, {"node_id": payload["node_id"]})
        return {"node": rows[0]["props"] if rows else None}

    async def edges_out(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = (
            "MATCH (n {id:$node_id})-[r]->(m) "
            "WHERE $edge_type IS NULL OR type(r) = $edge_type "
            "RETURN type(r) AS edge_type, properties(r) AS edge_props, m.id AS to_id, labels(m) AS to_labels "
            "LIMIT $limit"
        )
        rows = await self.graph.query(
            query,
            {
                "node_id": payload["node_id"],
                "edge_type": payload.get("edge_type"),
                "limit": int(payload.get("limit", 20)),
            },
        )
        return {"edges": rows}

    async def edges_in(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = (
            "MATCH (m)-[r]->(n {id:$node_id}) "
            "WHERE $edge_type IS NULL OR type(r) = $edge_type "
            "RETURN type(r) AS edge_type, properties(r) AS edge_props, m.id AS from_id, labels(m) AS from_labels "
            "LIMIT $limit"
        )
        rows = await self.graph.query(
            query,
            {
                "node_id": payload["node_id"],
                "edge_type": payload.get("edge_type"),
                "limit": int(payload.get("limit", 20)),
            },
        )
        return {"edges": rows}

    async def find_nodes(self, payload: dict[str, Any]) -> dict[str, Any]:
        label = self._safe_name(str(payload["label"]))
        if not label:
            return {"nodes": [], "error": "Invalid label"}
        filters = payload.get("filters", {})
        where_parts: list[str] = []
        params: dict[str, Any] = {"limit": int(payload.get("limit", 20))}

        for key, value in filters.items():
            param_name = f"f_{key}"
            where_parts.append(f"n.{key} = ${param_name}")
            params[param_name] = value

        where = " AND ".join(where_parts) if where_parts else "true"
        query = (
            f"MATCH (n:{label}) WHERE {where} "
            "RETURN n.id AS id, labels(n) AS labels, properties(n) AS props LIMIT $limit"
        )
        rows = await self.graph.query(query, params)
        return {"nodes": rows}

    async def spatial_nearby(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("center_entity_id"):
            center_rows = await self.graph.query(
                "MATCH (n {id:$id}) RETURN n.x AS x, n.y AS y LIMIT 1",
                {"id": payload["center_entity_id"]},
            )
            if not center_rows:
                return {"entities": []}
            x = center_rows[0]["x"]
            y = center_rows[0]["y"]
        else:
            x = payload["x"]
            y = payload["y"]

        radius = float(payload.get("radius", 100))
        query = (
            "MATCH (n) WHERE exists(n.x) AND exists(n.y) "
            "WITH n, (n.x - $x)*(n.x - $x) + (n.y - $y)*(n.y - $y) AS d2 "
            "WHERE d2 <= $r2 "
            "RETURN n.id AS id, labels(n) AS labels, properties(n) AS props ORDER BY d2 ASC LIMIT 50"
        )
        rows = await self.graph.query(query, {"x": x, "y": y, "r2": radius * radius})
        return {"entities": rows}

    async def semantic_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        query_text = str(payload.get("query", "")).lower()
        node_labels = self._safe_names([str(label) for label in (payload.get("node_labels") or [])])
        scope_to_owner = str(payload.get("scope_to_owner", "")).strip()
        min_similarity = float(payload.get("min_similarity", 0.6))
        limit = int(payload.get("limit", 10))

        if node_labels:
            label_clause = ":" + ":".join(node_labels)
            match_clause = f"MATCH (n{label_clause})"
        else:
            match_clause = "MATCH (n)"

        where_clauses = [
            "toLower(coalesce(n.text,'')) CONTAINS $q "
            "OR toLower(coalesce(n.name,'')) CONTAINS $q "
            "OR toLower(coalesce(n.label,'')) CONTAINS $q "
            "OR toLower(coalesce(n.description,'')) CONTAINS $q "
            "OR toLower(coalesce(n.content,'')) CONTAINS $q"
        ]

        params: dict[str, Any] = {
            "q": query_text,
            "min_similarity": max(0.0, min(1.0, min_similarity)),
            "limit": limit,
        }

        if scope_to_owner:
            where_clauses.append(
                "n.owner_id = $scope_to_owner "
                "OR n.id = $scope_to_owner "
                "OR EXISTS { MATCH (owner {id:$scope_to_owner})-[]->(n) }"
            )
            params["scope_to_owner"] = scope_to_owner

        query = (
            f"{match_clause} "
            f"WHERE {' AND '.join(f'({clause})' for clause in where_clauses)} "
            "WITH n, [token IN split(replace(replace(replace($q, '.', ' '), ',', ' '), ';', ' '), ' ') "
            "         WHERE size(token) > 2] AS query_tokens, "
            "toLower(trim(coalesce(n.text,'') + ' ' + coalesce(n.name,'') + ' ' + coalesce(n.label,'') + ' ' + "
            "coalesce(n.description,'') + ' ' + coalesce(n.content,''))) AS haystack "
            "WITH n, query_tokens, size([t IN query_tokens WHERE haystack CONTAINS t]) AS matches, size(query_tokens) AS total "
            "WITH n, CASE WHEN total = 0 THEN 0.0 ELSE toFloat(matches) / toFloat(total) END AS similarity "
            "WHERE similarity >= $min_similarity "
            "RETURN n.id AS id, labels(n) AS labels, properties(n) AS props, similarity "
            "ORDER BY similarity DESC, n.id ASC "
            "LIMIT $limit"
        )

        rows = await self.graph.query(query, params)
        return {"results": rows}

    async def traverse(self, payload: dict[str, Any]) -> dict[str, Any]:
        pattern = payload.get("pattern", [])
        if not pattern:
            return {"paths": []}

        hops = min(len(pattern), 5)
        clauses = ["(n0 {id:$start_node_id})"]
        for index in range(hops):
            hop = pattern[index] if index < len(pattern) else {}
            direction = str(hop.get("direction", "either")).lower()
            if direction == "out":
                left, right = "-", "->"
            elif direction == "in":
                left, right = "<-", "-"
            else:
                left, right = "-", "-"

            edge_types = hop.get("edge_types")
            if edge_types is None and hop.get("edge_type"):
                edge_types = [hop.get("edge_type")]

            type_filter = self._safe_names([str(item) for item in (edge_types or [])])
            if type_filter:
                rel = f"[r{index}:{'|'.join(type_filter)}]"
            else:
                rel = f"[r{index}]"

            clauses.append(f"{left}{rel}{right}(n{index + 1})")

        query = (
            "MATCH p="
            + "".join(clauses)
            + " RETURN "
            "[node IN nodes(p) | {id: node.id, labels: labels(node)}] AS node_path, "
            "[rel IN relationships(p) | {type: type(rel), props: properties(rel)}] AS edge_path "
            "LIMIT $limit"
        )
        rows = await self.graph.query(
            query,
            {
                "start_node_id": payload["start_node_id"],
                "limit": int(payload.get("limit", 20)),
            },
        )
        return {"paths": rows}
