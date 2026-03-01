GRAPH_PRIMITIVES = [
    {
        "name": "inspect",
        "description": "View all properties of a node by ID.",
        "parameters": {"node_id": {"type": "string", "required": True}},
    },
    {
        "name": "edges_out",
        "description": "List outgoing edges from a node.",
        "parameters": {
            "node_id": {"type": "string", "required": True},
            "edge_type": {"type": "string"},
            "limit": {"type": "integer", "default": 20},
        },
    },
    {
        "name": "edges_in",
        "description": "List incoming edges to a node.",
        "parameters": {
            "node_id": {"type": "string", "required": True},
            "edge_type": {"type": "string"},
            "limit": {"type": "integer", "default": 20},
        },
    },
    {
        "name": "find_nodes",
        "description": "Find nodes by label and property filters.",
        "parameters": {
            "label": {"type": "string", "required": True},
            "filters": {"type": "object"},
            "limit": {"type": "integer", "default": 20},
        },
    },
    {
        "name": "spatial_nearby",
        "description": "Find entities near a position.",
        "parameters": {
            "center_entity_id": {"type": "string"},
            "x": {"type": "number"},
            "y": {"type": "number"},
            "radius": {"type": "number", "default": 100},
            "entity_types": {"type": "array"},
        },
    },
    {
        "name": "semantic_search",
        "description": "Find nodes semantically similar to query.",
        "parameters": {
            "query": {"type": "string", "required": True},
            "node_labels": {"type": "array"},
            "scope_to_owner": {"type": "string"},
            "min_similarity": {"type": "number", "default": 0.6},
            "limit": {"type": "integer", "default": 10},
        },
    },
    {
        "name": "traverse",
        "description": "Multi-hop graph traversal.",
        "parameters": {
            "start_node_id": {"type": "string", "required": True},
            "pattern": {"type": "array", "required": True},
            "limit": {"type": "integer", "default": 20},
        },
    },
]
