import json
from typing import Any

import v2
from graph import ALIASES_FILE, GRAPH_FILE, load_graph, norm_name


def get_explorer_graph() -> dict[str, Any]:
    g = load_graph()
    if g is None:
        return {"nodes": [], "edges": [], "types": {}, "available": False, "message": "Graph not built yet."}

    nodes_dict = g.get("nodes", {})
    edges_list = g.get("edges", [])

    # Calculate degrees
    degrees = {k: 0 for k in nodes_dict}
    for e in edges_list:
        h = e.get("head")
        t = e.get("tail")
        if h in degrees:
            degrees[h] += 1
        if t in degrees:
            degrees[t] += 1

    type_counts = {}
    nodes = []
    for k, data in nodes_dict.items():
        node_type = data.get("type", "other")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
        nodes.append(
            {
                "id": k,
                "label": data.get("name", k),
                "type": node_type,
                "chunks": data.get("chunks", []),
                "chunk_count": len(data.get("chunks", [])),
                "degree": degrees.get(k, 0),
            }
        )

    edges = []
    for i, e in enumerate(edges_list):
        edges.append(
            {
                "id": f"edge_{i}",
                "source": e.get("head"),
                "target": e.get("tail"),
                "relation": e.get("relation"),
                "chunks": e.get("chunks", []),
                "count": e.get("count", 1),
            }
        )

    # Color map for entity types
    type_colors = {
        "error_code": "#e63946",
        "service": "#457b9d",
        "team": "#1d3557",
        "channel": "#2a9d8f",
        "plan": "#f4a261",
        "feature": "#e76f51",
        "policy": "#9b5de5",
        "role": "#00bbf9",
        "setting": "#00f5d4",
        "other": "#6c757d",
    }

    return {
        "available": True,
        "nodes": sorted(nodes, key=lambda n: -n["degree"]),
        "edges": edges,
        "type_counts": type_counts,
        "type_colors": type_colors,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "types_count": len(type_counts),
        },
    }


def get_node_details(node_id: str) -> dict[str, Any]:
    g = load_graph()
    if g is None:
        return {"error": "Graph not built yet"}

    normalized_id = norm_name(node_id)
    nodes_dict = g.get("nodes", {})

    node_key = None
    if normalized_id in nodes_dict:
        node_key = normalized_id
    else:
        # Search by display name
        for k, v in nodes_dict.items():
            if v.get("name", "").lower() == node_id.lower() or k == node_id.lower():
                node_key = k
                break

    if not node_key or node_key not in nodes_dict:
        return {"error": f"Node '{node_id}' not found in concept graph"}

    node = nodes_dict[node_key]
    edges_list = g.get("edges", [])

    # Find connected neighbors and relations
    connections = []
    related_chunk_ids = set(node.get("chunks", []))

    for e in edges_list:
        h = e.get("head")
        t = e.get("tail")
        rel = e.get("relation")
        e_chunks = e.get("chunks", [])

        if h == node_key:
            target_node = nodes_dict.get(t, {"name": t, "type": "unknown"})
            connections.append(
                {
                    "direction": "outgoing",
                    "relation": rel,
                    "target_id": t,
                    "target_name": target_node.get("name", t),
                    "target_type": target_node.get("type", "unknown"),
                    "chunks": e_chunks,
                }
            )
            related_chunk_ids.update(e_chunks)
        elif t == node_key:
            source_node = nodes_dict.get(h, {"name": h, "type": "unknown"})
            connections.append(
                {
                    "direction": "incoming",
                    "relation": rel,
                    "target_id": h,
                    "target_name": source_node.get("name", h),
                    "target_type": source_node.get("type", "unknown"),
                    "chunks": e_chunks,
                }
            )
            related_chunk_ids.update(e_chunks)

    # Fetch chunk texts
    all_chunks = v2.all_chunks()
    by_id = {c["id"]: c for c in all_chunks}
    chunks_data = []
    for cid in sorted(related_chunk_ids):
        if cid in by_id:
            c = by_id[cid]
            chunks_data.append(
                {
                    "id": cid,
                    "doc": c.get("doc"),
                    "position": c.get("position"),
                    "text": c.get("text", "")[:400] + ("..." if len(c.get("text", "")) > 400 else ""),
                }
            )

    # Suggested query based on node and type
    name = node.get("name", node_key)
    ntype = node.get("type", "concept")
    if ntype == "error_code":
        suggested_query = f"What causes error {name} and how do I resolve it?"
    elif ntype == "service":
        suggested_query = f"Which team owns {name} and what is its on-call channel?"
    elif ntype == "plan":
        suggested_query = f"What features are included in the {name} plan?"
    elif ntype == "policy":
        suggested_query = f"What are the main rules in the {name}?"
    else:
        suggested_query = f"Tell me more about {name}."

    return {
        "id": node_key,
        "name": name,
        "type": ntype,
        "forms": list(node.get("forms", {}).keys()),
        "chunks_count": len(node.get("chunks", [])),
        "connections": connections,
        "chunks": chunks_data,
        "suggested_query": suggested_query,
    }
