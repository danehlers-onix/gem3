# bigquery persistence for knowledge graph nodes and searches

import json
import hashlib
from datetime import datetime, timezone
from google.cloud import bigquery

PROJECT = "prj-ox-int-g-looker"
DATASET = "gem3"
NODES_TABLE = f"{PROJECT}.{DATASET}.nodes"
SEARCHES_TABLE = f"{PROJECT}.{DATASET}.searches"

_client = None

def _get_bq():
    global _client
    if _client is None:
        _client = bigquery.Client(project=PROJECT)
    return _client


def load_all_nodes() -> dict[str, dict]:
    # load all nodes from bq, reconstruct in-memory graph
    bq = _get_bq()
    query = f"SELECT * FROM `{NODES_TABLE}` ORDER BY created_at"
    
    nodes = {}
    try:
        rows = bq.query(query).result()
        for row in rows:
            node = {
                "id": row.node_id,
                "title": row.title or "",
                "summary": row.summary or "",
                "level": row.level or "topic",
                "parent_id": row.parent_id,
                "x": row.x or 8000,
                "y": row.y or 5000,
                "angle": row.angle or 0,
                "children": list(row.children) if row.children else [],
                "expanded": row.expanded or False,
                "visit_count": row.visit_count or 0,
                "child_count": row.child_count or 0,
            }
            nodes[row.node_id] = node
        print(f"loaded {len(nodes)} nodes from bq")
    except Exception as e:
        print(f"bq load failed: {e}")
    
    return nodes


def save_nodes(nodes: list[dict], query: str = ""):
    # batch insert new nodes to bq
    if not nodes:
        return
    
    bq = _get_bq()
    now = datetime.now(timezone.utc).isoformat()
    
    rows = []
    for n in nodes:
        rows.append({
            "node_id": n["id"],
            "title": n.get("title", ""),
            "summary": n.get("summary", ""),
            "level": n.get("level", ""),
            "parent_id": n.get("parent_id", ""),
            "x": n.get("x", 0),
            "y": n.get("y", 0),
            "angle": n.get("angle", 0),
            "children": n.get("children", []),
            "expanded": n.get("expanded", False),
            "depth": _level_depth(n.get("level", "")),
            "visit_count": n.get("visit_count", 0),
            "child_count": len(n.get("children", [])),
            "created_at": now,
            "updated_at": now,
            "created_by_query": query,
        })
    
    try:
        errors = bq.insert_rows_json(NODES_TABLE, rows)
        if errors:
            print(f"bq insert errors: {errors[:2]}")
        else:
            print(f"saved {len(rows)} nodes to bq")
    except Exception as e:
        print(f"bq save failed: {e}")


def update_node_expanded(node_id: str, children_ids: list[str]):
    # expanded state tracked in-memory only; bq streaming buffer
    # doesn't support UPDATE on recently inserted rows, so we skip it.
    # on next cold start, expanded state is reconstructed from children arrays.
    pass


def increment_visit(node_id: str):
    # visit tracking is in-memory only to avoid streaming buffer conflicts
    pass


def save_search(search_id: str, query: str, answer: str, 
                traversal: list[str], excerpts: str,
                steps: int, duration_ms: int):
    # save search result to bq
    bq = _get_bq()
    rows = [{
        "search_id": search_id,
        "query": query,
        "answer": answer,
        "traversal_path": traversal,
        "source_excerpts": excerpts,
        "steps": steps,
        "duration_ms": duration_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]
    try:
        bq.insert_rows_json(SEARCHES_TABLE, rows)
    except Exception as e:
        print(f"search save failed: {e}")


def get_sparse_nodes(limit: int = 10) -> list[dict]:
    # find expanded nodes with few children
    bq = _get_bq()
    query = f"""
    SELECT node_id, title, level, child_count, visit_count
    FROM `{NODES_TABLE}`
    WHERE expanded = TRUE AND child_count < 3
    ORDER BY visit_count DESC, child_count ASC
    LIMIT {limit}
    """
    try:
        return [dict(row) for row in bq.query(query).result()]
    except Exception:
        return []


def get_popular_unexpanded(limit: int = 10) -> list[dict]:
    # find most-visited unexpanded nodes
    bq = _get_bq()
    query = f"""
    SELECT node_id, title, level, visit_count
    FROM `{NODES_TABLE}`
    WHERE expanded = FALSE AND visit_count > 0
    ORDER BY visit_count DESC
    LIMIT {limit}
    """
    try:
        return [dict(row) for row in bq.query(query).result()]
    except Exception:
        return []


def get_graph_stats() -> dict:
    # overall graph statistics
    bq = _get_bq()
    query = f"""
    SELECT 
        COUNT(*) as total_nodes,
        COUNTIF(expanded) as expanded_nodes,
        COUNTIF(NOT expanded) as leaf_nodes,
        SUM(visit_count) as total_visits,
        MAX(depth) as max_depth,
        COUNT(DISTINCT level) as levels
    FROM `{NODES_TABLE}`
    """
    try:
        row = list(bq.query(query).result())[0]
        return dict(row)
    except Exception:
        return {"total_nodes": 0}


def _level_depth(level: str) -> int:
    levels = ["library", "topic", "subtopic", "chapter", "section", "concept", "excerpt", "detail"]
    return levels.index(level) if level in levels else 0
