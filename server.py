# generative zoom server — expands knowledge graph on demand via gemini

import json
import math
import hashlib
import asyncio
import os
from aiohttp import web
from google import genai
from google.genai import types

from gem3.config import get_gemini_client, GEMINI_MODEL
from gem3.bq_persist import (
    load_all_nodes, save_nodes, update_node_expanded,
    save_search, get_graph_stats, increment_visit,
)
import time


CANVAS_WIDTH = 16000
CANVAS_HEIGHT = 10000

LEVEL_ORDER = ["library", "topic", "subtopic", "chapter", "section", "concept", "excerpt", "detail"]

LEVEL_FONT_SIZES = {
    "library": 800, "topic": 400, "subtopic": 150, "chapter": 60,
    "section": 30, "concept": 16, "excerpt": 10, "detail": 7,
}
LEVEL_GRAY = {
    "library": 232, "topic": 200, "subtopic": 175, "chapter": 145,
    "section": 110, "concept": 70, "excerpt": 30, "detail": 15,
}
LEVEL_SPREAD = {
    "library": 5000, "topic": 2000, "subtopic": 800, "chapter": 350,
    "section": 150, "concept": 70, "excerpt": 30, "detail": 15,
}

INITIAL_SCALE = 900 / CANVAS_WIDTH

# In-memory store of all nodes and their state
nodes_store: dict[str, dict] = {}
# Pre-built grounded tree (loaded from ingest)
grounded_tree: dict | None = None
client: genai.Client | None = None


def get_client() -> genai.Client:
    # get gemini client
    global client
    if client is None:
        client = get_gemini_client()
    return client


def _hash_angle(text: str, seed: int = 0) -> float:
    h = int(hashlib.md5(f"{text}{seed}".encode()).hexdigest()[:8], 16)
    return ((h % 61) - 30)


def _hash_offset(text: str, spread: float, seed: int = 0) -> tuple[float, float]:
    h = hashlib.md5(f"{text}{seed}".encode()).hexdigest()
    hx = int(h[:8], 16)
    hy = int(h[8:16], 16)
    angle_rad = (hx % 360) * math.pi / 180
    radius = (hy % 1000) / 1000.0 * spread
    return (math.cos(angle_rad) * radius, math.sin(angle_rad) * radius)


def _next_level(level: str) -> str:
    idx = LEVEL_ORDER.index(level) if level in LEVEL_ORDER else 0
    return LEVEL_ORDER[min(idx + 1, len(LEVEL_ORDER) - 1)]


def _make_node_id(title: str, parent_id: str) -> str:
    return hashlib.md5(f"{parent_id}:{title}".encode()).hexdigest()[:12]


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def create_seed_nodes(topics: list[str] | None = None, query: str = "") -> list[dict]:
    # create initial seed nodes
    if topics is None:
        topics = ["Science", "Technology", "Philosophy", "Mathematics", "History"]
    
    root_id = "root"
    root_node = {
        "id": root_id,
        "title": "Knowledge",
        "level": "library",
        "x": CANVAS_WIDTH / 2,
        "y": CANVAS_HEIGHT / 2,
        "expanded": False,
        "children": [],
        "parent_id": None,
        "query": query,
    }
    nodes_store[root_id] = root_node
    
    all_nodes = [root_node]
    spread = LEVEL_SPREAD["library"]
    n = len(topics)
    
    for i, topic in enumerate(topics):
        angle_rad = (2 * math.pi * i / n) + _hash_angle("Knowledge", i) * 0.02
        dx, dy = _hash_offset(topic, spread, i)
        dx += math.cos(angle_rad) * spread * 0.5
        dy += math.sin(angle_rad) * spread * 0.5
        
        node_id = _make_node_id(topic, root_id)
        node = {
            "id": node_id,
            "title": topic,
            "level": "topic",
            "x": CANVAS_WIDTH / 2 + dx,
            "y": CANVAS_HEIGHT / 2 + dy,
            "expanded": False,
            "children": [],
            "parent_id": root_id,
            "query": query,
        }
        nodes_store[node_id] = node
        root_node["children"].append(node_id)
        all_nodes.append(node)
    
    root_node["expanded"] = True
    return all_nodes


async def expand_node(request: web.Request) -> web.Response:
    # expand single node
    data = await request.json()
    node_id = data.get("node_id")
    query = data.get("query", "")
    
    if node_id not in nodes_store:
        return web.json_response({"error": "not found"}, status=404)
    
    node = nodes_store[node_id]
    if node["expanded"]:
        children = [nodes_store[cid] for cid in node["children"] if cid in nodes_store]
        return web.json_response({"nodes": children, "already_expanded": True})
    
    new_nodes = await _expand_single(node, query)
    return web.json_response({"nodes": new_nodes, "already_expanded": False})


async def expand_batch(request: web.Request) -> web.Response:
    # batch expand multiple nodes
    data = await request.json()
    node_ids = data.get("node_ids", [])
    query = data.get("query", "")
    
    # Filter to unexpanded nodes that exist
    to_expand = []
    for nid in node_ids:
        if nid in nodes_store and not nodes_store[nid]["expanded"]:
            to_expand.append(nodes_store[nid])
    
    if not to_expand:
        return web.json_response({"results": {}})
    
    # If only 1 node, use fast single path
    if len(to_expand) == 1:
        new_nodes = await _expand_single(to_expand[0], query)
        return web.json_response({"results": {to_expand[0]["id"]: new_nodes}})
    
    # BATCH: expand ALL nodes in ONE Gemini call
    results = await _expand_batch(to_expand, query)
    return web.json_response({"results": results})


async def _expand_single(node: dict, query: str) -> list[dict]:
    # single node expansion
    parent_title = node["title"]
    current_level = node["level"]
    child_level = _next_level(current_level)
    
    # Build SHORT context (just parent + grandparent)
    ctx = parent_title
    if node.get("parent_id") and node["parent_id"] in nodes_store:
        ctx = nodes_store[node["parent_id"]]["title"] + " > " + ctx
    
    # At deep levels, generate REAL excerpts (facts, quotes, key findings)
    q_hint = f' (focus: {query})' if query else ''
    if child_level in ("excerpt", "detail", "concept"):
        prompt = (f'Give 3-4 facts about "{parent_title}"{q_hint}. '
                  f'JSON: [{{"title":"label","summary":"fact"}}]')
    else:
        prompt = f'List 3-4 {child_level}s under "{parent_title}"{q_hint}. JSON: [{{"title":"..","summary":".."}}]. Short titles.'
    
    children_data = await _call_gemini(prompt)
    return _place_children(node, children_data, child_level)


async def _expand_batch(nodes: list[dict], query: str) -> dict[str, list[dict]]:
    # batch expand via single gemini call
    # Build batch prompt
    items = []
    for i, node in enumerate(nodes):
        child_level = _next_level(node["level"])
        ctx = node["title"]
        if node.get("parent_id") and node["parent_id"] in nodes_store:
            ctx = nodes_store[node["parent_id"]]["title"] + " > " + ctx
        items.append(f'{i}. "{node["title"]}" ({node["level"]}>{child_level}, ctx: {ctx})')
    
    items_str = "\n".join(items)
    q_hint = f'\nFocus on: {query}' if query else ''
    
    prompt = f"""Generate children for these {len(nodes)} knowledge nodes:{q_hint}

{items_str}

For each, generate 4-5 children. Return JSON object mapping index to array:
{{"0":[{{"title":"...","summary":"..."}}], "1":[...], ...}}
Titles: 2-4 words max. Summaries: 1 short sentence."""
    
    # Run sync Gemini call in thread pool to avoid async client issues
    batch_data = await asyncio.to_thread(_sync_gemini_dict, prompt)
    
    # Place children for each node
    results = {}
    for i, node in enumerate(nodes):
        children_data = batch_data.get(str(i), [
            {"title": f"Aspect {j+1}", "summary": f"Part of {node['title']}"}
            for j in range(4)
        ])
        child_level = _next_level(node["level"])
        results[node["id"]] = _place_children(node, children_data, child_level)
    
    return results


def _sync_gemini_dict(prompt: str) -> dict:
    # sync gemini call returning dict
    try:
        c = get_gemini_client()  # Fresh client per call — avoids async close issues
        response = c.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.8),
        )
        text = response.text.strip()
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as e:
        print(f"gemini error: {e}")
    return {}


def _sync_gemini_list(prompt: str) -> list[dict]:
    # sync gemini call returning list
    try:
        c = get_gemini_client()  # Fresh client per call
        response = c.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.8),
        )
        text = response.text.strip()
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as e:
        print(f"gemini error: {e}")
    return [{"title": f"Detail {i+1}", "summary": "..."} for i in range(4)]


async def _call_gemini(prompt: str) -> list[dict]:
    # async gemini wrapper
    return await asyncio.to_thread(_sync_gemini_list, prompt)


def _place_children(node: dict, children_data: list[dict], child_level: str) -> list[dict]:
    # position children around parent
    parent_x, parent_y = node["x"], node["y"]
    spread = LEVEL_SPREAD.get(node["level"], 100)
    parent_angle = _hash_angle(node["title"], 0)
    node_id = node["id"]
    
    new_nodes = []
    n = len(children_data) if children_data else 1
    
    for i, child_data in enumerate(children_data or []):
        title = child_data.get("title", f"Item {i}")
        child_angle_rad = (2 * math.pi * i / n) + _hash_angle(node["title"], i) * 0.02
        dx, dy = _hash_offset(title, spread, i)
        dx += math.cos(child_angle_rad) * spread * 0.5
        dy += math.sin(child_angle_rad) * spread * 0.5
        
        child_id = _make_node_id(title, node_id)
        angle = _hash_angle(title, i) * 0.7 + parent_angle * 0.3
        
        child_node = {
            "id": child_id, "title": title,
            "summary": child_data.get("summary", ""),
            "level": child_level,
            "x": max(100, min(parent_x + dx, CANVAS_WIDTH - 200)),
            "y": max(100, min(parent_y + dy, CANVAS_HEIGHT - 100)),
            "angle": angle, "expanded": False,
            "children": [], "parent_id": node_id,
        }
        
        nodes_store[child_id] = child_node
        node["children"].append(child_id)
        new_nodes.append(child_node)
    
    node["expanded"] = True
    
    
    try:
        asyncio.get_event_loop().run_in_executor(
            None, save_nodes, new_nodes, node.get("query", ""))
        asyncio.get_event_loop().run_in_executor(
            None, update_node_expanded, node_id, [c["id"] for c in new_nodes])
    except Exception:
        pass  # Don't block on BQ errors
    
    return new_nodes


def build_canvas_html(seed_nodes: list[dict], query: str = "") -> str:
    # build canvas html
    
    # Render initial seed nodes
    initial_spans = []
    for node in seed_nodes:
        level = node["level"]
        font_size = LEVEL_FONT_SIZES.get(level, 14)
        gray = LEVEL_GRAY.get(level, 100)
        angle = _hash_angle(node["title"], 0)
        
        color = f"rgb({gray},{gray},{gray})"
        weight = "400"
        if query:
            title_lower = (node["title"]).lower()
            for w in query.lower().split():
                if len(w) > 3 and w in title_lower:
                    gray = max(0, gray - 50)
                    color = f"rgb({gray},{gray},{gray})"
                    weight = "700"
                    break
        
        x = node["x"] - (len(node["title"]) * font_size * 0.28)
        y = node["y"] - font_size / 2
        x = max(50, min(x, CANVAS_WIDTH - 200))
        y = max(50, min(y, CANVAS_HEIGHT - 100))
        
        initial_spans.append(
            f'<span data-node-id="{node["id"]}" data-level="{level}" data-expanded="{str(node["expanded"]).lower()}" '
            f'style="position:absolute;left:{x:.0f}px;top:{y:.0f}px;font-size:{font_size}px;'
            f'color:{color};font-weight:{weight};transform:rotate({angle:.1f}deg);'
            f'transform-origin:left center;white-space:nowrap;line-height:1;'
            f'pointer-events:auto;cursor:pointer;">'
            f'{_escape(node["title"])}</span>'
        )
    
    spans_html = "\n".join(initial_spans)
    
    n_nodes = len(nodes_store)
    
    return f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gem3 · infinite knowledge zoom</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{
    overflow: hidden;
    width: 100vw;
    height: 100vh;
    background: #f5f5f5;
    font-family: 'Helvetica Neue', 'Arial', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
}}
#world {{
    position: absolute;
    width: {CANVAS_WIDTH}px;
    height: {CANVAS_HEIGHT}px;
    transform-origin: 0 0;
    will-change: transform;
}}
[data-node-id] {{
    transition: color 0.2s, opacity 0.5s;
}}
[data-node-id]:hover {{
    color: rgb(0,0,0) !important;
    font-weight: 700 !important;
}}
.controls {{
    position: fixed;
    bottom: 16px;
    right: 16px;
    z-index: 10000;
    display: flex;
    flex-direction: column;
    gap: 6px;
}}
.ctrl-btn {{
    width: 34px; height: 34px;
    border: 1px solid #ccc;
    background: rgba(245,245,245,0.95);
    border-radius: 50%;
    font-size: 16px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    color: #666; user-select: none;
}}
.ctrl-btn:hover {{ background: #eee; color: #222; }}
#zoom-level {{
    position: fixed; bottom: 16px; left: 16px;
    font-size: 11px; color: #bbb; z-index: 10000;
}}
#gen-status {{
    position: fixed; top: 12px; right: 12px;
    font-size: 11px; color: #999; z-index: 10000;
    background: rgba(245,245,245,0.9);
    padding: 3px 8px; border-radius: 4px;
    display: none;
}}
.hint {{
    position: fixed; bottom: 40px; left: 50%;
    transform: translateX(-50%);
    font-size: 11px; color: #ccc; z-index: 10000;
    pointer-events: none; transition: opacity 2s;
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: scale(0.7); }}
    to {{ opacity: 1; transform: scale(1); }}
}}
.node-new {{
    animation: fadeIn 0.4s ease-out;
}}
#search-box {{
    position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
    z-index: 20000; display: flex; align-items: center; gap: 0;
    background: white; border-radius: 24px; padding: 4px 4px 4px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12); width: min(560px, 90vw);
    transition: box-shadow 0.2s;
}}
#search-box:focus-within {{ box-shadow: 0 4px 16px rgba(0,0,0,0.18); }}
#search-input {{
    flex: 1; border: none; outline: none; font-size: 15px;
    background: transparent; padding: 8px 0;
    font-family: inherit;
}}
#search-btn {{
    width: 36px; height: 36px; border: none; border-radius: 50%;
    background: #4285f4; color: white; cursor: pointer;
    font-size: 16px; display: flex; align-items: center; justify-content: center;
    transition: background 0.15s;
}}
#search-btn:hover {{ background: #3367d6; }}
#search-btn:disabled {{ background: #ccc; cursor: default; }}
#search-status {{
    position: fixed; top: 72px; left: 50%; transform: translateX(-50%);
    z-index: 20000; font-size: 12px; color: #666; display: none;
    background: rgba(255,255,255,0.95); padding: 6px 16px; border-radius: 16px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.1);
}}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}
.pulsing {{ animation: pulse 1s infinite; }}
#answer-panel {{
    position: fixed; bottom: 0; left: 0; right: 0;
    z-index: 20000; max-height: 50vh; overflow-y: auto;
    background: white; border-top: 1px solid #e0e0e0;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
    padding: 20px 24px; display: none;
    font-size: 14px; line-height: 1.6;
    transform: translateY(100%); transition: transform 0.3s ease;
}}
#answer-panel.show {{ display: block; transform: translateY(0); }}
#answer-panel .close-btn {{
    position: absolute; top: 8px; right: 12px;
    background: none; border: none; font-size: 18px; cursor: pointer; color: #999;
}}
#answer-panel .path-bar {{
    display: flex; gap: 4px; align-items: center; flex-wrap: wrap;
    margin-bottom: 12px; font-size: 12px; color: #888;
}}
#answer-panel .path-node {{
    background: #f0f0f0; padding: 2px 8px; border-radius: 10px;
    cursor: pointer; transition: background 0.15s;
}}
#answer-panel .path-node:hover {{ background: #4285f4; color: white; }}
#answer-panel .answer-text {{ white-space: pre-wrap; color: #333; }}
#answer-panel .meta {{ font-size: 11px; color: #aaa; margin-top: 8px; }}
#graph-stats {{
    position: fixed; bottom: 16px; left: 50%;
    transform: translateX(-50%);
    font-size: 11px; color: #ccc; z-index: 10000;
}}
/* ── Traversal Panel (left sidebar) ── */
#traversal-panel {{
    position: fixed; top: 0; left: 0; bottom: 0; width: 280px;
    z-index: 15000; background: rgba(255,255,255,0.97);
    border-right: 1px solid #e0e0e0; overflow-y: auto;
    transform: translateX(-100%); transition: transform 0.3s ease;
    padding: 64px 12px 16px; font-size: 13px;
    box-shadow: 2px 0 12px rgba(0,0,0,0.06);
}}
#traversal-panel.open {{ transform: translateX(0); }}
#traversal-panel .tp-header {{
    font-size: 11px; color: #999; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 8px; padding: 0 4px;
}}
#traversal-panel .tp-query {{
    font-size: 14px; font-weight: 600; color: #333; margin-bottom: 12px; padding: 0 4px;
}}
.tp-step {{
    position: relative; padding-left: 24px; margin-bottom: 4px;
}}
.tp-step::before {{
    content: ''; position: absolute; left: 9px; top: 0; bottom: -4px;
    width: 2px; background: #e0e0e0;
}}
.tp-step:last-child::before {{ bottom: 50%; }}
.tp-node {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 10px; border-radius: 16px; cursor: pointer;
    transition: all 0.15s; max-width: 100%; font-size: 12px;
    background: #f5f5f5; color: #555; border: 1px solid transparent;
    position: relative;
}}
.tp-node::before {{
    content: ''; position: absolute; left: -15px; top: 50%;
    width: 12px; height: 2px; background: #e0e0e0;
}}
.tp-node:hover {{ background: #e8f0fe; color: #1a73e8; border-color: #c5d8f8; }}
.tp-node.active {{ background: #4285f4; color: white; border-color: #4285f4; font-weight: 600; }}
.tp-node .tp-level {{ font-size: 9px; opacity: 0.7; text-transform: uppercase; }}
.tp-node .tp-dot {{ width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }}
.tp-children {{
    padding-left: 16px; margin: 2px 0 4px;
}}
.tp-child {{
    font-size: 11px; color: #888; padding: 2px 8px; border-radius: 10px;
    cursor: pointer; display: inline-block; margin: 1px 2px;
    background: #fafafa; transition: all 0.15s;
}}
.tp-child:hover {{ background: #e8f0fe; color: #1a73e8; }}
.tp-thinking {{
    padding: 4px 10px 4px 24px; font-size: 11px; color: #999;
    font-style: italic;
}}
@keyframes tpPulse {{ 0%,100%{{ opacity: 1; }} 50%{{ opacity: 0.3; }} }}
.tp-thinking.active {{ animation: tpPulse 1s infinite; }}
.tp-answer {{
    margin-top: 12px; padding: 10px; border-radius: 8px;
    background: #f0f7ff; border: 1px solid #d0e4f7;
    font-size: 12px; line-height: 1.5; color: #333;
    white-space: pre-wrap;
}}
.tp-answer-label {{ font-size: 10px; color: #4285f4; font-weight: 600; margin-bottom: 4px; }}
.tp-meta {{ font-size: 10px; color: #aaa; margin-top: 6px; }}
/* ── SVG branches on canvas ── */
#branches {{
    position: absolute; top: 0; left: 0;
    width: {CANVAS_WIDTH}px; height: {CANVAS_HEIGHT}px;
    pointer-events: none;
}}
#branches line {{
    stroke: #e0e0e0; stroke-width: 1; opacity: 0.4;
}}
/* ── Search highlighting: selected vs dimmed ── */
[data-node-id].search-active {{
    color: #111 !important;
    font-weight: 900 !important;
    opacity: 1 !important;
    text-shadow: 0 0 8px rgba(66,133,244,0.3);
    z-index: 100;
}}
[data-node-id].search-path {{
    color: #333 !important;
    font-weight: 700 !important;
    opacity: 0.9 !important;
}}
[data-node-id].search-dimmed {{
    opacity: 0.12 !important;
    transition: opacity 0.4s;
}}
#node-popup {{
    position: fixed;
    z-index: 25000;
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 8px 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    max-width: 280px;
    font-size: 12px;
    line-height: 1.4;
    pointer-events: none;
    opacity: 0;
    transform: translateY(4px);
    transition: opacity 0.2s, transform 0.2s;
}}
#node-popup.show {{
    opacity: 1;
    transform: translateY(0);
}}
#node-popup .popup-title {{
    font-weight: 700;
    color: #111;
    margin-bottom: 2px;
}}
#node-popup .popup-level {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 1px 6px;
    border-radius: 8px;
    display: inline-block;
    margin-bottom: 4px;
}}
#node-popup .popup-desc {{
    color: #555;
    font-size: 11px;
}}
#node-popup .popup-stats {{
    color: #999;
    font-size: 10px;
    margin-top: 4px;
    border-top: 1px solid #f0f0f0;
    padding-top: 4px;
}}
</style>
</head>
<body>
<div id="search-box">
    <input id="search-input" type="text" placeholder="Ask anything — Gemini will navigate the knowledge graph..."
        value="{_escape(query) if query else ''}" autocomplete="off">
    <button id="search-btn" onclick="doSearch()"></button>
</div>
<div id="search-status"></div>
<div id="answer-panel">
    <button class="close-btn" onclick="closeAnswer()">✕</button>
    <div class="path-bar" id="path-bar"></div>
    <div class="answer-text" id="answer-text"></div>
    <div class="meta" id="answer-meta"></div>
</div>

<div id="traversal-panel">
    <div class="tp-header">Traversal Path</div>
    <div class="tp-query" id="tp-query"></div>
    <div id="tp-tree"></div>
    <div id="tp-status" class="tp-thinking"></div>
    <div id="tp-answer-box"></div>
</div>

<div id="world">
    <svg id="branches" xmlns="http://www.w3.org/2000/svg"></svg>
    {spans_html}
</div>

<div class="controls">
    <div class="ctrl-btn" onclick="zoomBy(1.5)">+</div>
    <div class="ctrl-btn" onclick="zoomBy(1/1.5)">−</div>
    <div class="ctrl-btn" onclick="resetView()">⟳</div>
</div>
<div id="zoom-level"></div>
<div id="gen-status">generating...</div>
<div id="graph-stats">{n_nodes} nodes · gemini 3 flash preview</div>
<div class="hint" id="hint">scroll to zoom · drag to pan · search to discover</div>
<div id="node-popup">
    <div class="popup-title" id="popup-title"></div>
    <div class="popup-level" id="popup-level"></div>
    <div class="popup-desc" id="popup-desc"></div>
    <div class="popup-stats" id="popup-stats"></div>
</div>

<script>
const world = document.getElementById('world');
const zoomLabel = document.getElementById('zoom-level');
const genStatus = document.getElementById('gen-status');
const hint = document.getElementById('hint');
const CW = {CANVAS_WIDTH};
const CH = {CANVAS_HEIGHT};
const INITIAL_SCALE = {INITIAL_SCALE};
const QUERY = "{query.replace('"', '\\"') if query else ""}";

const FONT_SIZES = {json.dumps(LEVEL_FONT_SIZES)};
const GRAY_VALUES = {json.dumps(LEVEL_GRAY)};

let scale = INITIAL_SCALE;
let panX = 0, panY = 0;

// Track expanding nodes to avoid duplicate requests
const expanding = new Set();
const expanded = new Set();

function resetView() {{
    scale = INITIAL_SCALE;
    panX = (window.innerWidth - CW * scale) / 2;
    panY = (window.innerHeight - CH * scale) / 2;
    apply();
}}

let visTimer = null;
function apply() {{
    world.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
    zoomLabel.textContent = (scale / INITIAL_SCALE).toFixed(1) + 'x';
    // Throttle visibility updates — DOM changes are expensive
    clearTimeout(visTimer);
    visTimer = setTimeout(updateVisibility, 80);
    checkExpansions();
}}
function updateVisibility() {{
    // LOD: show current layer (full) + 1 parent (grayed) + 2 children (grayed)
    // Effective pixel sweet spot: 14-30px = full opacity
    // Parent zone: 30-160px = fading gray (1 layer above)
    // Child zone: 4-14px = fading gray (2 layers below)
    // Outside 4-160px = hidden
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    
    const nodes = document.querySelectorAll('[data-node-id]');
    for (let i = 0; i < nodes.length; i++) {{
        const el = nodes[i];
        // Skip nodes with search highlighting (those are controlled by search)
        if (el.classList.contains('search-active') || el.classList.contains('search-path')) continue;
        
        const fs = parseFloat(el.style.fontSize) || 14;
        const eff = fs * scale;
        
        // Hard cutoff
        if (eff < 4 || eff > 160) {{
            el.style.display = 'none';
            continue;
        }}
        
        // Viewport culling
        const nx = parseFloat(el.style.left) || 0;
        const ny = parseFloat(el.style.top) || 0;
        const sx = nx * scale + panX;
        const sy = ny * scale + panY;
        const margin = 300;
        if (sx < -margin || sx > vw + margin || sy < -margin || sy > vh + margin) {{
            el.style.display = 'none';
            continue;
        }}
        
        el.style.display = '';
        
        if (eff >= 14 && eff <= 30) {{
            // SWEET SPOT: current layer — full visibility
            el.style.opacity = '1';
        }} else if (eff > 30 && eff <= 160) {{
            // PARENT ZONE: 1 layer above — gray fade
            const t = (eff - 30) / (160 - 30);
            el.style.opacity = Math.max(0.08, (1 - t) * 0.5).toFixed(2);
        }} else if (eff >= 4 && eff < 14) {{
            // CHILD ZONE: 2 layers below — gray fade
            const t = (eff - 4) / (14 - 4);
            el.style.opacity = Math.max(0.08, t * 0.4).toFixed(2);
        }}
    }}
}}

function zoomBy(factor) {{
    zoomAt(window.innerWidth / 2, window.innerHeight / 2, factor);
}}

function zoomAt(sx, sy, factor) {{
    const ns = Math.max(0.003, Math.min(scale * factor, 50));
    panX = sx - (sx - panX) * (ns / scale);
    panY = sy - (sy - panY) * (ns / scale);
    scale = ns;
    apply();
}}

// ── Collect visible nodes and batch-expand them ──
let batchTimer = null;
let batchPending = false;

function checkExpansions() {{
    // Debounce: wait 150ms after last zoom/pan before firing
    clearTimeout(batchTimer);
    batchTimer = setTimeout(doBatchExpand, 150);
}}

async function doBatchExpand() {{
    if (batchPending) return;
    
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const toExpand = [];
    
    document.querySelectorAll('[data-node-id]').forEach(el => {{
        const nodeId = el.dataset.nodeId;
        if (el.dataset.expanded === 'true' || expanding.has(nodeId) || expanded.has(nodeId)) return;
        
        const fontSize = FONT_SIZES[el.dataset.level] || 14;
        if (fontSize * scale < 30) return;
        
        const rect = el.getBoundingClientRect();
        if (rect.right > -100 && rect.left < vw + 100 && rect.bottom > -100 && rect.top < vh + 100) {{
            toExpand.push(nodeId);
        }}
    }});
    
    if (toExpand.length === 0) return;
    
    // Mark all as expanding
    toExpand.forEach(id => expanding.add(id));
    batchPending = true;
    genStatus.style.display = 'block';
    genStatus.textContent = `generating ${{toExpand.length}} nodes...`;
    
    try {{
        const resp = await fetch('/api/expand_batch', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ node_ids: toExpand, query: QUERY }}),
        }});
        const data = await resp.json();
        
        if (data.results) {{
            for (const [parentId, nodes] of Object.entries(data.results)) {{
                if (!nodes) continue;
                nodes.forEach(node => placeNode(node));
                const el = document.querySelector(`[data-node-id="${{parentId}}"]`);
                if (el) el.dataset.expanded = 'true';
                expanded.add(parentId);
            }}
        }}
    }} catch (err) {{
        console.error('Batch expand failed:', err);
    }}
    
    toExpand.forEach(id => expanding.delete(id));
    batchPending = false;
    genStatus.style.display = 'none';
    
    // Check again in case zoom moved during generation
    setTimeout(checkExpansions, 200);
}}

function placeNode(node) {{
    if (document.querySelector(`[data-node-id="${{node.id}}"]`)) return;
    
    const level = node.level;
    const fontSize = FONT_SIZES[level] || 14;
    let gray = GRAY_VALUES[level] || 100;
    let weight = '400';
    let color = `rgb(${{gray}},${{gray}},${{gray}})`;
    
    if (QUERY) {{
        const tl = (node.title + ' ' + (node.summary || '')).toLowerCase();
        const qw = QUERY.toLowerCase().split(' ').filter(w => w.length > 3);
        const m = qw.filter(w => tl.includes(w)).length;
        if (m > 0) {{
            const d = Math.max(0, gray - m * 50);
            color = `rgb(${{d}},${{d}},${{d}})`;
            weight = '700';
        }}
    }}
    
    const angle = node.angle || 0;
    const x = Math.max(50, Math.min(node.x - (node.title.length * fontSize * 0.28), CW - 200));
    const y = Math.max(50, Math.min(node.y - fontSize / 2, CH - 100));
    
    const span = document.createElement('span');
    span.dataset.nodeId = node.id;
    span.dataset.level = level;
    span.dataset.expanded = 'false';
    span.className = 'node-new';
    span.style.cssText = `position:absolute;left:${{x.toFixed(0)}}px;top:${{y.toFixed(0)}}px;font-size:${{fontSize}}px;color:${{color}};font-weight:${{weight}};transform:rotate(${{angle.toFixed(1)}}deg);transform-origin:left center;white-space:nowrap;line-height:1;pointer-events:auto;cursor:pointer;`;
    span.textContent = node.title;
    if (node.summary) span.title = node.summary;
    world.appendChild(span);
}}

// ── Mouse wheel zoom ──
window.addEventListener('wheel', function(e) {{
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.08 : 1 / 1.08;
    zoomAt(e.clientX, e.clientY, factor);
    hint.style.opacity = '0';
}}, {{ passive: false }});

// ── Drag to pan ──
let dragging = false, lastX, lastY;
window.addEventListener('mousedown', e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; document.body.style.cursor = 'grabbing'; }});
window.addEventListener('mousemove', e => {{ if (!dragging) return; panX += e.clientX - lastX; panY += e.clientY - lastY; lastX = e.clientX; lastY = e.clientY; apply(); }});
window.addEventListener('mouseup', () => {{ dragging = false; document.body.style.cursor = 'default'; }});

// ── Touch ──
let lastTouchDist = 0, lastTouchMid = {{x:0,y:0}};
window.addEventListener('touchstart', e => {{
    if (e.touches.length === 2) {{
        const dx = e.touches[1].clientX - e.touches[0].clientX;
        const dy = e.touches[1].clientY - e.touches[0].clientY;
        lastTouchDist = Math.sqrt(dx*dx+dy*dy);
        lastTouchMid = {{x:(e.touches[0].clientX+e.touches[1].clientX)/2, y:(e.touches[0].clientY+e.touches[1].clientY)/2}};
    }} else if (e.touches.length === 1) {{
        dragging = true; lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
    }}
}}, {{passive:false}});
window.addEventListener('touchmove', e => {{
    e.preventDefault();
    if (e.touches.length === 2) {{
        const dx = e.touches[1].clientX-e.touches[0].clientX, dy = e.touches[1].clientY-e.touches[0].clientY;
        const dist = Math.sqrt(dx*dx+dy*dy);
        const mid = {{x:(e.touches[0].clientX+e.touches[1].clientX)/2,y:(e.touches[0].clientY+e.touches[1].clientY)/2}};
        if (lastTouchDist > 0) zoomAt(mid.x, mid.y, dist/lastTouchDist);
        panX += mid.x-lastTouchMid.x; panY += mid.y-lastTouchMid.y;
        lastTouchDist = dist; lastTouchMid = mid; apply();
    }} else if (e.touches.length === 1 && dragging) {{
        panX += e.touches[0].clientX-lastX; panY += e.touches[0].clientY-lastY;
        lastX = e.touches[0].clientX; lastY = e.touches[0].clientY; apply();
    }}
}}, {{passive:false}});
window.addEventListener('touchend', () => {{ dragging = false; lastTouchDist = 0; }});

// Periodically check for expansions during zoom
let checkTimer;
function scheduleCheck() {{ clearTimeout(checkTimer); checkTimer = setTimeout(checkExpansions, 300); }}

// ── Search with animated zoom traversal ──
const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const searchStatus = document.getElementById('search-status');
const answerPanel = document.getElementById('answer-panel');
const pathBar = document.getElementById('path-bar');
const answerText = document.getElementById('answer-text');
const answerMeta = document.getElementById('answer-meta');

searchInput.addEventListener('keydown', e => {{ if (e.key === 'Enter') doSearch(); }});

function animateTo(wx, wy, targetScale, duration) {{
    return new Promise(resolve => {{
        const startScale = scale, startPanX = panX, startPanY = panY;
        const endPanX = innerWidth/2 - wx * targetScale;
        const endPanY = innerHeight/2 - wy * targetScale;
        const start = performance.now();
        function step(now) {{
            const t = Math.min(1, (now - start) / duration);
            const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
            scale = startScale + (targetScale - startScale) * ease;
            panX = startPanX + (endPanX - startPanX) * ease;
            panY = startPanY + (endPanY - startPanY) * ease;
            world.style.transform = `translate(${{panX}}px,${{panY}}px) scale(${{scale}})`;
            zoomLabel.textContent = (scale/INITIAL_SCALE).toFixed(1) + 'x';
            if (t < 1) requestAnimationFrame(step); else resolve();
        }}
        requestAnimationFrame(step);
    }});
}}

let searchStepCount = 0;
let searchTraversal = [];
let animQueue = [];
let animating = false;
let searchPathIds = new Set(); // track all nodes on the search path

const nodePopup = document.getElementById('node-popup');
const popupTitle = document.getElementById('popup-title');
const popupLevel = document.getElementById('popup-level');
const popupDesc = document.getElementById('popup-desc');
const popupStats = document.getElementById('popup-stats');

function highlightSearchNode(nodeId, level, title, desc, reasoning) {{
    // Add current to path
    searchPathIds.add(nodeId);
    
    // Apply classes to ALL visible nodes
    document.querySelectorAll('[data-node-id]').forEach(el => {{
        el.classList.remove('search-active', 'search-path', 'search-dimmed');
        if (el.dataset.nodeId === nodeId) {{
            el.classList.add('search-active');
        }} else if (searchPathIds.has(el.dataset.nodeId)) {{
            el.classList.add('search-path');
        }} else if (el.style.display !== 'none') {{
            el.classList.add('search-dimmed');
        }}
    }});
    
    // Show popup near center of screen
    const col = LEVEL_COLORS[level] || '#666';
    popupTitle.textContent = title;
    popupLevel.textContent = level;
    popupLevel.style.background = col + '22';
    popupLevel.style.color = col;
    popupDesc.textContent = desc || '';
    popupStats.textContent = reasoning || `Step ${{searchPathIds.size}} · ${{level}}`;
    nodePopup.style.left = (innerWidth / 2 + 40) + 'px';
    nodePopup.style.top = (innerHeight / 2 - 30) + 'px';
    nodePopup.classList.add('show');
}}

function clearSearchHighlight() {{
    searchPathIds.clear();
    document.querySelectorAll('.search-active,.search-path,.search-dimmed').forEach(el => {{
        el.classList.remove('search-active', 'search-path', 'search-dimmed');
    }});
    nodePopup.classList.remove('show');
}}

async function processAnimQueue() {{
    if (animating) return;
    animating = true;
    while (animQueue.length > 0) {{
        const job = animQueue.shift();
        if (job.type === 'zoom') {{
            const zoomLevel = 0.08 + job.step * 0.3;
            searchStatus.textContent = ` ${{job.level}}: ${{job.title}}`;
            await animateTo(job.x, job.y, zoomLevel, 600);
            // Highlight this node + dim peers
            const nd = nodesData[job.node_id] || {{}};
            highlightSearchNode(job.node_id, job.level, job.title, nd.summary || '', job.reasoning || '');
        }} else if (job.type === 'nodes') {{
            job.nodes.forEach(n => placeNode(n));
            apply();
            // Dim new children that aren't on the path
            setTimeout(() => {{
                job.nodes.forEach(n => {{
                    const el = document.querySelector(`[data-node-id="${{n.id}}"]`);
                    if (el && !searchPathIds.has(n.id)) el.classList.add('search-dimmed');
                }});
            }}, 50);
        }}
    }}
    animating = false;
}}

// ── Traversal Panel + SVG Branches ──
const tpPanel = document.getElementById('traversal-panel');
const tpTree = document.getElementById('tp-tree');
const tpStatus = document.getElementById('tp-status');
const tpAnswerBox = document.getElementById('tp-answer-box');
const tpQuery = document.getElementById('tp-query');
const branchesSvg = document.getElementById('branches');
const LEVEL_COLORS = {{library:'#9e9e9e',topic:'#4285f4',subtopic:'#34a853',chapter:'#fbbc04',section:'#ea4335',concept:'#a142f4',excerpt:'#24c1e0',detail:'#f538a0'}};
// Store nodes data for branch drawing
const nodesData = {{}};

function addBranch(px, py, cx, cy) {{
    const line = document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('x1', px); line.setAttribute('y1', py);
    line.setAttribute('x2', cx); line.setAttribute('y2', cy);
    branchesSvg.appendChild(line);
}}

function tpAddStep(step) {{
    const div = document.createElement('div');
    div.className = 'tp-step';
    div.id = `tp-step-${{step.node_id}}`;
    const col = LEVEL_COLORS[step.level] || '#666';
    div.innerHTML = `<div class="tp-node" onclick="tpZoomTo('${{step.node_id}}',${{step.x}},${{step.y}},${{step.step}})">` +
        `<span class="tp-dot" style="background:${{col}}"></span>` +
        `<span>${{step.title}}</span>` +
        `<span class="tp-level">${{step.level}}</span></div>` +
        `<div class="tp-children" id="tp-kids-${{step.node_id}}"></div>`;
    tpTree.appendChild(div);
    tpPanel.scrollTop = tpPanel.scrollHeight;
    // Highlight active
    tpTree.querySelectorAll('.tp-node').forEach(n => n.classList.remove('active'));
    div.querySelector('.tp-node').classList.add('active');
}}

function tpAddChildren(parentId, nodes) {{
    const box = document.getElementById(`tp-kids-${{parentId}}`);
    if (!box) return;
    nodes.forEach(n => {{
        const chip = document.createElement('span');
        chip.className = 'tp-child';
        chip.textContent = n.title;
        chip.onclick = () => animateTo(n.x, n.y, 0.5, 500);
        box.appendChild(chip);
        // Draw branch on canvas
        nodesData[n.id] = n;
        if (nodesData[parentId]) addBranch(nodesData[parentId].x, nodesData[parentId].y, n.x, n.y);
    }});
}}

function tpZoomTo(nodeId, x, y, step) {{
    // User clicks a previous step — zoom canvas there (search keeps running)
    const zl = 0.08 + step * 0.3;
    animateTo(x, y, zl, 500);
    tpTree.querySelectorAll('.tp-node').forEach(n => n.classList.remove('active'));
    const el = document.querySelector(`#tp-step-${{nodeId}} .tp-node`);
    if (el) el.classList.add('active');
}}

function doSearch() {{
    const q = searchInput.value.trim();
    if (!q) return;
    searchBtn.disabled = true;
    searchStatus.style.display = 'block';
    searchStatus.className = 'pulsing';
    searchStatus.textContent = ' Connecting to Gemini...';
    closeAnswer();
    clearSearchHighlight();
    searchStepCount = 0;
    searchTraversal = [];
    animQueue = [];
    // Reset traversal panel
    tpTree.innerHTML = '';
    tpAnswerBox.innerHTML = '';
    tpStatus.textContent = '';
    tpQuery.textContent = ` ${{q}}`;
    tpPanel.classList.add('open');
    
    const es = new EventSource(`/api/search_stream?q=${{encodeURIComponent(q)}}`);
    
    es.addEventListener('start', e => {{
        searchStatus.textContent = ' Scanning knowledge graph...';
        tpStatus.textContent = 'Scanning...';
        tpStatus.classList.add('active');
    }});
    
    es.addEventListener('prescan', e => {{
        const d = JSON.parse(e.data);
        searchStatus.textContent = ` ${{d.msg}}`;
        if (d.nodes && d.nodes.length > 0) {{
            d.nodes.forEach(n => {{ placeNode(n); nodesData[n.id] = n; }});
            const best = d.nodes[0];
            animQueue.push({{ type: 'zoom', x: best.x, y: best.y, step: 0, level: best.level, title: best.title }});
            processAnimQueue();
        }}
    }});
    
    es.addEventListener('thinking', e => {{
        const d = JSON.parse(e.data);
        searchStatus.textContent = ` ${{d.msg}}`;
        tpStatus.textContent = d.msg;
    }});
    
    es.addEventListener('zoom', e => {{
        const step = JSON.parse(e.data);
        searchTraversal.push(step);
        nodesData[step.node_id] = step;
        tpAddStep(step);
        animQueue.push({{ type: 'zoom', ...step }});
        processAnimQueue();
    }});
    
    es.addEventListener('nodes', e => {{
        const d = JSON.parse(e.data);
        if (d.nodes) {{
            d.nodes.forEach(n => nodesData[n.id] = n);
            tpAddChildren(d.parent_id, d.nodes);
            animQueue.push({{ type: 'nodes', nodes: d.nodes }});
            processAnimQueue();
        }}
    }});
    
    es.addEventListener('answer', e => {{
        const data = JSON.parse(e.data);
        pathBar.innerHTML = searchTraversal.map(t =>
            `<span class="path-node" onclick="animateTo(${{t.x}},${{t.y}},0.5,500)">${{t.title}}</span><span>></span>`
        ).join('') + '<span style="color:#4285f4;font-weight:700">Answer</span>';
        answerText.textContent = data.answer || '';
        answerMeta.textContent = `${{data.steps}} steps · ${{data.duration_ms}}ms · ${{data.total_nodes}} nodes`;
        answerPanel.style.display = 'block';
        requestAnimationFrame(() => answerPanel.classList.add('show'));
        document.getElementById('graph-stats').textContent = `${{data.total_nodes}} nodes · gemini 3 flash preview`;
        // Traversal panel answer
        tpStatus.textContent = '';
        tpStatus.classList.remove('active');
        tpAnswerBox.innerHTML = `<div class="tp-answer"><div class="tp-answer-label"> Answer</div>${{data.answer}}</div>` +
            `<div class="tp-meta">${{data.steps}} steps · ${{data.duration_ms}}ms · ${{data.total_nodes}} nodes</div>`;
    }});
    
    es.addEventListener('done', e => {{
        es.close();
        searchBtn.disabled = false;
        tpStatus.classList.remove('active');
        setTimeout(() => {{ searchStatus.style.display = 'none'; searchStatus.className = ''; }}, 1500);
    }});
    
    es.onerror = () => {{
        es.close();
        searchBtn.disabled = false;
        tpStatus.textContent = ' Connection lost';
        tpStatus.classList.remove('active');
        searchStatus.textContent = ' Connection lost';
        setTimeout(() => searchStatus.style.display = 'none', 3000);
    }};
}}

function closeAnswer() {{
    answerPanel.classList.remove('show');
    setTimeout(() => answerPanel.style.display = 'none', 300);
}}

// Init
resetView();
</script>
</body>
</html>'''




def build_grounded_html(tree, query: str = "", highlighted_ids: set[str] | None = None) -> str:
    # render pre-built knowledge tree to html
    from gem3.models import KnowledgeNode
    
    highlighted = highlighted_ids or set()
    elements: list[str] = []
    
    def render_node(node: KnowledgeNode):
        x = node.metadata.get("x", CANVAS_WIDTH / 2)
        y = node.metadata.get("y", CANVAS_HEIGHT / 2)
        font_size = node.metadata.get("font_size", LEVEL_FONT_SIZES.get(node.level, 10))
        gray = node.metadata.get("gray", LEVEL_GRAY.get(node.level, 100))
        angle = node.metadata.get("angle", 0)
        grounded = node.metadata.get("grounded", False)
        
        # Query highlighting: nodes on relevant paths get darkened
        weight = "400"
        color = f"rgb({gray},{gray},{gray})"
        extra_class = ""
        
        if node.id in highlighted:
            darkened = max(0, gray - 80)
            color = f"rgb({darkened},{darkened},{darkened})"
            weight = "700"
            extra_class = "highlighted"
        elif query:
            title_lower = (node.title + " " + node.summary).lower()
            for w in query.lower().split():
                if len(w) > 3 and w in title_lower:
                    darkened = max(0, gray - 50)
                    color = f"rgb({darkened},{darkened},{darkened})"
                    weight = "700"
                    break
        
        # Display text
        display = node.title
        if node.level == "excerpt" and node.content:
            display = node.content[:60]
        
        # Position
        px = x - (len(display) * font_size * 0.28)
        py = y - font_size / 2
        px = max(50, min(px, CANVAS_WIDTH - 200))
        py = max(50, min(py, CANVAS_HEIGHT - 100))
        
        # Data attributes for traceability
        data_attrs = f'data-node-id="{node.id}" data-level="{node.level}"'
        if grounded:
            data_attrs += f' data-grounded="true" data-doc="{_escape(str(node.metadata.get("doc_title", "")))}"'
        
        tooltip = node.summary or ""
        if grounded:
            tooltip = f" {node.metadata.get('doc_title', '')} | {node.content[:100]}"
        
        el = (
            f'<span {data_attrs} class="{extra_class}" '
            f'title="{_escape(tooltip)}" '
            f'style="position:absolute;left:{px:.0f}px;top:{py:.0f}px;'
            f'font-size:{font_size}px;color:{color};font-weight:{weight};'
            f'transform:rotate({angle:.1f}deg);transform-origin:left center;'
            f'white-space:nowrap;line-height:1;pointer-events:auto;cursor:pointer;'
            f'">{_escape(display)}</span>'
        )
        elements.append(el)
        
        for child in node.children:
            render_node(child)
    
    render_node(tree)
    elements_html = "\n".join(elements)
    
    # Count stats
    n_total = len(elements)
    n_grounded = sum(1 for e in elements if 'data-grounded="true"' in e)
    n_highlighted = len(highlighted)
    
    query_bar = ""
    if query:
        query_bar = f'''<div id="query-bar" style="position:fixed;top:12px;left:12px;right:12px;z-index:10000;display:flex;gap:8px;align-items:center;">
<input id="query-input" type="text" value="{_escape(query)}" placeholder="Search knowledge..."
  style="flex:1;padding:6px 12px;border:1px solid #ccc;border-radius:20px;font-size:13px;background:rgba(245,245,245,0.95);outline:none;"
  onkeydown="if(event.key==='Enter')window.location='/?q='+encodeURIComponent(this.value)">
<span style="font-size:11px;color:#aaa;">{n_highlighted} paths · {n_grounded} excerpts · {n_total} nodes</span>
</div>'''
    else:
        query_bar = f'''<div id="query-bar" style="position:fixed;top:12px;left:12px;right:12px;z-index:10000;">
<input id="query-input" type="text" placeholder="Search knowledge..."
  style="width:300px;padding:6px 12px;border:1px solid #ddd;border-radius:20px;font-size:13px;background:rgba(245,245,245,0.95);outline:none;"
  onkeydown="if(event.key==='Enter')window.location='/?q='+encodeURIComponent(this.value)">
<span style="font-size:11px;color:#bbb;margin-left:8px;">{n_grounded} source excerpts · {n_total} nodes</span>
</div>'''
    
    return f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>gem3 · grounded knowledge zoom</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
html,body{{overflow:hidden;width:100vw;height:100vh;background:#f5f5f5;
font-family:'Helvetica Neue','Arial',system-ui,sans-serif;-webkit-font-smoothing:antialiased;}}
#world{{position:absolute;width:{CANVAS_WIDTH}px;height:{CANVAS_HEIGHT}px;transform-origin:0 0;will-change:transform;}}
[data-node-id]{{transition:color 0.15s;}}
[data-node-id]:hover{{color:rgb(0,0,0)!important;font-weight:700!important;}}
[data-grounded]:hover{{text-decoration:underline;}}
.highlighted{{text-shadow:0 0 2px rgba(0,0,0,0.1);}}
.controls{{position:fixed;bottom:16px;right:16px;z-index:10000;display:flex;flex-direction:column;gap:6px;}}
.ctrl-btn{{width:34px;height:34px;border:1px solid #ccc;background:rgba(245,245,245,0.95);border-radius:50%;
font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;color:#666;user-select:none;}}
.ctrl-btn:hover{{background:#eee;color:#222;}}
#zoom-level{{position:fixed;bottom:16px;left:16px;font-size:11px;color:#bbb;z-index:10000;}}
.hint{{position:fixed;bottom:40px;left:50%;transform:translateX(-50%);font-size:11px;color:#ccc;z-index:10000;
pointer-events:none;transition:opacity 2s;}}
</style>
</head>
<body>
{query_bar}
<div id="world">{elements_html}</div>
<div class="controls">
<div class="ctrl-btn" onclick="zoomBy(1.5)">+</div>
<div class="ctrl-btn" onclick="zoomBy(1/1.5)">−</div>
<div class="ctrl-btn" onclick="resetView()">⟳</div>
</div>
<div id="zoom-level"></div>
<div class="hint" id="hint">scroll to zoom · drag to pan · zoom into highlighted paths to find source excerpts</div>
<script>
const world=document.getElementById('world'),zoomLabel=document.getElementById('zoom-level'),hint=document.getElementById('hint');
const CW={CANVAS_WIDTH},CH={CANVAS_HEIGHT},IS={INITIAL_SCALE};
let scale=IS,panX=0,panY=0;
function resetView(){{scale=IS;panX=(innerWidth-CW*scale)/2;panY=(innerHeight-CH*scale)/2;apply();}}
function apply(){{world.style.transform=`translate(${{panX}}px,${{panY}}px) scale(${{scale}})`;zoomLabel.textContent=(scale/IS).toFixed(1)+'x';document.querySelectorAll('[data-node-id]').forEach(el=>{{const fs=parseFloat(el.style.fontSize)||14;const eff=fs*scale;if(eff>40){{el.style.opacity=Math.max(0.03,1-(eff-40)/120).toFixed(2);}}else{{el.style.opacity='1';}}}});}}
function zoomBy(f){{zoomAt(innerWidth/2,innerHeight/2,f);}}
function zoomAt(sx,sy,f){{const ns=Math.max(0.003,Math.min(scale*f,50));panX=sx-(sx-panX)*(ns/scale);panY=sy-(sy-panY)*(ns/scale);scale=ns;apply();}}
addEventListener('wheel',e=>{{e.preventDefault();zoomAt(e.clientX,e.clientY,e.deltaY<0?1.08:1/1.08);hint.style.opacity='0';}},{{passive:false}});
let dr=false,lx,ly;
addEventListener('mousedown',e=>{{dr=true;lx=e.clientX;ly=e.clientY;document.body.style.cursor='grabbing';}});
addEventListener('mousemove',e=>{{if(!dr)return;panX+=e.clientX-lx;panY+=e.clientY-ly;lx=e.clientX;ly=e.clientY;apply();}});
addEventListener('mouseup',()=>{{dr=false;document.body.style.cursor='default';}});
let ltd=0,ltm={{x:0,y:0}};
addEventListener('touchstart',e=>{{if(e.touches.length===2){{const dx=e.touches[1].clientX-e.touches[0].clientX,dy=e.touches[1].clientY-e.touches[0].clientY;ltd=Math.sqrt(dx*dx+dy*dy);ltm={{x:(e.touches[0].clientX+e.touches[1].clientX)/2,y:(e.touches[0].clientY+e.touches[1].clientY)/2}};}}else if(e.touches.length===1){{dr=true;lx=e.touches[0].clientX;ly=e.touches[0].clientY;}}}},{{passive:false}});
addEventListener('touchmove',e=>{{e.preventDefault();if(e.touches.length===2){{const dx=e.touches[1].clientX-e.touches[0].clientX,dy=e.touches[1].clientY-e.touches[0].clientY,dist=Math.sqrt(dx*dx+dy*dy),mid={{x:(e.touches[0].clientX+e.touches[1].clientX)/2,y:(e.touches[0].clientY+e.touches[1].clientY)/2}};if(ltd>0)zoomAt(mid.x,mid.y,dist/ltd);panX+=mid.x-ltm.x;panY+=mid.y-ltm.y;ltd=dist;ltm=mid;apply();}}else if(e.touches.length===1&&dr){{panX+=e.touches[0].clientX-lx;panY+=e.touches[0].clientY-ly;lx=e.touches[0].clientX;ly=e.touches[0].clientY;apply();}}}},{{passive:false}});
addEventListener('touchend',()=>{{dr=false;ltd=0;}});
resetView();
</script>
</body>
</html>'''


def find_relevant_paths(tree, query: str) -> set[str]:
    # find query-relevant nodes, trace paths up, return highlighted ids
    from gem3.models import KnowledgeNode
    
    highlighted = set()
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 2]
    
    def search_and_trace(node: KnowledgeNode, ancestors: list[str]) -> bool:
        # search and trace ancestors
        current_path = ancestors + [node.id]
        found = False
        
        # Check if this node matches the query
        searchable = (node.title + " " + node.summary + " " + node.content).lower()
        matches = sum(1 for w in query_words if w in searchable)
        
        # Search children first (DFS)
        for child in node.children:
            if search_and_trace(child, current_path):
                found = True
        
        # If this node or any descendant matched, highlight the path
        if matches >= 2 or (matches >= 1 and node.level in ("excerpt", "concept", "section")):
            found = True
        
        if found:
            for nid in current_path:
                highlighted.add(nid)
        
        return found
    
    search_and_trace(tree, [])
    return highlighted


async def grounded_index(request: web.Request) -> web.Response:
    # serve grounded page
    query = request.query.get('q', '')
    tree = request.app.get('grounded_tree')
    
    if tree is None:
        return web.Response(text="No knowledge tree loaded. Run ingest first.", status=500)
    
    # Find relevant paths if query provided
    highlighted = set()
    if query:
        highlighted = find_relevant_paths(tree, query)
    
    html = build_grounded_html(tree, query, highlighted)
    return web.Response(text=html, content_type='text/html')


def _prescan_graph(query: str) -> list[dict]:
    # keyword scan of existing graph, returns matches sorted by relevance
    query_words = [w.lower() for w in query.split() if len(w) > 2]
    if not query_words:
        return []
    
    scored = []
    for n in nodes_store.values():
        if n.get("level") in ("library", "root"):
            continue
        searchable = (n.get("title", "") + " " + n.get("summary", "")).lower()
        score = sum(2 if w in n.get("title", "").lower() else 1
                    for w in query_words if w in searchable)
        if score > 0:
            scored.append((score, n))
    
    scored.sort(key=lambda x: -x[0])
    return [n for _, n in scored[:8]]


async def _speculative_expand_all(nodes: list[dict], query: str) -> dict[str, list[dict]]:
    # speculatively expand all candidate nodes in parallel
    unexpanded = [n for n in nodes if not n.get("expanded")]
    if not unexpanded:
        return {}
    
    # Expand all unexpanded nodes in ONE batch Gemini call
    if len(unexpanded) == 1:
        new_nodes = await _expand_single(unexpanded[0], query)
        return {unexpanded[0]["id"]: new_nodes}
    else:
        return await _expand_batch(unexpanded, query)


async def api_search_stream(request: web.Request) -> web.StreamResponse:
    # streaming search via sse with concurrent expansion
    query = request.query.get("q", "")
    if not query:
        return web.json_response({"error": "query required"}, status=400)
    
    resp = web.StreamResponse()
    resp.content_type = "text/event-stream"
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Connection"] = "keep-alive"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    await resp.prepare(request)
    
    async def send(event_type: str, data: dict):
        await resp.write(f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode())
    
    t0 = time.time()
    search_id = hashlib.md5(f"{query}{t0}".encode()).hexdigest()[:12]
    traversal = []
    
    await send("start", {"search_id": search_id, "query": query})
    
    
    prescan_hits = _prescan_graph(query)
    if prescan_hits:
        await send("prescan", {
            "msg": f"Found {len(prescan_hits)} related nodes instantly",
            "nodes": prescan_hits,
        })
    
    
    current_nodes = [n for n in nodes_store.values() if n["level"] == "topic"]
    if not current_nodes:
        current_nodes = [n for n in nodes_store.values() if n.get("parent_id") == "root"]
    
    # Cache of speculatively pre-expanded nodes: {node_id: [children]}
    preexpanded: dict[str, list[dict]] = {}
    
    for step in range(3):
        if not current_nodes:
            break
        
        await send("thinking", {"step": step, "msg": f"Step {step+1}: analyzing {len(current_nodes)} nodes..."})
        
        
        node_options = "\n".join(
            f"  {i}. {n['title']} ({n['level']}) — {n.get('summary', '')[:60]}"
            for i, n in enumerate(current_nodes[:15])
        )
        pick_prompt = f'Query: "{query}"\n\nNodes:\n{node_options}\n\nPick 1-2 most relevant. Return JSON: {{"picks": [0], "reasoning": "why"}}'
        
        # Fire BOTH tasks simultaneously
        pick_task = asyncio.to_thread(_sync_gemini_dict, pick_prompt)
        
        # Only pre-expand if nodes aren't already expanded
        unexpanded_candidates = [n for n in current_nodes[:15] if not n.get("expanded") and n["id"] not in preexpanded]
        expand_task = _speculative_expand_all(unexpanded_candidates, query) if unexpanded_candidates else asyncio.sleep(0)
        
        # Wait for BOTH to complete (they run concurrently!)
        results = await asyncio.gather(pick_task, expand_task, return_exceptions=True)
        
        picks_data = results[0] if isinstance(results[0], dict) else {}
        spec_expanded = results[1] if isinstance(results[1], dict) else {}
        preexpanded.update(spec_expanded)
        
        raw_picks = picks_data.get("picks", [0])
        pick_indices = [int(p) for p in raw_picks if str(p).isdigit()][:2] or [0]
        
        
        next_level_nodes = []
        for idx in pick_indices:
            if idx >= len(current_nodes):
                continue
            picked = current_nodes[idx]
            
            step_info = {
                "node_id": picked["id"], "title": picked["title"],
                "level": picked["level"], "x": picked["x"], "y": picked["y"],
                "step": step, "reasoning": picks_data.get("reasoning", ""),
            }
            traversal.append(step_info)
            await send("zoom", step_info)
            
            # Children are already ready from speculative expand!
            if picked["id"] in preexpanded:
                children = preexpanded[picked["id"]]
                await send("nodes", {"parent_id": picked["id"], "nodes": children})
                next_level_nodes.extend(children)
            elif picked.get("expanded"):
                existing = [nodes_store[cid] for cid in picked.get("children", []) if cid in nodes_store]
                await send("nodes", {"parent_id": picked["id"], "nodes": existing})
                next_level_nodes.extend(existing)
            else:
                # Fallback: expand now (shouldn't happen often with speculative)
                child_level = _next_level(picked["level"])
                children_data = await _call_gemini(
                    f'List 5 {child_level}s under "{picked["title"]}" for "{query}". JSON: [{{"title":"...","summary":"..."}}]'
                )
                new_nodes = _place_children(picked, children_data, child_level)
                await send("nodes", {"parent_id": picked["id"], "nodes": new_nodes})
                next_level_nodes.extend(new_nodes)
        
        current_nodes = next_level_nodes
    
    
    await send("thinking", {"step": -1, "msg": "Synthesizing answer..."})
    
    path_summary = " > ".join(t["title"] for t in traversal)
    leaf_context = "\n".join(f"- {n['title']}: {n.get('summary', '')}" for n in current_nodes[:10])
    
    answer = await asyncio.to_thread(_sync_gemini_text,
        f"Knowledge path: {path_summary}\nLeaf nodes:\n{leaf_context}\n\nAnswer concisely: \"{query}\"\nReference the path. Markdown.")
    
    elapsed = int((time.time() - t0) * 1000)
    
    await send("answer", {
        "answer": answer, "traversal": traversal,
        "steps": len(traversal), "duration_ms": elapsed,
        "total_nodes": len(nodes_store),
    })
    
    # BQ save in background
    try:
        asyncio.get_event_loop().run_in_executor(
            None, save_search, search_id, query, answer,
            [t["node_id"] for t in traversal], leaf_context,
            len(traversal), elapsed)
    except Exception:
        pass
    
    await send("done", {"search_id": search_id})
    await resp.write_eof()
    return resp


def _sync_gemini_text(prompt: str) -> str:
    # sync gemini text call
    try:
        c = get_gemini_client()
        response = c.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.5),
        )
        return response.text.strip()
    except Exception as e:
        print(f"gemini error: {e}")
    return "Could not generate answer."


async def api_stats(request: web.Request) -> web.Response:
    # graph stats endpoint
    stats = await asyncio.to_thread(get_graph_stats)
    stats["in_memory_nodes"] = len(nodes_store)
    return web.json_response(stats)


async def index(request: web.Request) -> web.Response:
    # serve canvas page
    # If a grounded tree is loaded, use grounded mode
    if request.app.get('grounded_tree'):
        return await grounded_index(request)
    
    query = request.query.get('q', '')
    seed_topics = request.app.get('seed_topics', None)
    
    # Don't re-seed if nodes already loaded from BQ
    if len(nodes_store) > 1:
        # Already have nodes — just rebuild HTML from current state
        all_nodes = list(nodes_store.values())
    else:
        all_nodes = create_seed_nodes(seed_topics, query)
        # Save seed nodes to BQ
        try:
            asyncio.get_event_loop().run_in_executor(None, save_nodes, all_nodes, "seed")
        except Exception:
            pass
    
    html = build_canvas_html(all_nodes, query)
    return web.Response(text=html, content_type='text/html')


async def start_server(
    host: str = "0.0.0.0",
    port: int = 3333,
    topics: list[str] | None = None,
    query: str = "",
    tree=None,
):
    # start server
    global nodes_store
    
    
    print("loading graph from bq...")
    try:
        bq_nodes = load_all_nodes()
        if bq_nodes:
            nodes_store.update(bq_nodes)
            print(f"loaded {len(bq_nodes)} nodes from bq")
        else:
            print("empty graph, seeding on first request")
    except Exception as e:
        print(f"bq load failed: {e}")
    
    app = web.Application()
    app['seed_topics'] = topics
    app['grounded_tree'] = tree
    
    app.router.add_get('/', index)
    app.router.add_post('/api/expand', expand_node)
    app.router.add_post('/api/expand_batch', expand_batch)
    app.router.add_get('/api/search_stream', api_search_stream)
    app.router.add_get('/api/stats', api_stats)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    
    print(f"serving on http://localhost:{port}")
    print(f"{len(nodes_store)} nodes loaded")
    if query:
        print(f"query: {query}")
    
    await site.start()
    
    
    async def _pregen():
        # pre-expand topics
        await asyncio.sleep(2)  # Let server finish starting
        topics = [n for n in nodes_store.values() if n.get("level") == "topic" and not n.get("expanded")]
        if topics:
            print(f"pre-expanding {len(topics)} topics...")
            try:
                await _expand_batch(topics, "")
                print(f"pre-expanded {len(topics)} topics")
            except Exception as e:
                print(f"pre-gen failed: {e}")
    
    asyncio.create_task(_pregen())
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    import asyncio
    import sys
    
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    asyncio.run(start_server(query=query))
