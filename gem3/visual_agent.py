"""
gem3 Visual Agent — Gemini 3 Flash navigates the zoom canvas visually.

The agent SEES a rendered image of the knowledge canvas at each zoom level,
decides where to zoom based on the query, and builds an explainability
subgraph of its traversal decisions.

Flow:
  1. Render canvas at current viewport → PIL image
  2. Send image to Gemini with query + zoom history
  3. Gemini returns: zoom region (x%, y%), reasoning, found_answer flag
  4. Zoom to that region, re-render, repeat
  5. When excerpts are readable, Gemini extracts the answer
  6. Full traversal recorded as explainability subgraph
"""

from __future__ import annotations
import io
import json
import math
import time
import base64
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types

from gem3.config import get_gemini_client, GEMINI_MODEL, SCREENSHOT_DIR
from gem3.models import KnowledgeNode



CANVAS_W = 16000
CANVAS_H = 10000
RENDER_W = 1200  # Output image width
RENDER_H = 800   # Output image height


def get_client() -> genai.Client:
    return get_gemini_client()




def render_viewport(
    tree: KnowledgeNode,
    viewport_x: float,
    viewport_y: float,
    viewport_w: float,
    viewport_h: float,
    query: str = "",
    highlighted_ids: set[str] | None = None,
) -> Image.Image:
    """
    Render a portion of the knowledge canvas as a PIL image.
    
    viewport_x/y/w/h define the visible region in canvas coordinates.
    Only nodes visible in this region are drawn.
    """
    highlighted = highlighted_ids or set()
    img = Image.new('RGB', (RENDER_W, RENDER_H), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    
    # Scale factor: canvas coords → pixel coords
    sx = RENDER_W / viewport_w
    sy = RENDER_H / viewport_h
    
    # Collect all visible nodes
    visible_nodes = []
    _collect_visible(tree, viewport_x, viewport_y, viewport_w, viewport_h, visible_nodes)
    
    # Sort by font size descending (draw big/faint first, small/dark on top)
    visible_nodes.sort(key=lambda n: n.metadata.get("font_size", 10), reverse=True)
    
    for node in visible_nodes:
        nx = node.metadata.get("x", CANVAS_W / 2)
        ny = node.metadata.get("y", CANVAS_H / 2)
        font_size = node.metadata.get("font_size", 10)
        gray = node.metadata.get("gray", 100)
        
        # Transform to pixel coordinates
        px = (nx - viewport_x) * sx
        py = (ny - viewport_y) * sy
        
        # Effective font size in pixels
        eff_size = max(1, int(font_size * sx))
        if eff_size < 4:
            continue  # Too small to render
        
        # CRITICAL: Fade out high-level nodes when zoomed deep
        # Large parent text (huge eff_size) should become nearly invisible
        # so excerpt text (small eff_size) is clear and readable
        zoom_fade = 1.0
        if eff_size > 30:
            # Very aggressive fade: 30px=full, 70px=half, 110px+=nearly invisible
            zoom_fade = max(0.02, 1.0 - (eff_size - 30) / 80)
        
        eff_size = min(eff_size, 200)  # Cap for rendering
        
        # Query/highlight coloring
        is_highlighted = node.id in highlighted
        is_grounded = node.metadata.get("grounded", False)
        
        # Apply zoom fade to gray (higher = more faded/transparent)
        faded_gray = int(gray + (245 - gray) * (1 - zoom_fade))
        
        # Only darken (highlight) nodes at natural readable size, NOT oversized parent text
        if zoom_fade > 0.8:
            if is_highlighted:
                faded_gray = max(0, faded_gray - int(80 * zoom_fade))
            elif query:
                text_lower = (node.title + " " + node.summary).lower()
                for w in query.lower().split():
                    if len(w) > 3 and w in text_lower:
                        faded_gray = max(0, faded_gray - int(40 * zoom_fade))
                        break
        
        color = (faded_gray, faded_gray, faded_gray)
        if is_grounded and is_highlighted:
            color = (max(0, faded_gray - 20), faded_gray, max(0, faded_gray - 20))
        
        # Display text
        display = node.title
        if node.level == "excerpt" and node.content:
            # Show FULL excerpt text when zoomed in enough
            if eff_size >= 8:
                display = node.content
            else:
                display = node.content[:60]
        
        # Try to use a font at the right size
        render_size = max(6, min(eff_size, 120))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", render_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/System/Library/Fonts/SFNSText.ttf", render_size)
            except (OSError, IOError):
                font = ImageFont.load_default()
        
        # Draw text (handle rotation approximately with horizontal text)
        # PIL rotation is complex, so we draw horizontal with slight offset
        angle = node.metadata.get("angle", 0)
        
        # Sanitize text to avoid font rendering issues
        display = display.encode('ascii', 'replace').decode('ascii')
        
        # For long excerpt text, wrap it
        try:
            if len(display) > 80 and eff_size >= 8:
                lines = _wrap_text(display, 60)
                for li, line in enumerate(lines[:8]):
                    draw.text((px, py + li * (eff_size + 2)), line, fill=color, font=font)
            else:
                draw.text((px, py), display[:100], fill=color, font=font)
        except (OSError, Exception):
            pass  # Skip problematic text
        
        # Draw grounded indicator
        if is_grounded and eff_size >= 10:
            doc_title = str(node.metadata.get("doc_title", "")).encode('ascii', 'replace').decode('ascii')
            try:
                small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(8, eff_size // 3))
                draw.text((px, py - max(8, eff_size // 3) - 2), f"[{doc_title}]", fill=(150, 150, 150), font=small_font)
            except (OSError, IOError, Exception):
                pass
    
    # Draw viewport info
    try:
        info_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        info_font = ImageFont.load_default()
    
    zoom_level = CANVAS_W / viewport_w
    draw.text((10, RENDER_H - 25), f"zoom: {zoom_level:.1f}x | visible: {len(visible_nodes)} nodes", fill=(180, 180, 180), font=info_font)
    
    # Draw grid overlay (10 regions for agent reference)
    _draw_grid(draw, 10, 10)
    
    return img


def _collect_visible(
    node: KnowledgeNode,
    vx: float, vy: float, vw: float, vh: float,
    result: list,
):
    """Collect nodes visible in the viewport region."""
    nx = node.metadata.get("x", CANVAS_W / 2)
    ny = node.metadata.get("y", CANVAS_H / 2)
    font_size = node.metadata.get("font_size", 10)
    
    # Check if node is in viewport (with generous margin based on font size)
    margin = font_size * 5  # Text can extend beyond its position
    if (nx + margin >= vx and nx - margin <= vx + vw and
        ny + margin >= vy and ny - margin <= vy + vh):
        result.append(node)
    
    # Always recurse into children (they might be visible even if parent isn't)
    for child in node.children:
        _collect_visible(child, vx, vy, vw, vh, result)


def _draw_grid(draw: ImageDraw.Draw, cols: int, rows: int):
    """Draw a faint grid for agent reference coordinates."""
    for i in range(1, cols):
        x = int(RENDER_W * i / cols)
        draw.line([(x, 0), (x, RENDER_H)], fill=(230, 230, 230), width=1)
    for j in range(1, rows):
        y = int(RENDER_H * j / rows)
        draw.line([(0, y), (RENDER_W, y)], fill=(230, 230, 230), width=1)
    
    # Label grid intersections
    try:
        grid_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
    except (OSError, IOError):
        grid_font = ImageFont.load_default()
    
    for i in range(cols + 1):
        for j in range(rows + 1):
            if i % 2 == 0 and j % 2 == 0:
                x = int(RENDER_W * i / cols)
                y = int(RENDER_H * j / rows)
                draw.text((x + 2, y + 1), f"{i},{j}", fill=(210, 210, 210), font=grid_font)


def _wrap_text(text: str, width: int) -> list[str]:
    """Wrap text to specified character width."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines




class ZoomStep:
    """A single step in the visual navigation."""
    def __init__(self, step_num: int, viewport: tuple, zoom_level: float,
                 reasoning: str, target_region: str, visible_nodes: int):
        self.step_num = step_num
        self.viewport = viewport  # (x, y, w, h)
        self.zoom_level = zoom_level
        self.reasoning = reasoning
        self.target_region = target_region
        self.visible_nodes = visible_nodes
        self.image_path: str = ""
        self.found_excerpts: list[str] = []


class VisualSearchResult:
    """Complete result of a visual agent search."""
    def __init__(self, query: str):
        self.query = query
        self.steps: list[ZoomStep] = []
        self.answer: str = ""
        self.source_excerpts: list[dict] = []
        self.total_time: float = 0


def visual_search(
    query: str,
    tree: KnowledgeNode,
    max_steps: int = 6,
    verbose: bool = True,
) -> VisualSearchResult:
    """
    Gemini 3 Flash visually navigates the knowledge canvas to answer a query.
    
    1. Renders the full canvas overview
    2. Gemini sees the image and decides where to zoom
    3. Zooms in, re-renders, repeats
    4. When excerpts are readable, extracts the answer
    5. Returns full traversal with explainability
    """
    client = get_client()
    result = VisualSearchResult(query)
    start_time = time.time()
    
    # Find query-relevant paths for highlighting
    from gem3.ingest import count_nodes
    from server import find_relevant_paths
    highlighted = find_relevant_paths(tree, query)
    
    # Start at full overview
    vx, vy, vw, vh = 0, 0, CANVAS_W, CANVAS_H
    
    history = []  # Zoom decision history for context
    
    for step_num in range(max_steps):
        zoom_level = CANVAS_W / vw
        
        if verbose:
            print(f"\n   Step {step_num + 1}/{max_steps} — zoom {zoom_level:.1f}x")
        
        # Render current viewport
        img = render_viewport(tree, vx, vy, vw, vh, query, highlighted)
        
        # Save image
        img_path = SCREENSHOT_DIR / f"search_step_{step_num}.png"
        img.save(img_path)
        
        # Count visible nodes
        visible = []
        _collect_visible(tree, vx, vy, vw, vh, visible)
        n_visible = len(visible)
        n_excerpts = sum(1 for n in visible if n.level == "excerpt")
        
        if verbose:
            print(f"     visible: {n_visible} nodes, {n_excerpts} excerpts")
        
        # Convert image to bytes for Gemini
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # Build history context
        history_text = ""
        if history:
            history_text = "Previous zoom decisions:\n"
            for h in history:
                history_text += f"  Step {h['step']}: zoomed to {h['target']} because: {h['reasoning']}\n"
        
        # Check if we can read excerpts at this zoom level
        readable_excerpts = []
        for node in visible:
            if node.level == "excerpt" and node.content:
                eff_size = node.metadata.get("font_size", 10) * (RENDER_W / vw)
                if eff_size >= 6:
                    readable_excerpts.append({
                        "title": node.title,
                        "content": node.content,
                        "doc_title": node.metadata.get("doc_title", "unknown"),
                        "doc_index": node.metadata.get("doc_index", -1),
                    })
        
        # Build Gemini prompt
        if readable_excerpts and step_num >= 2:
            # EXTRACTION MODE — we can read excerpts, synthesize answer
            excerpts_text = "\n\n".join([
                f" {e['doc_title']}:\n{e['content']}" for e in readable_excerpts
            ])
            
            prompt = f"""You are a visual knowledge navigator. You've zoomed into a knowledge canvas
and can now read the source excerpts.

Query: "{query}"

{history_text}

Source excerpts visible at this zoom level:
{excerpts_text}

You are looking at a rendered image of the knowledge canvas at {zoom_level:.1f}x zoom.
The image shows a grid overlay (10x10) for reference.

TASKS:
1. Identify which visible excerpts are most relevant to the query
2. Synthesize an answer from the relevant excerpts
3. Cite the source documents

Return JSON:
{{
  "answer": "your synthesized answer based on the real excerpts",
  "relevant_excerpts": ["excerpt content 1", "excerpt content 2"],
  "source_docs": ["doc title 1", "doc title 2"],
  "reasoning": "why these excerpts answer the query",
  "found_answer": true,
  "confidence": 0.0-1.0
}}"""
        else:
            # NAVIGATION MODE — decide where to zoom next
            prompt = f"""You are a visual knowledge navigator exploring an infinite zoom knowledge canvas.
The canvas shows a hierarchy: large faint words are broad topics, smaller darker words are specific content.
Bold/darker words are highlighted because they match the query.
A 10x10 grid overlay helps you specify coordinates.

Query: "{query}"

Current zoom: {zoom_level:.1f}x
Visible nodes: {n_visible}
Visible excerpts: {n_excerpts}

{history_text}

Look at this image of the knowledge canvas. Identify the region that is most relevant
to the query. Consider:
- Bold/dark text indicates query-relevant paths
- Smaller text means more specific/detailed content
- You want to zoom toward areas with relevant smaller text
- Clusters of small text likely contain source excerpts

Return JSON:
{{
  "target_x_pct": 0.0-1.0 (horizontal position to center zoom on),
  "target_y_pct": 0.0-1.0 (vertical position to center zoom on),
  "zoom_factor": 2.0-4.0 (how much to zoom in),
  "target_description": "what you see at the target region",
  "reasoning": "why this region is relevant to the query",
  "found_answer": false
}}"""
        
        # Send to Gemini with image
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(temperature=0.3),
            )
            
            response_text = response.text.strip()
            # Parse JSON from response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                decision = json.loads(response_text[start:end])
            else:
                decision = {"target_x_pct": 0.5, "target_y_pct": 0.5, "zoom_factor": 2.0,
                           "reasoning": "Could not parse response", "found_answer": False}
        
        except Exception as e:
            print(f"      Gemini error: {e}")
            decision = {"target_x_pct": 0.5, "target_y_pct": 0.5, "zoom_factor": 2.0,
                       "reasoning": f"Error: {e}", "found_answer": False}
        
        # Record step
        reasoning = decision.get("reasoning", "")
        target_desc = decision.get("target_description", decision.get("target_region", ""))
        
        step = ZoomStep(
            step_num=step_num,
            viewport=(vx, vy, vw, vh),
            zoom_level=zoom_level,
            reasoning=reasoning,
            target_region=target_desc,
            visible_nodes=n_visible,
        )
        step.image_path = str(img_path)
        result.steps.append(step)
        
        if verbose:
            print(f"      {reasoning[:120]}")
        
        # Check if answer found
        if decision.get("found_answer"):
            result.answer = decision.get("answer", "")
            result.source_excerpts = [
                {"content": e, "doc": d}
                for e, d in zip(
                    decision.get("relevant_excerpts", []),
                    decision.get("source_docs", [])
                )
            ]
            if verbose:
                print(f"      Answer found! Confidence: {decision.get('confidence', '?')}")
            break
        
        # Zoom to target region (clamp & normalize if agent used grid coords 0-10)
        tx = decision.get("target_x_pct", 0.5)
        ty = decision.get("target_y_pct", 0.5)
        if tx > 1.0:
            tx = tx / 10.0  # Agent used grid coords 0-10
        if ty > 1.0:
            ty = ty / 10.0
        tx = max(0.05, min(tx, 0.95))
        ty = max(0.05, min(ty, 0.95))
        zf = decision.get("zoom_factor", 2.5)
        zf = max(1.5, min(zf, 3.0))  # Cap at 3x to avoid overshoot
        
        # Calculate new viewport centered on target
        target_cx = vx + vw * tx
        target_cy = vy + vh * ty
        new_vw = vw / zf
        new_vh = vh / zf
        vx = target_cx - new_vw / 2
        vy = target_cy - new_vh / 2
        
        # Clamp viewport
        vx = max(0, min(vx, CANVAS_W - new_vw))
        vy = max(0, min(vy, CANVAS_H - new_vh))
        vw = new_vw
        vh = new_vh
        
        history.append({
            "step": step_num + 1,
            "target": target_desc,
            "reasoning": reasoning[:100],
            "zoom": zoom_level,
        })
        
        if verbose:
            print(f"     → Zooming to ({tx:.0%}, {ty:.0%}) @ {zf:.1f}x")
    
    # If no answer found after all steps, set a "not found" message
    if not result.answer:
        result.answer = ""
        # Check if we found any related content
        all_visible = []
        _collect_visible(tree, vx, vy, vw, vh, all_visible)
        nearby_topics = [n.title for n in all_visible if n.level in ("topic", "subtopic", "chapter")][:5]
        if nearby_topics:
            result.answer = (
                f"No exact match found in the knowledge base for this query. "
                f"The agent explored related areas including: {', '.join(nearby_topics)}. "
                f"Consider adding documents that cover this topic."
            )
    
    result.total_time = time.time() - start_time
    
    # Generate explainability subgraph
    _save_explainability(result)
    
    return result


def _save_explainability(result: VisualSearchResult):
    """Save the traversal as an explainability graph."""
    graph = {
        "query": result.query,
        "total_time_seconds": result.total_time,
        "total_steps": len(result.steps),
        "answer": result.answer,
        "source_excerpts": result.source_excerpts,
        "traversal": [],
    }
    
    for step in result.steps:
        graph["traversal"].append({
            "step": step.step_num + 1,
            "zoom_level": f"{step.zoom_level:.1f}x",
            "viewport": {
                "x": step.viewport[0], "y": step.viewport[1],
                "w": step.viewport[2], "h": step.viewport[3],
            },
            "visible_nodes": step.visible_nodes,
            "reasoning": step.reasoning,
            "target_region": step.target_region,
            "image": step.image_path,
            "found_excerpts": step.found_excerpts,
        })
    
    graph_path = SCREENSHOT_DIR / "explainability_graph.json"
    with open(graph_path, "w") as f:
        json.dump(graph, f, indent=2)


def print_explainability(result: VisualSearchResult):
    """Pretty-print the explainability subgraph."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree
    
    console = Console()
    
    tree = Tree(f" [bold]Query:[/] {result.query}")
    
    for step in result.steps:
        zoom_str = f"{step.zoom_level:.1f}x"
        node = tree.add(
            f"[cyan]Step {step.step_num + 1}[/] — zoom {zoom_str} — "
            f"{step.visible_nodes} nodes visible"
        )
        node.add(f"[dim] {step.reasoning[:150]}[/]")
        if step.target_region:
            node.add(f"[dim] {step.target_region[:100]}[/]")
    
    console.print(tree)
    
    if result.answer:
        console.print(Panel.fit(
            result.answer,
            title=" Answer",
            border_style="green",
        ))
        
        if result.source_excerpts:
            console.print("\n[bold]Source Excerpts:[/]")
            for exc in result.source_excerpts:
                console.print(f"   [dim]{exc.get('doc', '')}[/]")
                console.print(f"     {exc.get('content', '')[:200]}")
    
    console.print(f"\n[dim]Total time: {result.total_time:.1f}s | Steps: {len(result.steps)}[/]")
    console.print(f"[dim]Screenshots: {SCREENSHOT_DIR}[/]")
