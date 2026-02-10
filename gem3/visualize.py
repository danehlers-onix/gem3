"""
Visual HTML Generation — Single-Page Infinite Zoom Canvas.

ALL levels of the knowledge hierarchy exist on ONE massive canvas.
You start fully zoomed out — only topic-level words are readable.
Everything deeper is literally sub-pixel. As you zoom in:
  topics become enormous ghosts → subtopics emerge → chapters appear →
  sections materialize → concepts crystallize → excerpts become readable

The zoom is continuous and smooth — scroll wheel zooms, drag to pan.
Size, lightness, rotation, and position encode the knowledge structure.
"""

from __future__ import annotations
import asyncio
import math
import hashlib
from pathlib import Path

from gem3.config import (
    GEMINI_MODEL, HTML_DIR, SCREENSHOT_DIR,
    VIEWPORT_WIDTH, VIEWPORT_HEIGHT,
)
from gem3.models import KnowledgeNode, ZoomState, ZOOM_LEVELS



# The canvas is massive. At the initial zoom (~0.06x), the whole thing
# fits in the viewport. Only the biggest words (400-800px) are readable.
# Everything else is sub-pixel noise until you zoom closer.

CANVAS_WIDTH = 16000
CANVAS_HEIGHT = 10000

# Font sizes at CANVAS scale (not screen scale!)
# At initial zoom (0.06x), a 400px word renders as ~24px on screen (readable).
# A 60px word renders as ~3.6px (invisible). That's the point.
LEVEL_FONT_SIZES = {
    "library": 800,
    "topic": 400,
    "subtopic": 150,
    "chapter": 60,
    "section": 30,
    "concept": 16,
    "excerpt": 10,
}

# Grayscale: lighter = higher level (fades as you zoom past it)
LEVEL_GRAY = {
    "library": 232,
    "topic": 200,
    "subtopic": 175,
    "chapter": 145,
    "section": 110,
    "concept": 70,
    "excerpt": 30,
}

# How far children spread from their parent center
LEVEL_SPREAD = {
    "library": 5000,
    "topic": 2000,
    "subtopic": 800,
    "chapter": 350,
    "section": 150,
    "concept": 70,
    "excerpt": 30,
}

# Initial zoom: fit canvas width into ~900px viewport
INITIAL_SCALE = 900 / CANVAS_WIDTH  # ≈ 0.056


def _hash_angle(text: str, seed: int = 0) -> float:
    """Deterministic pseudo-random angle from text, range [-30, 30] degrees."""
    h = int(hashlib.md5(f"{text}{seed}".encode()).hexdigest()[:8], 16)
    return ((h % 61) - 30)


def _hash_offset(text: str, spread: float, seed: int = 0) -> tuple[float, float]:
    """Deterministic pseudo-random offset from parent center."""
    h = hashlib.md5(f"{text}{seed}".encode()).hexdigest()
    hx = int(h[:8], 16)
    hy = int(h[8:16], 16)
    angle_rad = (hx % 360) * math.pi / 180
    radius = (hy % 1000) / 1000.0 * spread
    return (math.cos(angle_rad) * radius, math.sin(angle_rad) * radius)


def render_full_canvas(root: KnowledgeNode, query: str = "") -> str:
    """
    Render the ENTIRE knowledge hierarchy as one zoomable HTML page.
    """
    elements: list[str] = []

    def place_node(
        node: KnowledgeNode,
        cx: float,
        cy: float,
        depth: int = 0,
        parent_angle: float = 0,
    ):
        level = node.level
        font_size = LEVEL_FONT_SIZES.get(level, 14)
        gray = LEVEL_GRAY.get(level, 100)
        spread = LEVEL_SPREAD.get(level, 50)

        # Rotation
        angle = _hash_angle(node.title, depth)
        angle = angle * 0.7 + parent_angle * 0.3

        # Query relevance darkening
        color = f"rgb({gray},{gray},{gray})"
        weight = "400"
        if query:
            title_lower = (node.title + " " + node.summary).lower()
            query_words = [w for w in query.lower().split() if len(w) > 3]
            matches = sum(1 for w in query_words if w in title_lower)
            if matches > 0:
                darkened = max(0, gray - matches * 50)
                color = f"rgb({darkened},{darkened},{darkened})"
                weight = "700"

        display_text = node.title
        if level == "excerpt" and node.content:
            display_text = node.content[:80]

        # Position (rough centering on cx, cy)
        x = cx - (len(display_text) * font_size * 0.28)
        y = cy - font_size / 2

        # Clamp to canvas bounds
        x = max(50, min(x, CANVAS_WIDTH - 200))
        y = max(50, min(y, CANVAS_HEIGHT - 100))

        el = (
            f'<span data-node-id="{node.id}" data-level="{level}" '
            f'style="'
            f'position:absolute;'
            f'left:{x:.0f}px;top:{y:.0f}px;'
            f'font-size:{font_size}px;'
            f'color:{color};'
            f'font-weight:{weight};'
            f'transform:rotate({angle:.1f}deg);'
            f'transform-origin:left center;'
            f'white-space:nowrap;'
            f'line-height:1;'
            f'pointer-events:auto;'
            f'cursor:pointer;'
            f'">{_escape(display_text)}</span>'
        )
        elements.append(el)

        # Place children around this node
        if node.children:
            n = len(node.children)
            for i, child in enumerate(node.children):
                child_angle_rad = (2 * math.pi * i / n) + _hash_angle(node.title, i) * 0.02
                dx, dy = _hash_offset(child.title, spread, i)
                dx += math.cos(child_angle_rad) * spread * 0.5
                dy += math.sin(child_angle_rad) * spread * 0.5
                place_node(child, cx + dx, cy + dy, depth + 1, angle)

    # Root at canvas center
    place_node(root, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)

    elements_html = "\n".join(elements)

    query_indicator = ""
    if query:
        query_indicator = f'<div id="query-label" style="position:fixed;top:12px;left:12px;font-size:12px;color:#aaa;z-index:10000;background:rgba(245,245,245,0.9);padding:4px 10px;border-radius:4px;"> {_escape(query)}</div>'

    return f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gem3 · knowledge zoom</title>
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
    transition: color 0.15s;
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
    width: 34px;
    height: 34px;
    border: 1px solid #ccc;
    background: rgba(245,245,245,0.95);
    border-radius: 50%;
    font-size: 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #666;
    user-select: none;
}}
.ctrl-btn:hover {{ background: #eee; color: #222; }}
#zoom-level {{
    position: fixed;
    bottom: 16px;
    left: 16px;
    font-size: 11px;
    color: #bbb;
    z-index: 10000;
}}
.hint {{
    position: fixed;
    bottom: 40px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 11px;
    color: #ccc;
    z-index: 10000;
    pointer-events: none;
    transition: opacity 2s;
}}
</style>
</head>
<body>
{query_indicator}
<div id="world">
{elements_html}
</div>

<div class="controls">
    <div class="ctrl-btn" onclick="zoomBy(1.5)">+</div>
    <div class="ctrl-btn" onclick="zoomBy(1/1.5)">−</div>
    <div class="ctrl-btn" onclick="resetView()">⟳</div>
</div>
<div id="zoom-level"></div>
<div class="hint" id="hint">scroll to zoom · drag to pan</div>

<script>
const world = document.getElementById('world');
const zoomLabel = document.getElementById('zoom-level');
const hint = document.getElementById('hint');
const CW = {CANVAS_WIDTH};
const CH = {CANVAS_HEIGHT};

// State
let scale = {INITIAL_SCALE};
let panX = 0;
let panY = 0;

// Center the view initially
function resetView() {{
    scale = {INITIAL_SCALE};
    panX = (window.innerWidth - CW * scale) / 2;
    panY = (window.innerHeight - CH * scale) / 2;
    apply();
}}

function apply() {{
    world.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
    // Show zoom level
    const zoomPct = (scale / {INITIAL_SCALE}).toFixed(1);
    zoomLabel.textContent = zoomPct + 'x';
}}

function zoomBy(factor) {{
    const cx = window.innerWidth / 2;
    const cy = window.innerHeight / 2;
    zoomAt(cx, cy, factor);
}}

function zoomAt(screenX, screenY, factor) {{
    const newScale = Math.max(0.005, Math.min(scale * factor, 30));
    // Zoom toward the mouse position
    panX = screenX - (screenX - panX) * (newScale / scale);
    panY = screenY - (screenY - panY) * (newScale / scale);
    scale = newScale;
    apply();
}}

// ── Mouse wheel zoom ──
window.addEventListener('wheel', function(e) {{
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.08 : 1 / 1.08;
    zoomAt(e.clientX, e.clientY, factor);
    // Hide hint after first interaction
    hint.style.opacity = '0';
}}, {{ passive: false }});

// ── Drag to pan ──
let dragging = false;
let lastX, lastY;

window.addEventListener('mousedown', function(e) {{
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    document.body.style.cursor = 'grabbing';
}});

window.addEventListener('mousemove', function(e) {{
    if (!dragging) return;
    panX += e.clientX - lastX;
    panY += e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    apply();
}});

window.addEventListener('mouseup', function() {{
    dragging = false;
    document.body.style.cursor = 'default';
}});

// ── Touch support ──
let lastTouchDist = 0;
let lastTouchMid = {{x:0, y:0}};

window.addEventListener('touchstart', function(e) {{
    if (e.touches.length === 2) {{
        const dx = e.touches[1].clientX - e.touches[0].clientX;
        const dy = e.touches[1].clientY - e.touches[0].clientY;
        lastTouchDist = Math.sqrt(dx*dx + dy*dy);
        lastTouchMid = {{
            x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
            y: (e.touches[0].clientY + e.touches[1].clientY) / 2,
        }};
    }} else if (e.touches.length === 1) {{
        dragging = true;
        lastX = e.touches[0].clientX;
        lastY = e.touches[0].clientY;
    }}
}}, {{ passive: false }});

window.addEventListener('touchmove', function(e) {{
    e.preventDefault();
    if (e.touches.length === 2) {{
        const dx = e.touches[1].clientX - e.touches[0].clientX;
        const dy = e.touches[1].clientY - e.touches[0].clientY;
        const dist = Math.sqrt(dx*dx + dy*dy);
        const mid = {{
            x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
            y: (e.touches[0].clientY + e.touches[1].clientY) / 2,
        }};
        if (lastTouchDist > 0) {{
            zoomAt(mid.x, mid.y, dist / lastTouchDist);
        }}
        panX += mid.x - lastTouchMid.x;
        panY += mid.y - lastTouchMid.y;
        lastTouchDist = dist;
        lastTouchMid = mid;
        apply();
    }} else if (e.touches.length === 1 && dragging) {{
        panX += e.touches[0].clientX - lastX;
        panY += e.touches[0].clientY - lastY;
        lastX = e.touches[0].clientX;
        lastY = e.touches[0].clientY;
        apply();
    }}
}}, {{ passive: false }});

window.addEventListener('touchend', function() {{
    dragging = false;
    lastTouchDist = 0;
}});

// Init
resetView();
</script>
</body>
</html>'''


def generate_zoom_html(
    node: KnowledgeNode,
    zoom_state: ZoomState,
    query: str = "",
    highlight_nodes: list[str] | None = None,
) -> str:
    """Generate the single-page canvas."""
    html = render_full_canvas(node, query)
    node.html_cache = html
    html_path = HTML_DIR / f"zoom_{node.id}_{zoom_state.zoom_level}.html"
    html_path.write_text(html, encoding="utf-8")
    return html


def _escape(text: str) -> str:
    """Basic HTML escape."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


async def capture_screenshot(html_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Capture a screenshot of the canvas at a specific zoom region."""
    from playwright.async_api import async_playwright

    html_path = Path(html_path)
    if output_path is None:
        output_path = SCREENSHOT_DIR / f"{html_path.stem}.png"
    output_path = Path(output_path)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT}
        )
        await page.goto(f"file://{html_path.resolve()}")
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(output_path), full_page=False)
        await browser.close()

    return output_path


def capture_screenshot_sync(html_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Synchronous wrapper for screenshot capture."""
    return asyncio.run(capture_screenshot(html_path, output_path))
