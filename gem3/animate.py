"""
gem3 Zoom Animation — Generate cinematic MP4 of visual search traversal.

Creates a smooth animated video showing:
  - The knowledge canvas at each zoom level
  - Highlighted selection box where the agent will zoom
  - Decision reasoning overlay text
  - Smooth zoom interpolation between steps
  - Final answer panel with source citations
"""

from __future__ import annotations
import math
import subprocess
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from gem3.config import SCREENSHOT_DIR, OUTPUT_DIR
from gem3.visual_agent import (
    render_viewport, VisualSearchResult, ZoomStep,
    CANVAS_W, CANVAS_H, RENDER_W, RENDER_H,
)
from gem3.models import KnowledgeNode



FPS = 30
HOLD_FRAMES = 60        # Hold each decision for 2 seconds
ZOOM_FRAMES = 45        # 1.5 seconds of smooth zoom
FINAL_HOLD = 90         # 3 seconds on answer
FRAME_W = 1200
FRAME_H = 800


def _get_font(size: int):
    """Get a font at the given size."""
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _ease_in_out(t: float) -> float:
    """Smooth easing function (cubic ease-in-out)."""
    if t < 0.5:
        return 4 * t * t * t
    return 1 - pow(-2 * t + 2, 3) / 2


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def _draw_decision_overlay(
    img: Image.Image,
    step_num: int,
    total_steps: int,
    reasoning: str,
    target_desc: str,
    zoom_level: float,
    target_box: tuple[float, float, float, float] | None = None,
    is_answer: bool = False,
    answer_text: str = "",
    confidence: float = 0.0,
):
    """Draw decision reasoning overlay on the image."""
    draw = ImageDraw.Draw(img)
    
    # Semi-transparent top bar
    bar_height = 65
    for y in range(bar_height):
        alpha = int(200 * (1 - y / bar_height))
        draw.line([(0, y), (FRAME_W, y)], fill=(30, 30, 40, alpha) if img.mode == 'RGBA' else (30, 30, 40))
    
    # Top bar: step info
    title_font = _get_font(18)
    small_font = _get_font(13)
    
    draw.text((15, 8), f"Step {step_num + 1}/{total_steps}", fill=(255, 255, 255), font=title_font)
    draw.text((15, 32), f"zoom {zoom_level:.1f}x", fill=(180, 220, 255), font=small_font)
    
    # Reasoning text (right side of top bar)
    reason_short = reasoning[:90] + ("..." if len(reasoning) > 90 else "")
    draw.text((200, 12), reason_short, fill=(220, 220, 230), font=small_font)
    
    # Target description
    if target_desc:
        target_short = target_desc[:80]
        draw.text((200, 32), f"Target: {target_short}", fill=(100, 255, 150), font=small_font)
    
    # Draw target selection box (where the agent will zoom)
    if target_box:
        bx, by, bw, bh = target_box
        # Animated dashed border
        for i in range(4):
            offset = i * 2
            draw.rectangle(
                [bx - offset, by - offset, bx + bw + offset, by + bh + offset],
                outline=(0, 200, 255, 180) if i == 0 else (0, 200, 255, 60),
                width=2 if i == 0 else 1,
            )
        # Crosshair at center
        cx, cy = bx + bw / 2, by + bh / 2
        draw.line([(cx - 15, cy), (cx + 15, cy)], fill=(0, 200, 255), width=1)
        draw.line([(cx, cy - 15), (cx, cy + 15)], fill=(0, 200, 255), width=1)
    
    # Bottom bar: progress
    bar_y = FRAME_H - 40
    draw.rectangle([0, bar_y, FRAME_W, FRAME_H], fill=(30, 30, 40))
    
    # Progress bar
    progress = (step_num + 1) / total_steps
    draw.rectangle([15, bar_y + 12, 15 + int(200 * progress), bar_y + 18], fill=(0, 200, 255))
    draw.rectangle([15, bar_y + 12, 215, bar_y + 18], outline=(80, 80, 100))
    
    draw.text((230, bar_y + 8), "gem3 Visual Search Agent", fill=(120, 120, 140), font=small_font)
    
    # Answer overlay (final frame)
    if is_answer and answer_text:
        _draw_answer_panel(draw, answer_text, confidence)


def _draw_answer_panel(draw: ImageDraw.Draw, answer: str, confidence: float):
    """Draw the answer panel overlay."""
    panel_y = FRAME_H // 2 - 80
    panel_h = 160
    
    # Semi-transparent background
    draw.rectangle(
        [40, panel_y, FRAME_W - 40, panel_y + panel_h],
        fill=(20, 40, 20),
        outline=(0, 200, 100),
        width=2,
    )
    
    title_font = _get_font(16)
    answer_font = _get_font(14)
    
    draw.text((60, panel_y + 10), "Answer Found", fill=(0, 255, 120), font=title_font)
    
    conf_color = (0, 255, 100) if confidence >= 0.8 else (255, 200, 0)
    draw.text((FRAME_W - 160, panel_y + 10), f"Confidence: {confidence:.0%}", fill=conf_color, font=title_font)
    
    # Wrap answer text
    words = answer.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > 85:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    
    for i, line in enumerate(lines[:5]):
        draw.text((60, panel_y + 40 + i * 20), line, fill=(220, 255, 220), font=answer_font)


def generate_animation(
    result: VisualSearchResult,
    tree: KnowledgeNode,
    highlighted_ids: set[str] | None = None,
    output_path: str | None = None,
) -> str:
    """
    Generate a cinematic MP4 animation of the visual search traversal.
    
    Returns the path to the generated video file.
    """
    if output_path is None:
        output_path = str(OUTPUT_DIR / "search_traversal.mp4")
    
    frames_dir = OUTPUT_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old frames
    for f in frames_dir.glob("*.png"):
        f.unlink()
    
    frame_num = 0
    total_steps = len(result.steps)
    
    print(f"\n   Generating animation ({total_steps} steps)...")
    
    for step_idx, step in enumerate(result.steps):
        vx, vy, vw, vh = step.viewport
        
        # Get next viewport for zoom interpolation
        if step_idx + 1 < total_steps:
            next_step = result.steps[step_idx + 1]
            nvx, nvy, nvw, nvh = next_step.viewport
        else:
            nvx, nvy, nvw, nvh = vx, vy, vw, vh
        
        # Calculate target selection box (in pixel coords)
        if step_idx + 1 < total_steps:
            # Box showing where we'll zoom to
            box_x = (nvx - vx) / vw * FRAME_W
            box_y = (nvy - vy) / vh * FRAME_H
            box_w = nvw / vw * FRAME_W
            box_h = nvh / vh * FRAME_H
            target_box = (box_x, box_y, box_w, box_h)
        else:
            target_box = None
        
        # --- HOLD phase: show current view with decision overlay ---
        print(f"     Step {step_idx + 1}: rendering hold frames...")
        
        img = render_viewport(tree, vx, vy, vw, vh, result.query, highlighted_ids)
        
        is_last = step_idx == total_steps - 1
        hold_count = FINAL_HOLD if is_last else HOLD_FRAMES
        
        for f in range(hold_count):
            frame = img.copy()
            
            # Fade in the decision overlay
            fade = min(1.0, f / 15) if f < 15 else 1.0
            
            _draw_decision_overlay(
                frame,
                step_num=step_idx,
                total_steps=total_steps,
                reasoning=step.reasoning,
                target_desc=step.target_region,
                zoom_level=step.zoom_level,
                target_box=target_box if f >= 20 else None,  # Show box after 20 frames
                is_answer=False,  # Don't overlay answer panel — let the visual speak
                answer_text="",
                confidence=0.0,
            )
            
            frame_path = frames_dir / f"frame_{frame_num:05d}.png"
            frame.save(frame_path)
            frame_num += 1
        
        # --- ZOOM phase: smooth interpolation to next viewport ---
        if step_idx + 1 < total_steps:
            print(f"     Step {step_idx + 1}: rendering zoom transition...")
            
            for f in range(ZOOM_FRAMES):
                t = _ease_in_out(f / (ZOOM_FRAMES - 1))
                
                # Interpolate viewport
                ivx = _lerp(vx, nvx, t)
                ivy = _lerp(vy, nvy, t)
                ivw = _lerp(vw, nvw, t)
                ivh = _lerp(vh, nvh, t)
                
                izoom = CANVAS_W / ivw
                
                img = render_viewport(tree, ivx, ivy, ivw, ivh, result.query, highlighted_ids)
                
                _draw_decision_overlay(
                    img,
                    step_num=step_idx,
                    total_steps=total_steps,
                    reasoning=f"Zooming to {step.target_region}..." if step.target_region else "Zooming...",
                    target_desc="",
                    zoom_level=izoom,
                )
                
                frame_path = frames_dir / f"frame_{frame_num:05d}.png"
                img.save(frame_path)
                frame_num += 1
    
    print(f"     Total frames: {frame_num}")
    
    # Encode with ffmpeg
    has_ffmpeg = shutil.which("ffmpeg") is not None
    
    if has_ffmpeg:
        print(f"  🎥 Encoding MP4...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={FRAME_W}:{FRAME_H}",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True)
        print(f"   Video saved: {output_path}")
    else:
        # Fallback: animated GIF
        output_path = str(OUTPUT_DIR / "search_traversal.gif")
        print(f"   ffmpeg not found, generating GIF...")
        
        # Sample every 3rd frame for smaller GIF
        gif_frames = []
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        for i, fp in enumerate(frame_files):
            if i % 3 == 0:
                gif_frames.append(Image.open(fp).resize((600, 400)))
        
        if gif_frames:
            gif_frames[0].save(
                output_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=100,  # 100ms per frame (10fps after sampling)
                loop=0,
            )
            print(f"   GIF saved: {output_path}")
    
    # Clean up frames
    # (keep them for debugging, user can delete)
    
    return output_path


def generate_explainability_html(result: VisualSearchResult) -> str:
    """
    Generate an interactive HTML page showing the full search explainability.
    
    Shows step-by-step screenshots with reasoning, zoom path, and answer.
    """
    steps_html = ""
    for step in result.steps:
        img_name = Path(step.image_path).name if step.image_path else ""
        steps_html += f"""
        <div class="step" id="step-{step.step_num}">
          <div class="step-header">
            <span class="step-num">Step {step.step_num + 1}</span>
            <span class="zoom-level">{step.zoom_level:.1f}x zoom</span>
            <span class="node-count">{step.visible_nodes} nodes visible</span>
          </div>
          <div class="step-body">
            <img src="screenshots/{img_name}" alt="Step {step.step_num + 1}" class="step-img" />
            <div class="reasoning">
              <h4>Agent Reasoning</h4>
              <p>{step.reasoning}</p>
              {f'<p class="target">Target: {step.target_region}</p>' if step.target_region else ''}
            </div>
          </div>
        </div>
        """
    
    answer_html = ""
    if result.answer:
        sources = "".join([
            f'<li><strong>{e.get("doc", "")}</strong>: {e.get("content", "")[:200]}</li>'
            for e in result.source_excerpts
        ])
        answer_html = f"""
        <div class="answer-panel">
          <h2>Answer</h2>
          <p class="answer-text">{result.answer}</p>
          <h3>Source Excerpts</h3>
          <ul>{sources}</ul>
        </div>
        """
    else:
        answer_html = """
        <div class="answer-panel no-answer">
          <h2>No Definitive Answer Found</h2>
          <p>The agent explored the knowledge base but could not find source excerpts 
          directly matching this query. This may mean the topic is not covered in the 
          current document collection.</p>
        </div>
        """
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>gem3 Search: {result.query}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0f; color: #e0e0e0; padding: 40px;
  }}
  .header {{ 
    text-align: center; margin-bottom: 40px; padding: 30px;
    background: linear-gradient(135deg, #0a1628, #1a0a28);
    border-radius: 16px; border: 1px solid #1a2a4a;
  }}
  .header h1 {{ font-size: 28px; color: #fff; margin-bottom: 8px; }}
  .header .query {{ font-size: 18px; color: #60a0ff; font-style: italic; }}
  .header .stats {{ font-size: 14px; color: #888; margin-top: 10px; }}
  
  .timeline {{ 
    position: relative; padding-left: 40px; margin: 30px 0;
  }}
  .timeline::before {{
    content: ''; position: absolute; left: 18px; top: 0; bottom: 0;
    width: 3px; background: linear-gradient(to bottom, #00c8ff, #00ff80);
    border-radius: 3px;
  }}
  
  .step {{
    position: relative; margin-bottom: 30px;
    background: #12121a; border: 1px solid #2a2a3a; border-radius: 12px;
    overflow: hidden; transition: all 0.3s;
  }}
  .step:hover {{ border-color: #00c8ff; transform: translateX(4px); }}
  .step::before {{
    content: ''; position: absolute; left: -31px; top: 24px;
    width: 14px; height: 14px; border-radius: 50%;
    background: #00c8ff; border: 3px solid #0a0a0f;
  }}
  
  .step-header {{
    display: flex; gap: 20px; padding: 16px 20px;
    background: #1a1a2a; border-bottom: 1px solid #2a2a3a;
  }}
  .step-num {{ font-weight: 700; color: #00c8ff; font-size: 16px; }}
  .zoom-level {{ color: #ffaa00; font-size: 14px; }}
  .node-count {{ color: #888; font-size: 14px; }}
  
  .step-body {{ display: flex; gap: 20px; padding: 20px; }}
  .step-img {{ 
    width: 500px; height: auto; border-radius: 8px;
    border: 1px solid #2a2a3a;
  }}
  .reasoning {{ flex: 1; }}
  .reasoning h4 {{ color: #aaa; font-size: 13px; margin-bottom: 8px; text-transform: uppercase; }}
  .reasoning p {{ color: #ccc; line-height: 1.6; font-size: 14px; }}
  .reasoning .target {{ color: #00ff80; margin-top: 10px; font-weight: 500; }}
  
  .answer-panel {{
    margin-top: 40px; padding: 30px;
    background: linear-gradient(135deg, #0a2810, #0a1a28);
    border: 2px solid #00ff80; border-radius: 16px;
  }}
  .answer-panel h2 {{ color: #00ff80; margin-bottom: 15px; }}
  .answer-panel .answer-text {{ 
    font-size: 18px; line-height: 1.7; color: #e0ffe0;
    margin-bottom: 20px;
  }}
  .answer-panel h3 {{ color: #aaa; margin-bottom: 10px; font-size: 14px; }}
  .answer-panel ul {{ padding-left: 20px; }}
  .answer-panel li {{ color: #ccc; margin-bottom: 8px; line-height: 1.5; }}
  .answer-panel li strong {{ color: #60a0ff; }}
  
  .no-answer {{ border-color: #ff8800; background: linear-gradient(135deg, #281a0a, #1a0a28); }}
  .no-answer h2 {{ color: #ff8800; }}
</style>
</head>
<body>
<div class="header">
  <h1>gem3 Visual Search</h1>
  <div class="query">"{result.query}"</div>
  <div class="stats">
    {len(result.steps)} steps | {result.total_time:.1f}s | 
    {'Answer found' if result.answer else 'No match in knowledge base'}
  </div>
</div>

<div class="timeline">
  {steps_html}
</div>

{answer_html}

</body>
</html>"""
    
    output_path = str(OUTPUT_DIR / "search_explainability.html")
    with open(output_path, "w") as f:
        f.write(html)
    
    return output_path
