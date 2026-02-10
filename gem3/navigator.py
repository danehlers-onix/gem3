"""
Visual Navigation Agent — Gemini 3 Flash looks at screenshots and decides where to zoom.

This is the novel core of Visual RAG: instead of computing similarity scores or
matching keywords, the model SEES the visual representation of knowledge and
decides where to look based on its visual understanding.

The navigation loop:
1. Generate HTML for current zoom level
2. Screenshot the HTML
3. Gemini looks at the screenshot + query → decides which region to zoom into
4. We map the visual decision back to a node ID
5. Repeat until we reach sufficient depth or the model says "found it"
"""

from __future__ import annotations
import json
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types

from gem3.config import (
    get_gemini_client, GEMINI_MODEL, HTML_DIR, SCREENSHOT_DIR,
    MAX_ZOOM_DEPTH, MAX_PARALLEL_PATHS, ZOOM_CONFIDENCE_THRESHOLD,
)
from gem3.models import KnowledgeNode, ZoomState, NavigationDecision, RetrievalResult
from gem3.visualize import generate_zoom_html, capture_screenshot_sync


def get_client() -> genai.Client:
    return get_gemini_client()


def find_node_by_id(root: KnowledgeNode, target_id: str) -> KnowledgeNode | None:
    """Find a node in the tree by its ID."""
    if root.id == target_id:
        return root
    for child in root.children:
        found = find_node_by_id(child, target_id)
        if found:
            return found
    return None


def navigate_visual(
    query: str,
    root: KnowledgeNode,
    verbose: bool = True,
) -> RetrievalResult:
    """
    Main visual navigation loop.
    
    Given a query and a knowledge tree, this function:
    1. Starts at the library (top) level
    2. Generates a visual HTML representation
    3. Screenshots it
    4. Shows the screenshot to Gemini and asks "where should I zoom?"
    5. Zooms into the selected region
    6. Repeats until reaching terminal depth or convergence
    
    Supports parallel traversal: if the model identifies multiple
    relevant regions, we follow them all.
    """
    client = get_client()
    result = RetrievalResult(query=query)
    
    # Start with the root node
    active_paths: list[tuple[KnowledgeNode, ZoomState]] = [
        (root, ZoomState(
            path=[root.title],
            current_node_id=root.id,
            zoom_level=0,
            query=query,
        ))
    ]
    
    for depth in range(MAX_ZOOM_DEPTH):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  ZOOM LEVEL {depth} — {len(active_paths)} active path(s)")
            print(f"{'='*60}")
        
        next_paths: list[tuple[KnowledgeNode, ZoomState]] = []
        
        for path_idx, (current_node, zoom_state) in enumerate(active_paths):
            if verbose:
                print(f"\n  Path {path_idx}: {' → '.join(zoom_state.path)}")
                print(f"  Current: [{current_node.level}] {current_node.title}")
            
            # If this node has no children, it's terminal
            if not current_node.children:
                if verbose:
                    print(f"  → Terminal node (no children). Collecting.")
                result.retrieved_nodes.append(current_node)
                continue
            
            # Step 1: Generate visual HTML
            if verbose:
                print(f"  → Generating visual HTML...")
            html = generate_zoom_html(current_node, zoom_state, query)
            result.html_trace.append(html)
            
            # Step 2: Screenshot
            html_path = HTML_DIR / f"zoom_{current_node.id}_{depth}.html"
            html_path.write_text(html, encoding="utf-8")
            
            if verbose:
                print(f"  → Capturing screenshot...")
            screenshot_path = capture_screenshot_sync(html_path)
            result.screenshot_trace.append(str(screenshot_path))
            current_node.screenshot_path = str(screenshot_path)
            
            # Step 3: Ask Gemini to look at the screenshot and decide where to zoom
            if verbose:
                print(f"  → Gemini analyzing visual...")
            decision = _visual_decide(client, screenshot_path, query, current_node, zoom_state)
            
            if verbose:
                print(f"  → Decision: {decision.reasoning}")
                print(f"  → Targets: {decision.target_node_ids}")
                print(f"  → Confidence: {decision.confidence:.2f}")
                print(f"  → Terminal: {decision.is_terminal}")
            
            # Step 4: Process the decision
            if decision.is_terminal:
                # Collect the current node and any targeted children
                for tid in decision.target_node_ids:
                    target = find_node_by_id(current_node, tid)
                    if target:
                        result.retrieved_nodes.append(target)
                if not decision.target_node_ids:
                    result.retrieved_nodes.append(current_node)
                continue
            
            # Zoom into the targeted nodes (parallel if multiple)
            targets_added = 0
            for tid in decision.target_node_ids:
                if targets_added >= MAX_PARALLEL_PATHS:
                    break
                target = find_node_by_id(current_node, tid)
                if target:
                    new_state = ZoomState(
                        path=zoom_state.path + [target.title],
                        current_node_id=target.id,
                        zoom_level=depth + 1,
                        query=query,
                    )
                    next_paths.append((target, new_state))
                    result.zoom_trace.append(new_state)
                    targets_added += 1
            
            # If no valid targets found, try all children with low confidence
            if targets_added == 0 and current_node.children:
                if verbose:
                    print(f"  → No valid targets, exploring first child as fallback")
                child = current_node.children[0]
                new_state = ZoomState(
                    path=zoom_state.path + [child.title],
                    current_node_id=child.id,
                    zoom_level=depth + 1,
                    query=query,
                )
                next_paths.append((child, new_state))
                result.zoom_trace.append(new_state)
        
        if not next_paths:
            if verbose:
                print(f"\n  All paths terminated at depth {depth}")
            break
        
        active_paths = next_paths
    
    # Final synthesis
    if result.retrieved_nodes:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  SYNTHESIS — {len(result.retrieved_nodes)} nodes retrieved")
            print(f"{'='*60}")
        result.synthesis = _synthesize(client, query, result.retrieved_nodes)
        if verbose:
            print(f"\n{result.synthesis}")
    
    return result


def _visual_decide(
    client: genai.Client,
    screenshot_path: Path,
    query: str,
    current_node: KnowledgeNode,
    zoom_state: ZoomState,
) -> NavigationDecision:
    """
    The core visual reasoning step.
    
    Gemini looks at a screenshot of the current zoom level and decides:
    - Which elements are most relevant to the query
    - Whether to zoom into one or multiple elements
    - Whether we've reached sufficient depth
    """
    # Load the screenshot
    image = Image.open(screenshot_path)
    
    # Build the list of available node IDs for grounding
    available_ids = []
    if current_node.children:
        for child in current_node.children:
            available_ids.append(f"  {child.id} → [{child.level}] {child.title}")
    available_ids_text = "\n".join(available_ids)
    
    prompt = f"""You are a visual knowledge navigator. You are looking at a visual representation 
of a knowledge hierarchy at the "{current_node.level}" level.

USER QUERY: "{query}"

CURRENT POSITION: {' → '.join(zoom_state.path)}
ZOOM DEPTH: {zoom_state.zoom_level} of max {MAX_ZOOM_DEPTH}

AVAILABLE CHILDREN (node IDs you can zoom into):
{available_ids_text}

Look at this visual and decide:
1. Which element(s) are most relevant to the query?
2. Should you zoom into one element, or multiple (parallel traversal)?
3. Have you reached sufficient depth to answer the query?

Respond with a JSON object (and ONLY a JSON object):
{{
  "target_node_ids": ["id1", "id2"],  // which node IDs to zoom into (1-3 max)
  "reasoning": "brief explanation of why these elements are relevant",
  "confidence": 0.85,  // 0-1 how confident you are these are the right targets
  "is_terminal": false,  // true if current depth is sufficient to answer
  "should_synthesize": false  // true if multiple paths should be combined
}}"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                temperature=0.3,
            ),
        )
        
        # Parse the JSON response
        text = response.text.strip()
        # Remove markdown code fences if present
        if text.startswith('```'):
            lines = text.split('\n')
            text = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
            text = text.strip()
        
        # Find JSON in response
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return NavigationDecision(
                target_node_ids=data.get("target_node_ids", []),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.5),
                is_terminal=data.get("is_terminal", False),
                should_synthesize=data.get("should_synthesize", False),
            )
    except Exception as e:
        print(f"    Warning: Visual decision failed: {e}")
    
    # Fallback: pick the first child
    if current_node.children:
        return NavigationDecision(
            target_node_ids=[current_node.children[0].id],
            reasoning="Fallback: selecting first child",
            confidence=0.1,
        )
    
    return NavigationDecision(
        is_terminal=True,
        reasoning="No children available",
    )


def _synthesize(
    client: genai.Client,
    query: str,
    nodes: list[KnowledgeNode],
) -> str:
    """
    Synthesize the final answer from all retrieved nodes.
    """
    context_parts = []
    for node in nodes:
        text = node.content if node.content else node.summary
        context_parts.append(f"[{node.level}] {node.title}:\n{text[:1500]}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""Based on the following knowledge retrieved through visual navigation,
answer this query:

QUERY: "{query}"

RETRIEVED KNOWLEDGE:
{context}

Provide a comprehensive, well-structured answer. Reference specific sources when possible."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.5,
        ),
    )
    
    return response.text
