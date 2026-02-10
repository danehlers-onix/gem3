"""
Data models for the visual knowledge hierarchy.

The hierarchy is:
  Library (top-level overview)
    → Topic (broad subject area)
      → Subtopic (narrower focus)
        → Chapter (a specific article/document)
          → Section (part of a document)
            → Concept (atomic idea)
              → Excerpt (verbatim passage)

Each node at every level can be rendered as a visual HTML tile.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class KnowledgeNode(BaseModel):
    """A single node in the knowledge hierarchy."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    level: str  # library, topic, subtopic, chapter, section, concept, excerpt
    title: str
    summary: str = ""
    content: str = ""  # full text at this level
    children: list[KnowledgeNode] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    # Visual state
    html_cache: Optional[str] = None
    screenshot_path: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class ZoomState(BaseModel):
    """Tracks the current zoom position in the visual navigation."""
    path: list[str] = Field(default_factory=list)  # list of node IDs from root to current
    current_node_id: str = ""
    zoom_level: int = 0  # 0 = highest (library overview)
    query: str = ""
    parallel_paths: list[list[str]] = Field(default_factory=list)  # for parallel traversal


class NavigationDecision(BaseModel):
    """A decision made by Gemini about where to zoom next."""
    target_region: str = ""  # description of where to zoom
    target_node_ids: list[str] = Field(default_factory=list)  # which nodes to zoom into
    reasoning: str = ""
    confidence: float = 0.0
    should_synthesize: bool = False  # whether to combine multiple paths
    is_terminal: bool = False  # reached the deepest useful level


class RetrievalResult(BaseModel):
    """Final result of a visual RAG retrieval."""
    query: str
    zoom_trace: list[ZoomState] = Field(default_factory=list)
    retrieved_nodes: list[KnowledgeNode] = Field(default_factory=list)
    synthesis: str = ""
    html_trace: list[str] = Field(default_factory=list)  # HTML at each zoom level
    screenshot_trace: list[str] = Field(default_factory=list)  # screenshots at each level


# Hierarchy level ordering
ZOOM_LEVELS = [
    "library",
    "topic", 
    "subtopic",
    "chapter",
    "section",
    "concept",
    "excerpt",
]
