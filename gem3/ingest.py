"""
Knowledge Ingestion — Grounded RAG Hierarchy Builder.

The hierarchy is built BOTTOM-UP from real document chunks:
  1. Documents → split into real paragraphs/chunks (LEAVES)
  2. Gemini groups chunks → concepts (related ideas)
  3. Gemini groups concepts → sections → chapters → subtopics → topics
  4. Pre-compute zoom positions for instant rendering

Every leaf is REAL text. Every upper level traces back to real chunks.
The zoom IS the RAG — zooming is how you see the retrieval path.
"""

from __future__ import annotations
import json
import math
import hashlib
from pathlib import Path
from google import genai
from google.genai import types

from gem3.config import get_gemini_client, GEMINI_MODEL
from gem3.models import KnowledgeNode, ZOOM_LEVELS


CANVAS_WIDTH = 16000
CANVAS_HEIGHT = 10000

LEVEL_FONT_SIZES = {
    "library": 800, "topic": 400, "subtopic": 150, "chapter": 60,
    "section": 30, "concept": 16, "excerpt": 10,
}
LEVEL_GRAY = {
    "library": 232, "topic": 200, "subtopic": 175, "chapter": 145,
    "section": 110, "concept": 70, "excerpt": 30,
}
LEVEL_SPREAD = {
    "library": 5000, "topic": 2000, "subtopic": 800, "chapter": 350,
    "section": 150, "concept": 70, "excerpt": 30,
}


def get_client() -> genai.Client:
    return get_gemini_client()




def chunk_document(title: str, content: str, doc_index: int) -> list[dict]:
    """
    Split a document into real chunks (paragraphs/passages).
    Each chunk is a LEAF that will become an excerpt node.
    Returns list of {text, doc_index, doc_title, char_start, char_end}.
    """
    chunks = []
    # Split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    char_pos = 0
    for para in paragraphs:
        # Skip very short paragraphs (headers, etc.)
        if len(para) < 30:
            char_pos = content.find(para, char_pos) + len(para)
            continue
        
        # If paragraph is too long, split into sentences
        if len(para) > 500:
            sentences = _split_sentences(para)
            # Group sentences into ~200-400 char chunks
            current_chunk = ""
            chunk_start = content.find(para, char_pos)
            for sent in sentences:
                if len(current_chunk) + len(sent) > 400 and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "doc_index": doc_index,
                        "doc_title": title,
                        "char_start": chunk_start,
                        "char_end": chunk_start + len(current_chunk),
                    })
                    chunk_start += len(current_chunk)
                    current_chunk = sent + " "
                else:
                    current_chunk += sent + " "
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "doc_index": doc_index,
                    "doc_title": title,
                    "char_start": chunk_start,
                    "char_end": chunk_start + len(current_chunk),
                })
        else:
            start = content.find(para, char_pos)
            chunks.append({
                "text": para,
                "doc_index": doc_index,
                "doc_title": title,
                "char_start": start,
                "char_end": start + len(para),
            })
        
        char_pos = content.find(para, char_pos) + len(para)
    
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]




def ingest_documents(documents: list[dict[str, str]]) -> KnowledgeNode:
    """
    Build a grounded knowledge hierarchy from real documents.
    
    BOTTOM-UP: Real chunks → concepts → sections → chapters → topics
    Every upper level traces back to real document text.
    """
    client = get_client()
    
    # Step 1: Chunk all documents into real excerpts
    print("   Chunking documents into real excerpts...")
    all_chunks = []
    for i, doc in enumerate(documents):
        chunks = chunk_document(doc["title"], doc["content"], i)
        all_chunks.extend(chunks)
        print(f"    → {doc['title']}: {len(chunks)} chunks")
    
    print(f"   Total chunks: {len(all_chunks)}")
    
    # Step 2: Gemini builds hierarchy from chunks
    print("  🧠 Gemini deducing hierarchy from chunks...")
    hierarchy = _build_hierarchy_from_chunks(client, all_chunks, documents)
    
    # Step 3: Pre-compute zoom positions
    print("  📐 Pre-computing zoom positions...")
    _precompute_positions(hierarchy, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)
    
    return hierarchy


def _build_hierarchy_from_chunks(
    client: genai.Client,
    chunks: list[dict],
    documents: list[dict],
) -> KnowledgeNode:
    """
    Use Gemini to build a hierarchy from real chunks.
    The chunks are preserved as leaves — Gemini only deduces the UPPER levels.
    """
    # Prepare chunk summaries for Gemini (with indices for reference)
    chunk_refs = []
    for i, chunk in enumerate(chunks):
        # Show first 150 chars of each chunk
        preview = chunk["text"][:150].replace("\n", " ")
        chunk_refs.append(f"[{i}] (doc: {chunk['doc_title']}): {preview}")
    
    chunks_text = "\n".join(chunk_refs)
    doc_titles = [d["title"] for d in documents]
    
    prompt = f"""You have {len(chunks)} text chunks from {len(documents)} documents.
Documents: {json.dumps(doc_titles)}

Chunks:
{chunks_text}

Build a HIERARCHICAL knowledge structure that organizes these chunks.
The chunks are the LEAVES — do NOT modify or summarize them.
Your job is to create the UPPER levels that group and organize them.

Structure:
- 1 "library" root (the whole collection)
  - 2-5 "topic" nodes (broad subject areas)
    - 2-4 "subtopic" nodes each (narrower focus)
      - 1-3 "chapter" nodes each (related to specific docs)
        - 1-4 "section" nodes each (thematic groupings)
          - "concept" nodes with chunk_indices pointing to real chunks

For each node provide: title, summary, level, children.
For concept nodes (leaves), include "chunk_indices": [list of chunk indices this concept covers].

Return ONLY valid JSON. No markdown, no explanation."""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.5),
        )
        
        text = response.text.strip()
        # Find JSON
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            hierarchy_data = json.loads(text[start:end])
        else:
            print("   Could not parse hierarchy JSON, using fallback")
            return _create_grounded_fallback(chunks, documents)
            
    except Exception as e:
        print(f"   Hierarchy building failed: {e}")
        return _create_grounded_fallback(chunks, documents)
    
    # Convert JSON to KnowledgeNode tree, attaching real chunks as leaves
    return _json_to_grounded_tree(hierarchy_data, chunks)


def _json_to_grounded_tree(data: dict, chunks: list[dict]) -> KnowledgeNode:
    """
    Convert Gemini's hierarchy JSON into a KnowledgeNode tree.
    Concept nodes with chunk_indices get real excerpt children.
    """
    def build_node(d: dict, depth: int = 0) -> KnowledgeNode:
        level = d.get("level", ZOOM_LEVELS[min(depth, len(ZOOM_LEVELS) - 1)])
        
        node = KnowledgeNode(
            level=level,
            title=d.get("title", "Untitled"),
            summary=d.get("summary", ""),
            metadata=d.get("metadata", {}),
        )
        
        # If this node has chunk_indices, create excerpt children from REAL text
        chunk_indices = d.get("chunk_indices", [])
        if chunk_indices:
            node.metadata["chunk_indices"] = chunk_indices
            for ci in chunk_indices:
                if 0 <= ci < len(chunks):
                    chunk = chunks[ci]
                    excerpt = KnowledgeNode(
                        level="excerpt",
                        title=chunk["text"][:40] + "..." if len(chunk["text"]) > 40 else chunk["text"],
                        summary="",
                        content=chunk["text"],  # REAL document text
                        metadata={
                            "chunk_index": ci,
                            "doc_index": chunk["doc_index"],
                            "doc_title": chunk["doc_title"],
                            "char_start": chunk["char_start"],
                            "char_end": chunk["char_end"],
                            "grounded": True,  # Flag: this is real text
                        },
                    )
                    node.children.append(excerpt)
        
        # Process child nodes
        for child_data in d.get("children", []):
            if isinstance(child_data, dict):
                node.children.append(build_node(child_data, depth + 1))
        
        return node
    
    return build_node(data)


def _create_grounded_fallback(chunks: list[dict], documents: list[dict]) -> KnowledgeNode:
    """Fallback: group chunks by document."""
    root = KnowledgeNode(
        level="library",
        title="Knowledge Library",
        summary=f"{len(documents)} documents, {len(chunks)} chunks",
    )
    
    topic = KnowledgeNode(
        level="topic",
        title="Documents",
        summary="All ingested documents",
    )
    
    # Group chunks by document
    doc_chunks: dict[int, list[dict]] = {}
    for chunk in chunks:
        di = chunk["doc_index"]
        if di not in doc_chunks:
            doc_chunks[di] = []
        doc_chunks[di].append(chunk)
    
    for di, dchunks in doc_chunks.items():
        doc = documents[di]
        chapter = KnowledgeNode(
            level="chapter",
            title=doc["title"],
            summary=doc["content"][:150],
            content=doc["content"],
            metadata={"doc_index": di},
        )
        
        # Add real chunks as excerpts
        for ci_offset, chunk in enumerate(dchunks):
            excerpt = KnowledgeNode(
                level="excerpt",
                title=chunk["text"][:40] + "...",
                summary="",
                content=chunk["text"],
                metadata={
                    "doc_index": di,
                    "doc_title": doc["title"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                    "grounded": True,
                },
            )
            chapter.children.append(excerpt)
        
        topic.children.append(chapter)
    
    root.children.append(topic)
    return root




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


def _precompute_positions(
    node: KnowledgeNode,
    cx: float,
    cy: float,
    depth: int = 0,
    parent_angle: float = 0,
):
    """
    Pre-compute x, y, angle for every node in the tree.
    Stored in node.metadata for instant rendering.
    """
    spread = LEVEL_SPREAD.get(node.level, 50)
    angle = _hash_angle(node.title, depth)
    angle = angle * 0.7 + parent_angle * 0.3
    
    # Clamp to canvas
    cx = max(200, min(cx, CANVAS_WIDTH - 200))
    cy = max(200, min(cy, CANVAS_HEIGHT - 200))
    
    node.metadata["x"] = cx
    node.metadata["y"] = cy
    node.metadata["angle"] = angle
    node.metadata["font_size"] = LEVEL_FONT_SIZES.get(node.level, 10)
    node.metadata["gray"] = LEVEL_GRAY.get(node.level, 100)
    
    # Position children around this node
    if node.children:
        n = len(node.children)
        for i, child in enumerate(node.children):
            child_angle_rad = (2 * math.pi * i / n) + _hash_angle(node.title, i) * 0.02
            dx, dy = _hash_offset(child.title, spread, i)
            dx += math.cos(child_angle_rad) * spread * 0.5
            dy += math.sin(child_angle_rad) * spread * 0.5
            
            _precompute_positions(child, cx + dx, cy + dy, depth + 1, angle)




def ingest_text(title: str, content: str) -> KnowledgeNode:
    """Ingest a single document."""
    return ingest_documents([{"title": title, "content": content}])


def ingest_directory(path: Path) -> KnowledgeNode:
    """Ingest all .txt and .md files from a directory."""
    documents = []
    for f in sorted(path.iterdir()):
        if f.suffix in (".txt", ".md", ".pdf"):
            documents.append({
                "title": f.stem.replace("_", " ").replace("-", " ").title(),
                "content": f.read_text(encoding="utf-8", errors="replace"),
            })
    if not documents:
        raise ValueError(f"No .txt or .md files found in {path}")
    return ingest_documents(documents)




def tree_to_dict(node: KnowledgeNode) -> dict:
    """Serialize the tree to a dict for JSON storage."""
    d = {
        "id": node.id,
        "level": node.level,
        "title": node.title,
        "summary": node.summary,
        "content": node.content,
        "metadata": node.metadata,
        "children": [tree_to_dict(c) for c in node.children],
    }
    return d


def dict_to_tree(d: dict) -> KnowledgeNode:
    """Deserialize a dict back to a KnowledgeNode tree."""
    node = KnowledgeNode(
        id=d.get("id", ""),
        level=d["level"],
        title=d["title"],
        summary=d.get("summary", ""),
        content=d.get("content", ""),
        metadata=d.get("metadata", {}),
    )
    for child_data in d.get("children", []):
        node.children.append(dict_to_tree(child_data))
    return node


def count_nodes(node: KnowledgeNode) -> dict[str, int]:
    """Count nodes at each level."""
    counts: dict[str, int] = {}
    def _count(n):
        counts[n.level] = counts.get(n.level, 0) + 1
        for c in n.children:
            _count(c)
    _count(node)
    return counts


def count_grounded(node: KnowledgeNode) -> int:
    """Count how many excerpt nodes are grounded (real text)."""
    count = 0
    def _count(n):
        nonlocal count
        if n.metadata.get("grounded"):
            count += 1
        for c in n.children:
            _count(c)
    _count(node)
    return count
