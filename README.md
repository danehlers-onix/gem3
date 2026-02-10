# 🌌 gem3 — Visual RAG: Infinite Zoom Knowledge Navigation

> No embeddings. No keywords. Pure vision.

A novel retrieval-augmented generation system that uses **Gemini Flash's image understanding** to create a visual, infinite-zoom approach to knowledge retrieval. Ingest your documents, zoom from topics down to source excerpts, and host it publicly — all from a single codebase.

## ⚡ Quick Start

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"

# Build knowledge tree from demo docs (one-time Gemini call)
python main.py demo

# Launch interactive zoom server
python main.py rag
# → Open http://localhost:3333
```

## How It Works

Documents are ingested into a **hierarchical knowledge tree** — from library → topics → subtopics → chapters → sections → concepts → **verbatim source excerpts**. The hierarchy is rendered as a single zoomable canvas where text size reflects depth:

```
🔤 HUGE FAINT TEXT     = broad topics (visible at overview)
  🔤 Medium text       = subtopics & chapters
    🔤 small dark text  = concepts & source excerpts (visible when zoomed)
```

Zooming in reveals deeper layers. Parent text **automatically fades** as you zoom, keeping the view clean. Query-relevant paths are **highlighted** (darkened/bolded) to guide navigation.

## Commands

| Command | What it does |
|---------|-------------|
| `python main.py rag` | Launch grounded zoom server (pre-built tree) |
| `python main.py search "query"` | Visual agent search → MP4 animation + explainability |
| `python main.py export` | Export standalone HTML (host anywhere, zero cost) |
| `python main.py export "query"` | Export with query highlighting baked in |
| `python main.py zoom` | Launch infinite generative zoom (Gemini generates on-the-fly) |
| `python main.py ingest ./docs/` | Ingest your own documents |
| `python main.py demo` | Build knowledge tree from sample science articles |

## 🚀 Public Hosting

The grounded zoom mode is **100% client-side** after page load — no API key needed at runtime, no server computation during zoom. This means multiple zero-cost hosting options:

### Option 1: Static HTML Export (Simplest)

```bash
# Export a single self-contained HTML file
python main.py export

# With a pre-baked query
python main.py export "quantum entanglement"

# Output: output/gem3_export.html (~50-200KB)
# Host on: GitHub Pages, Netlify, Vercel, S3, or just share the file
```

### Option 2: Fly.io (Free Tier)

```bash
chmod +x deploy.sh
./deploy.sh fly
# → Deploys to https://gem3-zoom.fly.dev
```

### Option 3: Google Cloud Run

```bash
./deploy.sh cloudrun
# → Deploys to https://gem3-zoom-xxxxx.run.app
```

### Option 4: Docker

```bash
./deploy.sh docker
# → Runs on http://localhost:8080
```

### Option 5: GitHub Pages (Free)

```bash
python main.py export
# Copy output/gem3_export.html → docs/index.html
# Push to GitHub → Settings → Pages → Deploy from docs/
```

## 🔍 Visual Search Agent

The `search` command launches a Gemini-powered agent that **sees** the canvas and visually navigates to answer queries:

```bash
python main.py search "how does quantum cryptography work"
```

This produces:
- **MP4 animation** showing the agent's zoom path
- **HTML explainability page** with step-by-step reasoning
- **Screenshots** at each decision point

The agent renders the canvas at each zoom level, sends the image to Gemini, gets back a zoom decision with reasoning, and repeats until source excerpts are readable.

## Architecture

```
gem3/
├── models.py        # KnowledgeNode data model
├── config.py        # Configuration (API keys, paths)
├── ingest.py        # Documents → hierarchical knowledge tree
├── visual_agent.py  # Gemini vision agent (sees canvas, decides where to zoom)
├── animate.py       # Cinematic MP4 generation of search traversal
├── navigator.py     # Visual navigation (legacy)
├── visualize.py     # HTML visualization (legacy)
server.py            # aiohttp server (grounded + generative zoom modes)
main.py              # CLI entry point
Dockerfile           # Container deployment
fly.toml             # Fly.io config
deploy.sh            # One-command deployment script
```

## What Makes This Novel

- **100% Visual Retrieval**: No vector embeddings — the model literally *looks* at the rendered canvas to find information
- **Zoom-as-Retrieval**: The hierarchy IS the retrieval mechanism. Zooming traces the path from query to source.
- **Grounded Excerpts**: Every leaf node is a real document chunk with provenance
- **Automatic Fade**: Parent text fades as you zoom deep, keeping excerpts clean and readable
- **Zero-Cost Hosting**: Export as a single HTML file — no server, no API key needed at runtime
- **Explainability**: The zoom path IS the explanation. Every navigation decision is recorded and visualizable.

## Ingest Your Own Documents

```bash
# From a directory of text/markdown files
python main.py ingest ./my_documents/

# Then serve or export
python main.py rag
python main.py export
```

Gemini analyzes your documents and builds the hierarchy automatically — deducing topics, subtopics, and organizing chunks into a navigable tree.
