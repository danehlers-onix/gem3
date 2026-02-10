"""
Configuration for gem3.

Supports two modes:
  - Vertex AI (GCP): Uses IAM auth, higher quota, no API key needed
  - API Key: For local dev, uses GEMINI_API_KEY env var
  
On Cloud Run / GCP, Vertex AI is used automatically.
"""

import os
from pathlib import Path
from google import genai

# GCP Project
GCP_PROJECT = os.environ.get("GCP_PROJECT", "prj-ox-int-g-looker")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")

# Gemini model — gemini-3-flash-preview requires global location
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

# Auth mode: prefer Vertex AI, fall back to API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
USE_VERTEX_AI = os.environ.get("USE_VERTEX_AI", "").lower() in ("true", "1", "yes") or (
    not GEMINI_API_KEY  # No API key → try Vertex AI
)


def get_gemini_client() -> genai.Client:
    """
    Get a Gemini client — Vertex AI on GCP, API key locally.
    
    On Cloud Run, the service account has Vertex AI access automatically.
    """
    if USE_VERTEX_AI:
        return genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
        )
    else:
        return genai.Client(api_key=GEMINI_API_KEY)


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
HTML_DIR = OUTPUT_DIR / "html"
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
DATA_DIR = PROJECT_ROOT / "data"

# Visual settings
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 900

# Navigation
MAX_ZOOM_DEPTH = 7  # library → excerpt
MAX_PARALLEL_PATHS = 3
ZOOM_CONFIDENCE_THRESHOLD = 0.3

# Ensure directories exist
for d in [OUTPUT_DIR, HTML_DIR, SCREENSHOT_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)
