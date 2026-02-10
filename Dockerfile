FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Vertex AI mode — no API key needed on Cloud Run (uses service account IAM)
ENV USE_VERTEX_AI=true
ENV GCP_PROJECT=prj-ox-int-g-looker
ENV GCP_LOCATION=global
ENV GEMINI_MODEL=gemini-3-flash-preview

# Pre-build knowledge tree from BigQuery at build time (optional)
# If a knowledge_tree.pkl already exists, skip this step
RUN if [ ! -f output/knowledge_tree.pkl ]; then \
    echo "No pre-built tree found. Tree will be built at first run."; \
    fi

# Runtime: serve grounded zoom
ENV PORT=8080
EXPOSE 8080

CMD ["python", "-c", "import asyncio; from server import start_server; import pickle; from pathlib import Path; tree=pickle.load(open('output/knowledge_tree.pkl','rb')) if Path('output/knowledge_tree.pkl').exists() else None; asyncio.run(start_server(port=8080, tree=tree))"]
