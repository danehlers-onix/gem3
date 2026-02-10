#!/bin/bash
# gem3 deployment script — deploy the knowledge zoom to the public internet
#
# Usage:
#   ./deploy.sh fly          # Deploy to Fly.io (free tier)
#   ./deploy.sh cloudrun     # Deploy to Google Cloud Run
#   ./deploy.sh docker       # Just build & run locally in Docker
#
# Prerequisites:
#   - Knowledge tree already built: python main.py demo
#   - GEMINI_API_KEY set (only needed for build-time tree generation)

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check that knowledge tree exists
if [ ! -f "output/knowledge_tree.pkl" ]; then
    echo "⚠️  No knowledge tree found. Building one first..."
    echo "   Run: python main.py demo"
    echo "   Or:  python main.py ingest path/to/docs/"
    exit 1
fi

case "${1:-help}" in

  fly)
    echo -e "${CYAN}🚀 Deploying to Fly.io...${NC}"
    
    # Check fly CLI
    if ! command -v fly &> /dev/null; then
        echo "Installing flyctl..."
        curl -L https://fly.io/install.sh | sh
    fi
    
    # Launch or deploy
    if ! fly status &> /dev/null; then
        echo -e "${GREEN}Creating new Fly app...${NC}"
        fly launch --no-deploy --copy-config
    fi
    
    # Deploy with the pre-built tree baked in
    echo -e "${GREEN}Deploying...${NC}"
    fly deploy
    
    echo -e "${GREEN}✅ Deployed! Opening...${NC}"
    fly open
    ;;

  cloudrun)
    echo -e "${CYAN}🚀 Deploying to Google Cloud Run...${NC}"
    
    PROJECT=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT" ]; then
        echo "Set your GCP project: gcloud config set project YOUR_PROJECT"
        exit 1
    fi
    
    REGION=${GCP_REGION:-us-central1}
    IMAGE="gcr.io/$PROJECT/gem3-zoom"
    
    # Build and push
    echo -e "${GREEN}Building container...${NC}"
    gcloud builds submit --tag "$IMAGE" --timeout=600
    
    # Deploy
    echo -e "${GREEN}Deploying to Cloud Run...${NC}"
    gcloud run deploy gem3-zoom \
        --image "$IMAGE" \
        --region "$REGION" \
        --platform managed \
        --allow-unauthenticated \
        --memory 512Mi \
        --cpu 1 \
        --min-instances 0 \
        --max-instances 3 \
        --port 8080 \
        --timeout 300
    
    URL=$(gcloud run services describe gem3-zoom --region "$REGION" --format 'value(status.url)')
    echo -e "${GREEN}✅ Deployed to: $URL${NC}"
    open "$URL" 2>/dev/null || echo "Open: $URL"
    ;;

  docker)
    echo -e "${CYAN}🐳 Building & running locally in Docker...${NC}"
    
    docker build -t gem3-zoom .
    
    echo -e "${GREEN}Starting on http://localhost:8080${NC}"
    docker run -it --rm -p 8080:8080 gem3-zoom
    ;;

  *)
    echo "gem3 deploy — host the knowledge zoom publicly"
    echo ""
    echo "Usage:"
    echo "  ./deploy.sh fly        Deploy to Fly.io (free tier, recommended)"
    echo "  ./deploy.sh cloudrun   Deploy to Google Cloud Run"
    echo "  ./deploy.sh docker     Build & run locally in Docker"
    echo ""
    echo "Prerequisites:"
    echo "  1. Build knowledge tree: python main.py demo"
    echo "  2. Or ingest your docs: python main.py ingest path/to/docs/"
    echo ""
    echo "The deployed app serves the pre-built knowledge zoom — no API key"
    echo "needed at runtime. Users can search and zoom into the knowledge base"
    echo "directly in their browser."
    ;;
esac
