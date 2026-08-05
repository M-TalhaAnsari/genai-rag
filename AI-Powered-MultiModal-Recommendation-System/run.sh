#!/bin/bash
# ============================================================
# Connoisseur — startup script
# Run: bash run.sh
# ============================================================

set -e  # exit on any error

echo ""
echo "🍽️  Connoisseur Restaurant Discovery System"
echo "============================================"

# ── Check .env ──────────────────────────────────────────────
if [ ! -f ".env" ]; then
  echo "❌  .env file not found. Copy .env.example and fill in your keys."
  echo "    cp .env.example .env"
  exit 1
fi

# ── Check Python ────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
  echo "❌  python3 not found. Install Python 3.11+."
  exit 1
fi

# ── Install dependencies ────────────────────────────────────
echo ""
echo "📦  Installing dependencies..."
pip install -r requirements.txt -q

# ── Create data directory ───────────────────────────────────
mkdir -p data

# ── Choice: what to start ───────────────────────────────────
echo ""
echo "What would you like to start?"
echo "  1) Backend API only     (uvicorn)"
echo "  2) Frontend only        (streamlit)"
echo "  3) Both (recommended)   (runs in background)"
echo "  4) MCP server only"
echo "  5) Exit"
echo ""
read -p "Choice [1-5]: " choice

case $choice in

  1)
    echo ""
    echo "🚀  Starting FastAPI backend on http://localhost:8000"
    echo "    Docs: http://localhost:8000/docs"
    echo ""
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ;;

  2)
    echo ""
    echo "🎨  Starting Streamlit frontend on http://localhost:8501"
    echo ""
    streamlit run frontend/app.py --server.port 8501
    ;;

  3)
    echo ""
    echo "🚀  Starting backend on http://localhost:8000 (background)..."
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    echo "    Backend PID: $BACKEND_PID"

    sleep 3   # wait for backend to be ready

    echo "🎨  Starting frontend on http://localhost:8501..."
    echo ""
    echo "    Press Ctrl+C to stop both."
    echo ""
    streamlit run frontend/app.py --server.port 8501

    # Cleanup backend when frontend exits
    kill $BACKEND_PID 2>/dev/null
    ;;

  4)
    echo ""
    echo "🔌  Starting MCP server..."
    python mcp_service/mcp_server.py
    ;;

  5)
    echo "Bye!"
    exit 0
    ;;

  *)
    echo "Invalid choice."
    exit 1
    ;;
esac
