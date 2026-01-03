#!/bin/bash
# =============================================================================
# MedGen Startup Script
# =============================================================================
# This script starts both the backend API server and the frontend React app
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    🏥 MedGen Startup                          ║"
echo "║     AI-Powered Synthetic Medical Data Generation Platform     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating from template...${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}📝 Please edit .env and add your OPENAI_API_KEY${NC}"
        exit 1
    else
        echo -e "${RED}❌ No .env.example found. Please create .env with OPENAI_API_KEY${NC}"
        exit 1
    fi
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating data directories...${NC}"
mkdir -p data/features data/chroma_db data/generated results

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}⚠️  uv not found. Installing...${NC}"
    pip install uv
fi

# Start backend server
echo -e "${GREEN}🚀 Starting backend server on http://localhost:5000${NC}"
uv run python backend.py &
BACKEND_PID=$!

# Wait for backend to start
echo -e "${BLUE}⏳ Waiting for backend to initialize...${NC}"
sleep 3

# Check if backend is running
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend is healthy${NC}"
else
    echo -e "${YELLOW}⚠️  Backend health check pending...${NC}"
fi

# Start frontend (optional - comment out if running separately)
if [ -d "frontend" ]; then
    echo -e "${GREEN}🎨 Starting frontend on http://localhost:3000${NC}"
    cd frontend
    if [ ! -d "node_modules" ]; then
        echo -e "${BLUE}📦 Installing frontend dependencies...${NC}"
        npm install
    fi
    npm start &
    FRONTEND_PID=$!
    cd ..
fi

echo -e "${GREEN}"
echo "════════════════════════════════════════════════════════════════"
echo "  MedGen is running!"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:5000"
echo "  Health:   http://localhost:5000/health"
echo ""
echo "  Press Ctrl+C to stop all services"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Trap Ctrl+C to kill both processes
trap "echo -e '\n${YELLOW}Shutting down...${NC}'; kill $BACKEND_PID 2>/dev/null; kill $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM

# Wait for processes
wait
