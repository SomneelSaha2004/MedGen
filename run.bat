@echo off
:: =============================================================================
:: MedGen Startup Script for Windows
:: =============================================================================
:: This script starts both the backend API server and the frontend React app
:: =============================================================================

echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║                    MedGen Startup                              ║
echo ║     AI-Powered Synthetic Medical Data Generation Platform     ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

:: Check for .env file
if not exist ".env" (
    echo [WARNING] No .env file found.
    if exist ".env.example" (
        copy .env.example .env
        echo [INFO] Created .env from template. Please edit and add your OPENAI_API_KEY
        pause
        exit /b 1
    ) else (
        echo [ERROR] No .env.example found. Please create .env with OPENAI_API_KEY
        pause
        exit /b 1
    )
)

:: Create necessary directories
echo [INFO] Creating data directories...
if not exist "data\features" mkdir data\features
if not exist "data\chroma_db" mkdir data\chroma_db
if not exist "data\generated" mkdir data\generated
if not exist "results" mkdir results

:: Start backend server
echo [INFO] Starting backend server on http://localhost:5000
start "MedGen Backend" cmd /k "uv run python backend.py"

:: Wait for backend to start
echo [INFO] Waiting for backend to initialize...
timeout /t 3 /nobreak > nul

:: Start frontend
if exist "frontend" (
    echo [INFO] Starting frontend on http://localhost:3000
    cd frontend
    if not exist "node_modules" (
        echo [INFO] Installing frontend dependencies...
        call npm install
    )
    start "MedGen Frontend" cmd /k "npm start"
    cd ..
)

echo.
echo ════════════════════════════════════════════════════════════════
echo   MedGen is starting!
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:5000
echo   Health:   http://localhost:5000/health
echo.
echo   Close the terminal windows to stop services
echo ════════════════════════════════════════════════════════════════
echo.
pause
