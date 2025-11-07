@echo off
echo Starting Bird Species Classification Backend Server...
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Please create one first:
    echo python -m venv venv
    echo venv\Scripts\activate
    echo pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if requirements are installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting server on http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.

uvicorn main:app --reload --host 127.0.0.1 --port 8000

pause

