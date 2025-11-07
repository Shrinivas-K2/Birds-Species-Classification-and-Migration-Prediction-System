#!/bin/bash

echo "Starting Bird Species Classification Backend Server..."
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Please create one first:"
    echo "python3 -m venv venv"
    echo "source venv/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Check if requirements are installed
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "Starting server on http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn main:app --reload --host 127.0.0.1 --port 8000


