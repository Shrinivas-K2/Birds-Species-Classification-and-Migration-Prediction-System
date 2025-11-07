# Quick Start Guide - Fix Connection Issues

## Problem: `ERR_CONNECTION_REFUSED`

This error means the **backend server is not running**. Follow these steps:

## Step 1: Install Missing Libraries

Based on your check, you need to install these missing libraries:

```bash
cd backend

# Make sure virtual environment is activated
# Windows:
venv\Scripts\activate

# Install all missing libraries at once
pip install fastapi python-multipart torchvision tensorflow pydantic
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

**Note:** For torchvision, you might need to install it separately:
```bash
pip install torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Step 2: Verify Installation

Run the check script again:
```bash
python check_installations.py
```

Make sure all libraries show ✅ before proceeding.

## Step 3: Start the Backend Server

### Option A: Using the startup script (Windows)
```bash
start_server.bat
```

### Option B: Manual start
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

You should see output like:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## Step 4: Verify Backend is Running

Open your browser and go to:
- http://127.0.0.1:8000/

You should see:
```json
{
  "message": "Bird Species Classification & Migration Prediction API is running",
  "endpoints": {
    "classify": "/api/classify",
    "migration": "/api/predict-migration",
    "health": "/api/health"
  }
}
```

## Step 5: Test the API

You can also check the API docs:
- http://127.0.0.1:8000/docs

## Troubleshooting

### If server won't start:

1. **Check for errors in the terminal** - Look for import errors or missing libraries
2. **Verify virtual environment is activated** - You should see `(venv)` in your prompt
3. **Check if port 8000 is already in use:**
   ```bash
   # Windows:
   netstat -ano | findstr :8000
   
   # If port is in use, change the port:
   uvicorn main:app --reload --host 127.0.0.1 --port 8001
   ```
   Then update frontend to use port 8001

### If libraries won't install:

1. **Update pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install libraries one by one:**
   ```bash
   pip install fastapi
   pip install python-multipart
   pip install torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install tensorflow
   pip install pydantic
   ```

3. **For TensorFlow issues**, try:
   ```bash
   pip install tensorflow --upgrade
   ```

## Important Notes

- **Keep the backend server running** - Don't close the terminal where the server is running
- **Frontend and Backend run separately** - You need both running:
  - Backend: `uvicorn main:app --reload` (in backend folder)
  - Frontend: `npm start` (in frontend folder)
- **Two terminal windows** - You'll need one for backend and one for frontend

## Complete Setup Command Sequence

```bash
# Terminal 1 - Backend
cd backend
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 - Frontend (open new terminal)
cd frontend
npm start
```


