# Bird Species Classification & Migration Prediction Backend

This is the backend API for the Bird Species Classification System. It provides endpoints for classifying bird species from images and predicting their migration patterns.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

### Step 1: Navigate to Backend Directory
```bash
cd backend
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with PyTorch or TensorFlow installation, you may need to install them separately:

#### For PyTorch (CPU version):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### For PyTorch (GPU version - if you have CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### For TensorFlow:
```bash
pip install tensorflow
```

## Starting the Server

### Option 1: Using Uvicorn (Recommended)
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Option 2: Using Python
```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The server will start on `http://127.0.0.1:8000`

## API Endpoints

- `GET /` - Health check and API information
- `POST /api/classify` - Classify bird species from uploaded image
- `POST /api/predict-migration` - Predict migration pattern for a bird species
- `GET /api/health` - Classification model health check
- `GET /api/migration-health` - Migration model health check

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Model Files Required

Make sure these files exist in the `backend/models/` directory:
- `efficientnetb0_birds_final (1).pt` - Bird classification model
- `species_labels.npy` or `species_label_classes.npy` - Species labels
- `bird_movement_all_species_1.keras` - Migration prediction model
- `bird_scaler_min.npy` - Scaler for migration model (optional)
- `bird_scaler_scale.npy` - Scaler for migration model (optional)

## Troubleshooting

### Port Already in Use
If port 8000 is already in use, change it:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

### Model Loading Issues
- Check that all model files exist in `backend/models/`
- Verify file paths in the console output
- Ensure you have enough RAM/VRAM for model loading

### Import Errors
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### CORS Issues
CORS is already configured to allow all origins. If you need to restrict it, edit `main.py`.

