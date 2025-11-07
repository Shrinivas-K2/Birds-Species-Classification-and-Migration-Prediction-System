from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import numpy as np
from pathlib import Path
from utils.Preproces import preprocess_image
from migration_predict_api import run_migration_for_species


def _prepare_labels(raw_labels):
    """Normalize labels loaded from numpy into a 1-D numpy array of strings.
    Handles cases where .npy contains a numpy scalar object or nested structures.
    """
    labels = raw_labels
    # If it's a numpy scalar that wraps an object/list
    if isinstance(labels, np.ndarray) and labels.ndim == 0:
        labels = labels.item()

    # If it's a list/tuple, convert to ndarray
    if isinstance(labels, (list, tuple)):
        labels = np.array(labels)

    # If still ndarray but not 1-D, flatten
    if isinstance(labels, np.ndarray):
        if labels.ndim == 0:
            # After item() above, this should not happen, but guard just in case
            labels = np.array([str(labels.item())])
        elif labels.ndim > 1:
            labels = labels.ravel()

        # Ensure dtype is str for safe indexing/printing
        try:
            labels = labels.astype(str)
        except Exception:
            labels = np.array([str(x) for x in labels.tolist()])

    # If it's something else (e.g., set), coerce to list then ndarray
    if not isinstance(labels, np.ndarray):
        try:
            labels = np.array(list(labels))
        except Exception:
            # Fallback to single unknown label
            labels = np.array(["unknown"])

    return labels

router = APIRouter()

# Get the backend directory path (where this file is located)
BACKEND_DIR = Path(__file__).parent
MODELS_DIR = BACKEND_DIR / "models"

# Load model and labels - use absolute paths
MODEL_PATH = MODELS_DIR / "efficientnetb0_birds_final (1).pt"
LABELS_PATH = MODELS_DIR / "species_labels.npy"
LABEL_CLASSES_PATH = MODELS_DIR / "species_label_classes.npy"

# Global variables for model and labels
model = None
species_labels = None
label_classes = None
species_index_to_name = None  # Dictionary mapping index -> species name

def decode_species_name(predicted_idx: int) -> str:
    """Decode predicted index to species name using available label mappings."""
    global species_index_to_name, species_labels, label_classes
    
    # Try dictionary first (from .npy file with dict format)
    if species_index_to_name is not None:
        if predicted_idx in species_index_to_name:
            return str(species_index_to_name[predicted_idx])
    
    # Try array format
    if species_labels is not None:
        try:
            if isinstance(species_labels, np.ndarray) and species_labels.size > 0:
                if 0 <= predicted_idx < len(species_labels):
                    return str(species_labels[predicted_idx])
        except Exception:
            pass
    
    # Try label_classes
    if label_classes is not None:
        try:
            if isinstance(label_classes, np.ndarray) and label_classes.size > 0:
                if 0 <= predicted_idx < len(label_classes):
                    return str(label_classes[predicted_idx])
        except Exception:
            pass
    
    # Fallback to "Unknown Bird" instead of index
    return "Unknown Bird"

def load_model():
    """Load the EfficientNet model and labels."""
    global model, species_labels, label_classes, species_index_to_name
    
    if model is None:
        try:
            # Load labels - try dictionary format first
            if LABELS_PATH.exists():
                loaded_data = np.load(str(LABELS_PATH), allow_pickle=True)
                # Handle numpy scalar wrapping
                if isinstance(loaded_data, np.ndarray) and loaded_data.ndim == 0:
                    loaded_data = loaded_data.item()
                
                # Check if it's a dictionary (index -> name mapping)
                if isinstance(loaded_data, dict):
                    species_index_to_name = loaded_data
                    # Also create array for backward compatibility
                    max_idx = max(loaded_data.keys()) if loaded_data else 0
                    species_labels = np.array([loaded_data.get(i, f"Unknown_{i}") for i in range(max_idx + 1)])
                    print(f"Loaded dictionary labels from: {LABELS_PATH} ({len(species_index_to_name)} species)")
                else:
                    species_labels = _prepare_labels(loaded_data)
                    print(f"Loaded array labels from: {LABELS_PATH}")
            
            if label_classes is None and LABEL_CLASSES_PATH.exists():
                loaded_classes = np.load(str(LABEL_CLASSES_PATH), allow_pickle=True)
                if isinstance(loaded_classes, np.ndarray) and loaded_classes.ndim == 0:
                    loaded_classes = loaded_classes.item()
                if isinstance(loaded_classes, dict):
                    # Merge with existing dictionary if available
                    if species_index_to_name is None:
                        species_index_to_name = loaded_classes
                    else:
                        species_index_to_name.update(loaded_classes)
                    label_classes = np.array([loaded_classes.get(i, f"Unknown_{i}") for i in range(max(loaded_classes.keys()) + 1)])
                else:
                    label_classes = _prepare_labels(loaded_classes)
                print(f"Loaded label classes from: {LABEL_CLASSES_PATH}")
            
            if species_labels is None and label_classes is None and species_index_to_name is None:
                raise FileNotFoundError(f"Label files not found. Checked: {LABELS_PATH}, {LABEL_CLASSES_PATH}")
            
            # Load PyTorch model
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading model from: {MODEL_PATH}")
            # PyTorch 2.6 changed torch.load default weights_only=True which breaks TorchScript archives
            # Try torch.jit.load first (for TorchScript), fallback to torch.load(weights_only=False)
            try:
                model = torch.jit.load(str(MODEL_PATH), map_location=device)
            except Exception:
                model = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
            
            # Ensure model is in eval mode and on correct device
            if hasattr(model, 'eval'):
                model.eval()
            if hasattr(model, 'to'):
                model.to(device)
            
            print(f"Model loaded successfully on {device}")
            try:
                print(f"Number of species: {len(species_labels)}")
            except Exception:
                print("Number of species: unknown (labels format)")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

@router.post("/classify")
async def classify_bird(image: UploadFile = File(...)):
    """
    Classify bird species from uploaded image.
    Returns the predicted bird species name.
    """
    try:
        # Check if image is provided
        if not image:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Load model if not already loaded
        if model is None:
            load_model()
        
        # Read and preprocess image
        contents = await image.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess image - pass PIL Image directly
        input_tensor = preprocess_image(image_pil)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get predicted species name
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        
        # Use the decode function to get species name
        predicted_species = decode_species_name(predicted_idx)
        
        # Clean up species name (remove any extra formatting)
        predicted_species = predicted_species.strip().replace("'", "").replace('"', '')
        
        # Check for low confidence or suspicious labels (out-of-distribution images)
        CONFIDENCE_THRESHOLD = 0.3  # 30% - if below this, consider it unknown
        SUSPICIOUS_LABELS = ['loonely', 'looney', 'lonely', 'unknown', 'unkown', 'none', 'n/a', 'na']
        
        # Convert to lowercase for comparison
        predicted_species_lower = predicted_species.lower()
        
        # If confidence is too low OR label contains suspicious words, return "Unknown Bird"
        if confidence < CONFIDENCE_THRESHOLD or any(suspicious in predicted_species_lower for suspicious in SUSPICIOUS_LABELS):
            predicted_species = "Unknown Bird"
            confidence = confidence  # Keep original confidence but mark as unknown
        
        return JSONResponse({
            "predicted_species": predicted_species,
            "confidence": round(confidence * 100, 2),
            "predicted_index": int(predicted_idx)
        })
        
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@router.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """
    Spec-compliant endpoint: accepts 'file' image upload, classifies species,
    then augments with migration info and returns a combined JSON.
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        # Ensure model
        if model is None:
            load_model()

        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = preprocess_image(image_pil)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_idx = predicted_idx.item()
        confidence = float(confidence.item())  # probability (0..1)

        # Use the decode function to get species name
        species_name = decode_species_name(predicted_idx)

        # Clean up species name (remove any extra formatting)
        species_name = species_name.strip().replace("'", "").replace('"', '')

        # Check for low confidence or suspicious labels (out-of-distribution images)
        CONFIDENCE_THRESHOLD = 0.3  # 30% - if below this, consider it unknown
        SUSPICIOUS_LABELS = ['loonely', 'looney', 'lonely', 'unknown', 'unkown', 'none', 'n/a', 'na']
        
        # Convert to lowercase for comparison
        species_name_lower = species_name.lower()
        
        # If confidence is too low OR label contains suspicious words, return "Unknown Bird"
        if confidence < CONFIDENCE_THRESHOLD or any(suspicious in species_name_lower for suspicious in SUSPICIOUS_LABELS):
            species_name = "Unknown Bird"
            confidence = confidence  # Keep original confidence but mark as unknown

        migration_info = run_migration_for_species(species_name)

        return JSONResponse({
            "species_name": species_name,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100.0, 2),
            **migration_info,
        })

    except Exception as e:
        print(f"Error in predict-image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Predict-image error: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}

