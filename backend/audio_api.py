from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import torch
import librosa
from pathlib import Path
from migration_predict_api import run_migration_for_species

router = APIRouter()

BACKEND_DIR = Path(__file__).parent
MODELS_DIR = BACKEND_DIR / "models"

AUDIO_MODEL_PATH = MODELS_DIR / "species_audio_model.pt"
SPECIES_LABELS_PATH = MODELS_DIR / "species_labels.npy"

audio_model = None
species_labels = None
species_index_to_name = None

def decode_species_name_audio(predicted_idx: int) -> str:
    """Decode predicted index to species name for audio model."""
    global species_index_to_name, species_labels
    
    # Try dictionary first
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
    
    return str(predicted_idx)

def load_audio_model():
    global audio_model, species_labels, species_index_to_name
    if audio_model is None:
        if not AUDIO_MODEL_PATH.exists():
            raise FileNotFoundError(f"Audio model not found: {AUDIO_MODEL_PATH}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            audio_model = torch.jit.load(str(AUDIO_MODEL_PATH), map_location=device)
        except Exception:
            audio_model = torch.load(str(AUDIO_MODEL_PATH), map_location=device, weights_only=False)
        if hasattr(audio_model, 'eval'):
            audio_model.eval()
        if hasattr(audio_model, 'to'):
            audio_model.to(device)
        if SPECIES_LABELS_PATH.exists():
            loaded_data = np.load(str(SPECIES_LABELS_PATH), allow_pickle=True)
            if isinstance(loaded_data, np.ndarray) and loaded_data.ndim == 0:
                loaded_data = loaded_data.item()
            if isinstance(loaded_data, dict):
                species_index_to_name = loaded_data
                max_idx = max(loaded_data.keys()) if loaded_data else 0
                species_labels = np.array([loaded_data.get(i, f"Unknown_{i}") for i in range(max_idx + 1)])
            else:
                species_labels = loaded_data

def extract_mfcc(file_bytes: bytes, sample_rate: int = 22050, n_mfcc: int = 40, duration: float = 5.0):
    import io
    import soundfile as sf
    # Load audio from bytes
    with io.BytesIO(file_bytes) as bio:
        y, sr = sf.read(bio, dtype='float32', always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate
    # Trim or pad to fixed duration
    target_len = int(duration * sr)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Normalize per-feature
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
    # Add batch/channel dims as [1, C, T] or [1, T, C] depending on model; assume [1, 1, n_mfcc, time]
    mfcc_tensor = torch.from_numpy(mfcc).float().unsqueeze(0).unsqueeze(0)
    return mfcc_tensor

@router.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        try:
            load_audio_model()
        except FileNotFoundError as e:
            raise HTTPException(status_code=501, detail=str(e))

        contents = await file.read()
        features = extract_mfcc(contents)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features = features.to(device)

        with torch.no_grad():
            outputs = audio_model(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_idx = int(predicted_idx.item())
        confidence = float(confidence.item())

        # Use the decode function to get species name
        species_name = decode_species_name_audio(predicted_idx)
        species_name = species_name.strip().replace("'", "").replace('"', '')

        migration_info = run_migration_for_species(species_name)

        return JSONResponse({
            "species_name": species_name,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100.0, 2),
            **migration_info,
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in predict-audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Predict-audio error: {str(e)}")


