from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from pathlib import Path
from tensorflow import keras
import pandas as pd
import csv
from datetime import datetime
from collections import defaultdict, Counter

router = APIRouter()

class SpeciesRequest(BaseModel):
    species_name: str

# Get the backend directory path (where this file is located)
BACKEND_DIR = Path(__file__).parent
MODELS_DIR = BACKEND_DIR / "models"
BIRDS_MOVEMENT_CSV = BACKEND_DIR / "updated_bird_movement.csv"

# Model paths - use absolute paths
MIGRATION_MODEL_PATH = MODELS_DIR / "bird_movement_all_species_1.keras"
SCALER_MIN_PATH = MODELS_DIR / "bird_scaler_min.npy"
SCALER_SCALE_PATH = MODELS_DIR / "bird_scaler_scale.npy"
SPECIES_LABELS_PATH = MODELS_DIR / "species_labels.npy"
SPECIES_CLASSES_PATH = MODELS_DIR / "species_label_classes.npy"

# Global variables
migration_model = None
scaler_min = None
scaler_scale = None
species_labels = None
movement_data = None  # dict[str, list[tuple[int,int,str]]]
movement_df = None
coord_scaler = None
label_classes = None
SEQ_LENGTH = 5

def _normalize_common_name(name: str) -> str:
    """Simple normalization: just convert to lowercase."""
    if not isinstance(name, str):
        return ""
    return name.lower().strip()
    
   

def load_migration_model():
    """Load the migration prediction model and scalers."""
    global migration_model, scaler_min, scaler_scale, species_labels
    
    if migration_model is None:
        try:
            # Load migration model
            if not MIGRATION_MODEL_PATH.exists():
                raise FileNotFoundError(f"Migration model file not found: {MIGRATION_MODEL_PATH}")
            
            print(f"Loading migration model from: {MIGRATION_MODEL_PATH}")
            migration_model = keras.models.load_model(str(MIGRATION_MODEL_PATH))
            
            # Load scalers
            if SCALER_MIN_PATH.exists():
                scaler_min = np.load(str(SCALER_MIN_PATH), allow_pickle=True)
                print(f"Loaded scaler_min from: {SCALER_MIN_PATH}")
            if SCALER_SCALE_PATH.exists():
                scaler_scale = np.load(str(SCALER_SCALE_PATH), allow_pickle=True)
                print(f"Loaded scaler_scale from: {SCALER_SCALE_PATH}")
            
            # Load species labels for mapping
            if SPECIES_LABELS_PATH.exists():
                species_labels = np.load(str(SPECIES_LABELS_PATH), allow_pickle=True)
                print(f"Loaded species labels from: {SPECIES_LABELS_PATH}")
            
            print("Migration model loaded successfully")
            
        except Exception as e:
            print(f"Error loading migration model: {str(e)}")
            raise

    # Lazily load movement CSV as well
    load_movement_data()
    load_movement_dataframe()
    load_label_classes()

def load_label_classes():
    global label_classes
    if label_classes is not None:
        return
    try:
        if SPECIES_CLASSES_PATH.exists():
            label_classes = np.load(str(SPECIES_CLASSES_PATH), allow_pickle=True)
    except Exception as e:
        print(f"Warning: failed to load species classes: {e}")

def load_movement_data():
    """Load bird movement CSV into memory: species -> [(year, month, country), ...].
    Uses common_name column for matching (falls back to Scientific Name if common_name missing).
    """
    global movement_data
    if movement_data is not None:
        return
    movement_data = defaultdict(list)
    try:
        if not BIRDS_MOVEMENT_CSV.exists():
            return
        with open(BIRDS_MOVEMENT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Normalize header names to lower-case for case-insensitive access
            headers_lower = [h.lower().strip() for h in (reader.fieldnames or [])]
            # Prefer common_name/"common name"/fall back to scientific_name
            for row in reader:
                # Lower-case keys for robust access
                r = {k.lower().strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                raw_name = r.get("common_name") or r.get("common name") or r.get("scientific_name") or r.get("scientific name") or ""
                name = _normalize_common_name(raw_name)
                country = r.get("country") or r.get("Country") or ""
                try:
                    year = int(r.get("year") or 0)
                except Exception:
                    year = 0
                try:
                    month = int(r.get("month") or 0)
                except Exception:
                    month = 0
                if isinstance(name, str) and isinstance(country, str) and name and country and 1 <= month <= 12:
                    movement_data[name.lower()].append((year, month, country))
        print(f"Loaded movement data for {len(movement_data)} species (using common names)")
    except Exception as e:
        print(f"Warning: failed to load movement CSV: {e}")

def load_movement_dataframe():
    """Load CSV with pandas and fit MinMax scaler on latitude/longitude.
    Normalizes column names to lower-case and creates a 'common_name' column
    (from 'common_name'/'common name' or fallback to 'scientific_name').
    """
    global movement_df, coord_scaler
    if movement_df is not None and coord_scaler is not None:
        return
    if not BIRDS_MOVEMENT_CSV.exists():
        return
    try:
        df = pd.read_csv(BIRDS_MOVEMENT_CSV)
        # Normalize columns to lower-case for case-insensitive access
        df.columns = [c.strip().lower() for c in df.columns]
        if "Date" in df.columns:
            try:
                df["Date"] = pd.to_datetime(df["Date"])
            except Exception:
                pass
        # Ensure required columns exist (lower-case)
        required_cols = ["latitude", "longitude"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        # Derive common_name column and normalize values
        if "common_name" in df.columns:
            df["common_name"] = df["common_name"].fillna("").astype(str).map(_normalize_common_name)
        elif "common name" in df.columns:
            df["common_name"] = df["common name"].fillna("").astype(str).map(_normalize_common_name)
        elif "scientific_name" in df.columns:
            df["common_name"] = df["scientific_name"].fillna("").astype(str).map(_normalize_common_name)
        else:
            raise ValueError("CSV must have either 'common_name'/'common name' or 'scientific_name' column")
        
        from sklearn.preprocessing import MinMaxScaler
        coord_scaler = MinMaxScaler()
        coord_scaler.fit(df[["latitude", "longitude"]])
        movement_df = df
        print(f"Loaded movement dataframe with {len(df)} records, {df['common_name'].nunique()} unique species")
    except Exception as e:
        print(f"Warning: failed loading movement dataframe: {e}")

def predict_next_location_with_lstm(species_name: str):
    """Predict next (lat, lon) using LSTM model given last SEQ_LENGTH steps of the species.
    Returns (lat, lon) or (None, None) on failure.
    Matches by common_name first, then falls back to Scientific Name.
    """
    if migration_model is None:
        load_migration_model()
    if migration_model is None or movement_df is None or coord_scaler is None:
        return (None, None)
    try:
        # Try to match by common_name first
        if "common_name" in movement_df.columns:
            species_df = movement_df[movement_df["common_name"].str.lower() == species_name.lower()].copy()
        else:
            species_df = pd.DataFrame()  # Empty if no common_name column
        
        # Fallback to Scientific Name if common_name match failed
        if species_df.empty and "scientific_name" in movement_df.columns:
            species_df = movement_df[movement_df["scientific_name"].str.lower() == species_name.lower()].copy()
        
        if species_df.empty or len(species_df) < SEQ_LENGTH:
            return (None, None)
        # Sort by date if present
        if "date" in species_df.columns:
            try:
                species_df = species_df.sort_values("date")
            except Exception:
                pass
        scaled = coord_scaler.transform(species_df[["latitude", "longitude"]])
        last_seq = scaled[-SEQ_LENGTH:]
        input_seq = np.expand_dims(last_seq, axis=0)

        # species id via label_classes if present, else via SPECIES_LABELS_PATH mapping
        species_id = 0
        try:
            if label_classes is not None:
                # classes_ array: find index
                classes_lower = [str(x).lower() for x in label_classes]
                species_id = int(classes_lower.index(species_name.lower()))
            elif species_labels is not None:
                classes_lower = [str(x).lower() for x in np.array(species_labels).ravel().tolist()]
                species_id = int(classes_lower.index(species_name.lower()))
        except Exception:
            species_id = 0
        species_input = np.array([[species_id]])

        # Predict
        pred_scaled = migration_model.predict([input_seq, species_input], verbose=0)
        pred_actual = coord_scaler.inverse_transform(np.array(pred_scaled).reshape(1, -1))
        lat, lon = float(pred_actual[0][0]), float(pred_actual[0][1])
        return (lat, lon)
    except Exception as e:
        print(f"Warning: LSTM next-location prediction failed: {e}")
        return (None, None)

def infer_next_region_month(species_name: str, ref_month: int | None = None):
    """Infer next region and month from movement_data using simple month+1 heuristic.
    - ref_month: defaults to current month.
    - If no exact month+1 data, fall back to most common country overall.
    - Matches by common_name with fuzzy matching fallback.
    Returns (next_region, next_month_name, is_migratory_bool).
    """
    load_movement_data()
    if not species_name:
        return ("Unknown", "Unknown", False)
    
    # Normalize species name for matching
    species_name_lower = _normalize_common_name(species_name)
    
    # Try exact match first
    data = movement_data.get(species_name_lower, [])
    
    # If no exact match, try fuzzy matching
    if not data:
        # 1) substring containment heuristic
        for key, value in movement_data.items():
            if species_name_lower in key or key in species_name_lower:
                if abs(len(species_name_lower) - len(key)) <= 10:
                    data = value
                    print(f"Fuzzy matched '{species_name}' to '{key}' (substring)")
                    break

    # 2) token-overlap (Jaccard) heuristic for generic/common-group names
    if not data:
        def jaccard(a: str, b: str) -> float:
            at = {t for t in a.replace('-', ' ').split() if t}
            bt = {t for t in b.replace('-', ' ').split() if t}
            if not at or not bt:
                return 0.0
            inter = len(at & bt)
            union = len(at | bt)
            return inter / union if union else 0.0

        best_key = None
        best_score = 0.0
        for key, value in movement_data.items():
            score = jaccard(species_name_lower, key)
            if score > best_score:
                best_score = score
                best_key = key
        # Require a reasonable overlap to avoid random matches
        if best_key is not None and best_score >= 0.5:
            data = movement_data[best_key]
            print(f"Fuzzy matched '{species_name}' to '{best_key}' (token overlap {best_score:.2f})")
    
    if not data:
        return ("Unknown", "Unknown", False)

    # Determine migratory by counting unique countries
    unique_countries = {c for (_, _, c) in data}
    is_migratory = len(unique_countries) > 1

    # Reference month
    if ref_month is None:
        ref_month = datetime.now().month
    next_month_num = 1 if ref_month == 12 else ref_month + 1

    # Countries in next month across all years
    countries_next_month = [c for (_, m, c) in data if m == next_month_num]
    if countries_next_month:
        next_country = Counter(countries_next_month).most_common(1)[0][0]
    else:
        # Fallback: most common country overall
        next_country = Counter([c for (_, _, c) in data]).most_common(1)[0][0]

    month_names = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    next_month_name = month_names[next_month_num - 1]
    return (next_country, next_month_name, is_migratory)

def get_species_index(species_name):
    """Get the index of a species from the labels."""
    global species_labels
    
    # Load labels if not already loaded
    if species_labels is None:
        if migration_model is None:
            load_migration_model()
        else:
            # Model loaded but labels not loaded yet
            if SPECIES_LABELS_PATH.exists():
                species_labels = np.load(str(SPECIES_LABELS_PATH), allow_pickle=True)
    
    if species_labels is None:
        return None
    
    try:
        # Try to find the species in labels
        species_name_lower = str(species_name).lower().strip()
        
        for idx, label in enumerate(species_labels):
            label_str = str(label).lower().strip()
            if species_name_lower in label_str or label_str in species_name_lower:
                return idx
        
        # If not found, return None
        return None
    except Exception as e:
        print(f"Error finding species index: {str(e)}")
        return None
    


# ...existing code...
# def run_migration_for_species(species_name: str) -> dict:
#     """Utility to compute migration info for a species and return spec-compliant fields.
#     Returns keys: is_migratory (bool), next_region (str), next_month (str),
#     plus additional diagnostics (migration_prediction, migration_status, predicted_movement, migration_score, confidence).
#     """
#     # Ensure model and CSV are loaded
#     if migration_model is None:
#         try:
#             load_migration_model()
#         except Exception:
#             # load_migration_model may raise when model missing; proceed to CSV-only flow
#             pass

#     # Normalize for CSV matching
#     species_norm = species_name.strip().lower() if isinstance(species_name, str) else ""
#     print(f"🔍 Migration lookup for: '{species_name}' (norm: '{species_norm}')")

#     # Always attempt CSV heuristic first (works even if model/labels missing)
#     csv_region, csv_month, csv_migratory = infer_next_region_month(species_name)

#     # Default response fields
#     result = {
#         "is_migratory": bool(csv_migratory),
#         "next_region": csv_region or "Unknown",
#         "next_month": csv_month or "Unknown",
#         "predicted_lat": None,
#         "predicted_lon": None,
#         "migration_prediction": "Prediction unavailable",
#         "migration_status": "unknown",
#         "predicted_movement": "No data",
#         "migration_score": 0.0,
#         "confidence": 0.0,
#     }

#     # If model not loaded, return CSV-only result

#     if migration_model is None:
#         result.update({
#             "migration_prediction": "Model unavailable - CSV heuristic used",
#             "migration_status": "csv_based",
#         })
#         return result
    


def run_migration_for_species(species_name: str) -> dict:
    """
    Compute migration info for a species and return spec-compliant fields.
    Returns keys: is_migratory, next_region, next_month, predicted_lat, predicted_lon, etc.
    """
    # Ensure model and CSV are loaded
    if migration_model is None:
        try:
            load_migration_model()
        except Exception:
            print("⚠️ Could not load migration model.")
            pass

    species_norm = species_name.strip().lower() if isinstance(species_name, str) else ""
    print(f"🔍 Migration lookup for: '{species_name}' (norm: '{species_norm}')")

    # Step 1️⃣ - CSV heuristic for region and month
    csv_region, csv_month, csv_migratory = infer_next_region_month(species_name)

    # Step 2️⃣ - Model prediction for next lat/lon
    predicted_lat, predicted_lon = predict_next_location_with_lstm(species_name)

    # Build response
    result = {
        "is_migratory": bool(csv_migratory),
        "next_region": csv_region or "Unknown",
        "next_month": csv_month or "Unknown",
        "predicted_lat": predicted_lat,
        "predicted_lon": predicted_lon,
        "migration_prediction": "Prediction available" if predicted_lat and predicted_lon else "Unavailable",
        "migration_status": "model_based" if predicted_lat and predicted_lon else "csv_based",
        "predicted_movement": f"Next coordinates: ({predicted_lat}, {predicted_lon})" if predicted_lat and predicted_lon else "No movement predicted",
        "migration_score": 1.0 if predicted_lat and predicted_lon else 0.0,
        "confidence": 0.95 if predicted_lat and predicted_lon else 0.0,
    }

    # Step 3️⃣ - Comment out fallback model-unavailable block
    # if migration_model is None:
    #     result.update({
    #         "migration_prediction": "Model unavailable - CSV heuristic used",
    #         "migration_status": "csv_based",
    #     })
    #     return result

    print(f"✅ Migration result: {result}")
    return result




    # Try to get species index; if missing we'll still continue with CSV
    species_idx = get_species_index(species_name)
    print(f"  species_idx from labels: {species_idx}")

    # Prepare features for model if possible
    input_features = np.array([[species_idx if species_idx is not None else 0]])
    try:
        if scaler_min is not None and scaler_scale is not None:
            scaler_min_arr = np.array(scaler_min)
            scaler_scale_arr = np.array(scaler_scale)
            if scaler_min_arr.shape != input_features.shape and scaler_min_arr.ndim == 1 and len(scaler_min_arr) > 0:
                scaler_min_arr = scaler_min_arr[0]
            if scaler_scale_arr.shape != input_features.shape and scaler_scale_arr.ndim == 1 and len(scaler_scale_arr) > 0:
                scaler_scale_arr = scaler_scale_arr[0]
            input_features = (input_features - scaler_min_arr) / (scaler_scale_arr + 1e-8)
    except Exception:
        pass

    try:
        expected_shape = getattr(migration_model, 'input_shape', None)
        if expected_shape and len(expected_shape) > 2:
            input_features = np.reshape(input_features, (1, 1, -1))
    except Exception:
        pass

    # Run model prediction (guarded)
    try:
        prediction = migration_model.predict(input_features, verbose=0)
        prediction_value = float(np.array(prediction).flatten()[0])
    except Exception as e:
        print(f"Warning: model prediction failed: {e}")
        prediction_value = 0.0

    # Interpret model score
    if prediction_value > 0.7:
        migration_status = "High Migration Activity"
        predicted_movement = "Likely migrating soon"
    elif prediction_value > 0.4:
        migration_status = "Moderate Migration Activity"
        predicted_movement = "Possible migration in near future"
    elif prediction_value > 0.1:
        migration_status = "Low Migration Activity"
        predicted_movement = "Resident or low migration"
    else:
        migration_status = "No Migration Expected"
        predicted_movement = "Resident species"

    # LSTM geo prediction (optional) — provide lat/lon if possible
    pred_lat, pred_lon = predict_next_location_with_lstm(species_name)
    if pred_lat is not None and pred_lon is not None:
        result["predicted_lat"] = float(pred_lat)
        result["predicted_lon"] = float(pred_lon)
        # Attempt mapping predicted lat/lon to country using movement_df (best-effort)
        try:
            if movement_df is not None:
                if "common_name" in movement_df.columns:
                    sp = movement_df[movement_df["common_name"].str.lower() == species_norm].copy()
                else:
                    sp = pd.DataFrame()
                if sp.empty and "scientific_name" in movement_df.columns:
                    sp = movement_df[movement_df["scientific_name"].str.lower() == species_norm].copy()
                if not sp.empty:
                    country_col = "country" if "country" in sp.columns else sp.columns[0]
                    lat_col = "latitude" if "latitude" in sp.columns else sp.columns[1]
                    lon_col = "longitude" if "longitude" in sp.columns else sp.columns[2]
                    med_by_country = sp.groupby(country_col)[[lat_col, lon_col]].median().reset_index()
                    med_by_country["dist"] = (med_by_country[lat_col] - pred_lat) ** 2 + (med_by_country[lon_col] - pred_lon) ** 2
                    best = med_by_country.sort_values("dist").iloc[0]
                    csv_region = str(best[country_col])
        except Exception:
            pass

    # Final decision combining model + CSV
    is_migratory = (prediction_value > 0.4) or csv_migratory
    result.update({
        "is_migratory": bool(is_migratory),
        # "next_region": csv_region or result["next_region"],
        # "next_month": csv_month or result["next_month"],
        "next_region":  result["next_region"],
        "next_month":   result["next_month"],
        "migration_prediction": migration_status,
        "migration_status": migration_status.lower().replace(" ", "_"),
        "predicted_movement": predicted_movement,
        "migration_score": round(prediction_value, 4),
        "confidence": round(min(abs(prediction_value) * 100, 100), 2),
    })
    return result





@router.post("/predict-migration")
async def predict_migration(request: SpeciesRequest):
    """
    Predict migration pattern for a given bird species.
    
    Args:
        request: JSON body containing species_name
    
    Returns:
        Migration prediction data (JSON)
    """
    try:
        species_name = request.species_name.strip()
        if not species_name:
            raise HTTPException(status_code=400, detail="Species name is required")

        # Ensure the model and related assets are loaded
        if migration_model is None:
            load_migration_model()

        # Run migration logic for the species
        result = run_migration_for_species(species_name)

        if not result:
            raise HTTPException(status_code=500, detail="Migration prediction failed")

        # Format response
        return JSONResponse(content={
            "species_name": species_name,
            "is_migratory": result.get("is_migratory", False),
            "next_region": result.get("next_region", "Unknown"),
            "next_month": result.get("next_month", "Unknown"),
            "migration_prediction": result.get("migration_prediction", "Unavailable"),
            "migration_status": result.get("migration_status", "unknown"),
            "predicted_movement": result.get("predicted_movement", "Unavailable"),
            "migration_score": result.get("migration_score", 0.0),
            "confidence": result.get("confidence", 0.0),
            "predicted_lat": result.get("predicted_lat"),
            "predicted_lon": result.get("predicted_lon")
        })

    except HTTPException as e:
        # Pass FastAPI-raised exceptions through
        raise e
    except Exception as e:
        print(f"Error in predict_migration: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")




@router.get("/migration-health")
async def migration_health_check():
    """Health check endpoint for migration model."""
    return {
        "status": "ok",
        "model_loaded": migration_model is not None,
        "scaler_loaded": scaler_min is not None and scaler_scale is not None
    }

