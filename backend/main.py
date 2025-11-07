# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from classify_api import router as classify_router
# from migration_predict_api import router as migration_router
# try:
#     from audio_api import router as audio_router
# except Exception:
#     audio_router = None

# app = FastAPI(title="Bird Species + Migration Prediction System")

# # Allow frontend requests
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include both routes
# app.include_router(classify_router, prefix="/api")
# app.include_router(migration_router, prefix="/api")
# if audio_router is not None:
#     app.include_router(audio_router, prefix="/api")

# @app.get("/")
# def home():
#     return {
#         "message": "Bird Species Classification & Migration Prediction API is running",
#         "endpoints": {
#             "classify": "/api/classify",
#             "predict_image": "/api/predict-image",
#             "migration": "/api/predict-migration",
#             "predict_audio": "/api/predict-audio" if audio_router is not None else None,
#             "health": "/api/health"
#         }
#     }


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from classify_api import router as classify_router
from migration_predict_api import router as migration_router

# 👇 Add this import if you created a separate file for migration endpoints
try:
    from routes import migration_routes
except ImportError:
    migration_routes = None

# Optional audio route
try:
    from audio_api import router as audio_router
except Exception:
    audio_router = None

app = FastAPI(title="Bird Species + Migration Prediction System")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routers
app.include_router(classify_router, prefix="/api")
app.include_router(migration_router, prefix="/api")

# ✅ Add this line — include new migration_routes if available
if migration_routes is not None:
    app.include_router(migration_routes.router, prefix="/api")

if audio_router is not None:
    app.include_router(audio_router, prefix="/api")

@app.get("/")
def home():
    return {
        "message": "Bird Species Classification & Migration Prediction API is running",
        "endpoints": {
            "classify": "/api/classify",
            "predict_image": "/api/predict-image",
            "migration": "/api/predict-migration",
            "predict_audio": "/api/predict-audio" if audio_router is not None else None,
            "health": "/api/health"
        }
    }

