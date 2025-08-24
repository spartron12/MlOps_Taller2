from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import numpy as np
import pickle
import joblib
import os
import logging
import time
import threading
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Penguin Species Prediction API",
    description="API para predecir la especie de pinguinos con distintos modelos disponibles",
    version="3.0.0"
)

MODELS_DIR = "/models"
models = {}  # Diccionario global para todos los modelos cargados
RELOAD_INTERVAL = 5  # Segundos entre revisiones automáticas

# Mapeo de especies de iris
penguin_species_mapping = {1: "Adelei", 2: "Chinstrap", 3: "Gentoo"}

# ===========================
# Función para cargar modelos
# ===========================
def discover_and_load_models():
    """Descubre automáticamente todos los archivos .pkl en el directorio y los carga"""
    global models
    current_models = set(models.keys())

    if not os.path.exists(MODELS_DIR):
        logger.warning(f"El directorio {MODELS_DIR} no existe")
        return

    pkl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]

    for f in pkl_files:
        name = os.path.splitext(f)[0]
        if name in current_models:
            continue  # Ya cargado
        path = os.path.join(MODELS_DIR, f)
        try:
            # Intentar cargar con joblib primero
            try:
                model_obj = joblib.load(path)
                logger.info(f"✅ Modelo {name} cargado con joblib desde {path}")
            except Exception:
                # Si falla, intentar con pickle
                with open(path, "rb") as file:
                    model_obj = pickle.load(file)
                    logger.info(f"✅ Modelo {name} cargado con pickle desde {path}")
            models[name] = {"model": model_obj}
        except Exception as e:
            logger.error(f"❌ Error cargando modelo {name}: {e}")

# ===========================
# Hilo para recarga automática
# ===========================
def auto_reload_models():
    """Hilo que recarga modelos automáticamente cada RELOAD_INTERVAL segundos."""
    while True:
        try:
            discover_and_load_models()
        except Exception as e:
            logger.error(f"Error en recarga automática de modelos: {e}")
        time.sleep(RELOAD_INTERVAL)

# ===========================
# Startup: cargar modelos y lanzar hilo
# ===========================
@app.on_event("startup")
async def startup_event():
    discover_and_load_models()
    thread = threading.Thread(target=auto_reload_models, daemon=True)
    thread.start()
    logger.info("Hilo de recarga automática iniciado")

# ===========================
# Esquema de entrada
# ===========================
class PenguinFeatures(BaseModel):
    bill_length_mm: float = Field(..., example=39.1)         # Largo del pico (mm) — requerido.
    bill_depth_mm: float = Field(..., example=18.7)          # Profundidad del pico (mm) — requerido.
    flipper_length_mm: float = Field(..., example=181.0)     # Largo de las aletas (mm) — requerido.
    body_mass_g: float = Field(..., example=3750.0)          # Masa corporal (g) — requerido.
    year: int = Field(..., example=2007)                     # Año de observación — requerido (se usó en entrenamiento).
    # Variables dummy para sex y island — se esperan 0/1 tal como se generaron en el preprocesado.
    sex_Female: int = Field(..., example=0)                  # Dummy sex=Female (0/1).
    sex_Male: int = Field(..., example=1)                    # Dummy sex=Male (0/1).
    island_Biscoe: int = Field(..., example=0)               # Dummy isla Biscoe (0/1).
    island_Dream: int = Field(..., example=0)                # Dummy isla Dream (0/1).
    island_Torgersen: int = Field(..., example=1)            # Dummy isla Torgersen (0/1).

# ===========================
# Validar modelo
# ===========================
def validate_model_name(model_name: str = Query(..., description="Selecciona el modelo a usar")):
    if not models:
        raise HTTPException(
            status_code=500, 
            detail="No hay modelos cargados aún. Intenta de nuevo en unos segundos."
        )
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo '{model_name}' no disponible. Modelos actuales: {list(models.keys())}"
        )
    return model_name

# ===========================
# Listar modelos
# ===========================
@app.get("/models")
def list_models():
    return {
        "available_models": list(models.keys()),
        "models_directory": MODELS_DIR,
        "total_models": len(models)
    }

# ===========================
# Predicción
# ===========================
@app.post("/predict")
def predict(features: PenguinFeatures, model_name: str = Depends(validate_model_name)):
    model = models[model_name]["model"]
    try:
        x = np.array([[features.bill_length_mm, features. bill_depth_mm, features.flipper_length_mm, features.body_mass_g,features.year,
                       features.sex_Female, features.sex_Male, features.island_Biscoe, features.island_Dream, features.island_Torgersen]])
        prediction = model.predict(x)[0]
        try:
            probabilities = model.predict_proba(x)[0]
            prob_dict = {str(cls): float(prob) for cls, prob in zip(model.classes_, probabilities)}
        except AttributeError:
            prob_dict = {"info": "Este modelo no proporciona probabilidades"}
        species_name = penguin_species_mapping.get(int(prediction), str(prediction)) if isinstance(prediction, (int, np.integer)) else str(prediction)
        return {
            "model_used": model_name,
            "model_type": type(model).__name__,
            "prediction": str(prediction),
            "species_name": species_name,
            "probabilities": prob_dict,
            "input_features": features.dict()
        }
    except Exception as e:
        logger.error(f"Error en predicción con modelo {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# ===========================
# Recarga manual de modelos
# ===========================
@app.post("/reload-models")
def reload_models():
    discover_and_load_models()
    return {
        "status": "success",
        "loaded_models": list(models.keys()),
        "total_loaded": len(models)
    }

# ===========================
# Debug
# ===========================
@app.get("/debug/directory")
def debug_directory():
    try:
        files = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
        pkl_files = [f for f in files if f.endswith(".pkl")]
        return {
            "directory": MODELS_DIR,
            "exists": os.path.exists(MODELS_DIR),
            "all_files": files,
            "pkl_files": pkl_files,
            "loaded_models": list(models.keys())
        }
    except Exception as e:
        return {"error": str(e)}