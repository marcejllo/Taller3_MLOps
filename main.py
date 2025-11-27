from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError, Field
from pandas import DataFrame
import joblib
import boto3
import json
from datetime import datetime
import numpy as np

S3_BUCKET = "taller3-mlops"         
S3_PREFIX = "student-predictions/"

try:
    s3_client = boto3.client("s3")
except Exception as e:
    print("⚠ Error inicializando S3:", e)
    s3_client = None

app = FastAPI(title="Student GPA Prediction API")


try:
    pipeline = joblib.load("pipeline_model/model.pkl")
    print("✓ Pipeline cargado")
except Exception as e:
    print("❌ Error cargando pipeline:", e)
    pipeline = None


# ======================================================
# Modelos de entrada/salida
# ======================================================

class StudentData(BaseModel):
    Age: int = Field(..., ge=15, le=22)
    Gender: int = Field(..., ge=0, le=1)
    Ethnicity: int = Field(..., ge=0, le=3)
    ParentalEducation: int = Field(..., ge=0, le=4)
    StudyTimeWeekly: float = Field(..., ge=0, le=40)
    Absences: int = Field(..., ge=0, le=30)
    Tutoring: int = Field(..., ge=0, le=1)
    ParentalSupport: int = Field(..., ge=0, le=4)
    Extracurricular: int = Field(..., ge=0, le=1)
    Sports: int = Field(..., ge=0, le=1)
    Music: int = Field(..., ge=0, le=1)
    Volunteering: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    predicted_gpa: float
    timestamp: str
    student_data: dict
    s3_saved: bool


# ======================================================
# Función para guardar en S3
# ======================================================

def save_prediction_to_s3(prediction_data: dict):
    if s3_client is None:
        return False

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        key = f"{S3_PREFIX}prediction_{timestamp}.json"

        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(prediction_data, indent=2),
            ContentType="application/json"
        )
        print(f"✓ Guardado en S3: {key}")
        return True

    except Exception as e:
        print("❌ Error guardando en S3:", e)
        return False


# ======================================================
# Endpoints
# ======================================================

@app.get("/")
def home():
    return {"API": "GPA Prediction API", "status": "OK"}


@app.get("/health")
def health():
    return {
        "pipeline_loaded": pipeline is not None,
        "s3_available": s3_client is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentData):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline no disponible")

    try:
        student_dict = student.model_dump()

        # Predicción usando método del pipeline
        pred = pipeline.predict_single(student_dict)
        pred = float(np.clip(pred, 0.0, 4.0))  # asegurar rango válido

        timestamp = datetime.now().isoformat()
        prediction_data = {
            "timestamp": timestamp,
            "student_data": student_dict,
            "predicted_gpa": pred
        }

        # Guardar en S3
        saved = save_prediction_to_s3(prediction_data)

        return PredictionResponse(
            predicted_gpa=round(pred, 2),
            timestamp=timestamp,
            student_data=student_dict,
            s3_saved=saved
        )

    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
