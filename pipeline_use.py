# Proceso #1
# Esto es lo que se usaría en FastAPI(no cambia nada respecto lo que se tenía)

import joblib
import pandas as pd

model = joblib.load('pipeline_model/model.pkl')

data = pd.DataFrame([{
    "StudentID": 1001,
    "Age": 17,
    "Gender": 1,
    "Ethnicity": 1,
    "ParentalEducation": 2,
    "StudyTimeWeekly": 15.5,
    "Absences": 2,
    "Tutoring": 1,
    "ParentalSupport": 2,
    "Extracurricular": 1,
    "Sports": 1,
    "Music": 0,
    "Volunteering": 1,
    "GradeClass": 2
}])

pred = model.predict(data)
print(pred)

# Proceso #2
# Usar MLFlow para publicar el endpoint

# Levantar el endpoint
# mlflow models serve -m /home/mendez/mlflow_test/MLOps-MLFlow/mlruns/535902209982421411/ff6fed6f64bf4362bce105df581645c5/artifacts/pipeline_medical_insurance -p 2222 --no-conda
# Hacer petición
# curl -X POST -H "Content-Type: application/json" -d '{"inputs": [{"age": 29, "bmi": 27.9, "children": 0, "sex": "female", "region": "southwest", "smoker": "no"}]}' http://127.0.0.1:2222/invocations
