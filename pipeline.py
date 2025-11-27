import os
import mlflow
import pandas as pd
import mlflow.sklearn
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlflow.models.signature import infer_signature
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings("ignore")

# Cargar datos
data = pd.read_csv('data/Student_performance_data.csv')
X = data.drop('GPA', axis=1)
y = data['GPA']

# Definir features
feature_info = joblib.load('artifacts/feature_info.pkl')
categorical_features = feature_info['categorical_cols']
numerical_features = feature_info['numerical_cols']

# Cargar mejores parámetros del modelo entrenado
best_params = {
    'n_estimators': 387,
    'learning_rate': 0.09605821612561193,
    'max_depth': 3,
    'min_samples_split': 8,
    'min_samples_leaf': 4,
    'subsample': 0.6343799307376653,
    'random_state': 42
}

# Label Encoder personalizado para ColumnTransformer
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
    
    def fit(self, X, y=None):
        X_copy = X.copy()
        for col in X_copy.columns:
            le = LabelEncoder()
            le.fit(X_copy[col].astype(str))
            self.encoders[col] = le
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            le = self.encoders[col]
            X_copy[col] = X_copy[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
        return X_copy

mlflow.set_experiment("Taller3_MLOps")

with mlflow.start_run(run_name="final_model_pipeline"):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', CustomLabelEncoder(), categorical_features)
        ]
    )

    model = GradientBoostingRegressor(
        **best_params
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X, y)

    input_example = X[:1]
    signature = infer_signature(X, pipeline.predict(X))

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="pipeline_student_gpa",
        input_example=input_example,
        signature=signature
    )

print("Pipeline saved with MLflow.")