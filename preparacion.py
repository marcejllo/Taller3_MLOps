import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path

# Crear directorio para artefactos
Path("artifacts").mkdir(exist_ok=True)

# Cargar datos
df = pd.read_csv('data/Student_performance_data.csv')

# 1. Separar features y target
X = df.drop('GPA', axis=1)
y = df['GPA']

print(f"\n1. Dimensiones originales:")
print(f"   Features: {X.shape}")
print(f"   Target: {y.shape}")

# 2. Identificar columnas categóricas y numéricas
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n2. Columnas categóricas: {categorical_cols}")
print(f"   Columnas numéricas: {numerical_cols}")

# 3. Codificar variables categóricas
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"\n   {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Guardar encoders
joblib.dump(label_encoders, 'artifacts/label_encoders.pkl')
print("\n Label encoders guardados")

# 4. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"\n3. División de datos:")
print(f"   Train: {X_train.shape[0]} muestras")
print(f"   Test: {X_test.shape[0]} muestras")

# 5. Escalar variables numéricas
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Guardar scaler
joblib.dump(scaler, 'artifacts/scaler.pkl')
print("\nScaler guardado")

# 6. Guardar datos procesados
np.save('artifacts/X_train.npy', X_train_scaled.values)
np.save('artifacts/X_test.npy', X_test_scaled.values)
np.save('artifacts/y_train.npy', y_train.values)
np.save('artifacts/y_test.npy', y_test.values)

# Guardar nombres de columnas
joblib.dump({
    'feature_names': X_encoded.columns.tolist(),
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols
}, 'artifacts/feature_info.pkl')

print("\nDatos procesados guardados en 'artifacts/'")

print("\n4. Resumen de transformaciones:")
print(f"   - Variables categóricas codificadas: {len(categorical_cols)}")
print(f"   - Variables numéricas escaladas: {len(numerical_cols)}")
print(f"   - Total de features: {X_train_scaled.shape[1]}")
