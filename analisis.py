"""
Taller 3 - MLOps
Análisis Exploratorio de Datos
Dataset: Students Performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Cargar datos
df = pd.read_csv('data/Student_performance_data.csv')

print("="*80)
print("ENTENDIMIENTO DE LOS DATOS")
print("="*80)

# Información general
print("\n1. INFORMACIÓN GENERAL DEL DATASET")
print(f"Dimensiones: {df.shape}")
print(f"\nColumnas: {df.columns.tolist()}")
print(f"\nTipos de datos:\n{df.dtypes}")

# Estadísticas descriptivas
print("\n2. ESTADÍSTICAS DESCRIPTIVAS")
print(df.describe())

# Valores nulos
print("\n3. VALORES NULOS")
print(df.isnull().sum())

# Análisis de la variable objetivo (GPA)
print("\n4. ANÁLISIS DE LA VARIABLE OBJETIVO: GPA")
print(f"Media: {df['GPA'].mean():.2f}")
print(f"Mediana: {df['GPA'].median():.2f}")
print(f"Desviación estándar: {df['GPA'].std():.2f}")


# Visualizaciones
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Distribución de GPA
axes[0, 0].hist(df['GPA'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribución de GPA')
axes[0, 0].set_xlabel('GPA')
axes[0, 0].set_ylabel('Frecuencia')

# GPA por género
df.boxplot(column='GPA', by='Gender', ax=axes[0, 1])
axes[0, 1].set_title('GPA por Género')
axes[0, 1].set_xlabel('Género')

# GPA vs Horas de estudio
axes[0, 2].scatter(df['StudyTimeWeekly'], df['GPA'], alpha=0.5)
axes[0, 2].set_title('GPA vs Horas de Estudio Semanales')
axes[0, 2].set_xlabel('Horas de Estudio')
axes[0, 2].set_ylabel('GPA')

# GPA vs Absences
axes[1, 0].scatter(df['Absences'], df['GPA'], alpha=0.5)
axes[1, 0].set_title('GPA vs Ausencias')
axes[1, 0].set_xlabel('Ausencias')
axes[1, 0].set_ylabel('GPA')

# GPA por nivel de educación parental
df.boxplot(column='GPA', by='ParentalEducation', ax=axes[1, 1])
axes[1, 1].set_title('GPA por Educación Parental')
axes[1, 1].tick_params(axis='x', rotation=45)

# GPA por participación en extracurriculares
df.boxplot(column='GPA', by='Extracurricular', ax=axes[1, 2])
axes[1, 2].set_title('GPA por Actividades Extracurriculares')

plt.tight_layout()
plt.savefig('exploratory_analysis.png', dpi=300, bbox_inches='tight')

# Correlaciones con GPA
print("\n5. CORRELACIONES CON GPA (Variables numéricas)")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()['GPA'].sort_values(ascending=False)
print(correlations)

# Matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Matriz de Correlación - Variables Numéricas')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')


