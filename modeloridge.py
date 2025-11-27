import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import optuna
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from mlflow.models.signature import infer_signature
import optuna.visualization as vis
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configurar MLflow
mlflow.set_experiment("Taller3_MLOps")

# Cargar datos
X_train = np.load('artifacts/X_train.npy')
X_test = np.load('artifacts/X_test.npy')
y_train = np.load('artifacts/y_train.npy')
y_test = np.load('artifacts/y_test.npy')


def objective_ridge(trial):

    alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
    
    model = Ridge(
        alpha=alpha,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    
    run_name = f"Iteration {trial.number + 1}"
    with mlflow.start_run(nested=True, run_name=run_name):
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
    
    return rmse

study_ridge = optuna.create_study(direction='minimize')

with mlflow.start_run(run_name="Ridge_Optimization"):
    study_ridge.optimize(objective_ridge, n_trials=10, show_progress_bar=True)
    
    mlflow.log_params(study_ridge.best_params)
    mlflow.log_metric("best_rmse", study_ridge.best_value)

    # Entrenar modelo final
    best_params = study_ridge.best_params
    final_model_ridge = Ridge(
        **best_params,
        random_state=42
    )
    final_model_ridge.fit(X_train, y_train)
    y_pred_final_ridge = final_model_ridge.predict(X_test)

    
    final_r2 = r2_score(y_test, y_pred_final_ridge)
    final_mae = mean_absolute_error(y_test, y_pred_final_ridge)
    final_rmse = mean_squared_error(y_test, y_pred_final_ridge)**0.5
    
    mlflow.log_metric("final_r2", final_r2)
    mlflow.log_metric("final_mae", final_mae)
    mlflow.log_metric("final_rmse", final_rmse)
    
    columns = [f"feature_{i}" for i in range(X_train.shape[1])]
    input_example = pd.DataFrame(X_train[:1], columns=columns)

    signature = infer_signature(X_train, final_model_ridge.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=final_model_ridge,
        artifact_path="model_ridge",
        input_example=input_example,
        signature=signature
    )
    
    opt_history_path = "optimization_history_ridge.png"
    opt_slice_path = "slice_plot_ridge.png"
    
    vis.plot_optimization_history(study_ridge).write_image(opt_history_path)
    vis.plot_slice(study_ridge).write_image(opt_slice_path)
    
    mlflow.log_artifact(opt_history_path)
    mlflow.log_artifact(opt_slice_path)

    os.system(f"rm *.png")
    
print("Best params:")
print(study_ridge.best_params)