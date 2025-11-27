import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlflow.models.signature import infer_signature
import optuna.visualization as vis

# Configurar MLflow
mlflow.set_experiment("Taller3_MLOps")

# Cargar datos
X_train = np.load('artifacts/X_train.npy')
X_test = np.load('artifacts/X_test.npy')
y_train = np.load('artifacts/y_train.npy')
y_test = np.load('artifacts/y_test.npy')


def objective_rf(trial):

    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    
    run_name = f"Iteration {trial.number + 1}"
    with mlflow.start_run(nested=True, run_name=run_name):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
    
    return rmse

study_rf = optuna.create_study(direction='minimize')

with mlflow.start_run(run_name="RandomForest_Optimization"):
    study_rf.optimize(objective_rf, n_trials=10, show_progress_bar=True)
    
    mlflow.log_params(study_rf.best_params)
    mlflow.log_metric("best_rmse", study_rf.best_value)

    # Entrenar modelo final
    best_params = study_rf.best_params
    final_model_rf = RandomForestRegressor(
        **best_params,
        random_state=42
    )
    final_model_rf.fit(X_train, y_train)
    y_pred_final_rf = final_model_rf.predict(X_test)

    
    final_r2 = r2_score(y_test, y_pred_final_rf)
    final_mae = mean_absolute_error(y_test, y_pred_final_rf)
    final_rmse = mean_squared_error(y_test, y_pred_final_rf)**0.5
    
    mlflow.log_metric("final_r2", final_r2)
    mlflow.log_metric("final_mae", final_mae)
    mlflow.log_metric("final_rmse", final_rmse)
    
    columns = [f"feature_{i}" for i in range(X_train.shape[1])]
    input_example = pd.DataFrame(X_train[:1], columns=columns)

    signature = infer_signature(X_train, final_model_rf.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=final_model_rf,
        artifact_path="model_random_forest",
        input_example=input_example,
        signature=signature
    )
    
    opt_history_path = "optimization_history.png"
    opt_slice_path = "slice_plot.png"
    
    vis.plot_optimization_history(study_rf).write_image(opt_history_path)
    vis.plot_slice(study_rf).write_image(opt_slice_path)
    
    mlflow.log_artifact(opt_history_path)
    mlflow.log_artifact(opt_slice_path)

    os.system(f"rm *.png")


