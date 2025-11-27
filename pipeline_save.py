import mlflow.sklearn

run_id = "28569859d0454bf89fd74f8caf3c057a"  # Reemplaza con tu RUN_ID
artifact_path = "pipeline_student_gpa"  
model_path = f"runs:/{run_id}/{artifact_path}"

loaded_model = mlflow.pyfunc.load_model(model_path)

output_dir = "pipeline_model"
mlflow.sklearn.save_model(sk_model=loaded_model._model_impl, path=output_dir)
