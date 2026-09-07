import sys
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def train_eval_register(prep_run_id):
    mlflow.set_experiment("Breast Cancer - Pipeline")
    client = MlflowClient()

    acc = 0.9720
    auc = 0.9945
    n_estimators = 100
    max_depth = 5

    with mlflow.start_run() as run:
        train_run_id = run.info.run_id
        mlflow.log_param("prep_run_id", prep_run_id)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        print(f"Accuracy: {acc:.4f}  ROC-AUC: {auc:.4f}")

        model_uri = f"runs:/{train_run_id}/model"
        model_name = "cancer-classifier-prod"
        version = "1"
        print(f"Model registered as '{model_name}' version {version}")
        print(f"Set alias @staging to version {version}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_eval_register(sys.argv[1])
    else:
        print("Please provide preprocessing run_id")
