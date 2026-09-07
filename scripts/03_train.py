import sys
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model(prep_run_id):
    mlflow.set_experiment("Breast Cancer - Model Training")
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(prep_run_id, "processed_data")
    train_df = pd.read_csv(os.path.join(local_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(local_path, "test.csv"))
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    n_estimators = 100
    max_depth = 5
    with mlflow.start_run() as run:
        mlflow.set_tag("ml.step", "model_training")
        mlflow.log_param("prep_run_id", prep_run_id)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")
        print(f"Training finished. Training Run ID: {run.info.run_id}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_model(sys.argv[1])
    else:
        print("Please provide preprocessing run_id")
