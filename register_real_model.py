import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Breast Cancer - Pipeline")

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")
    run_id = run.info.run_id

model_uri = f"runs:/{run_id}/model"
model_name = "cancer-classifier-prod"
reg_model = mlflow.register_model(model_uri, model_name)

client = MlflowClient()
client.set_registered_model_alias(model_name, "staging", reg_model.version)
print(f"Successfully registered {model_name} version {reg_model.version} with alias @staging")
