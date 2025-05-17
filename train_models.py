import mlflow
import mlflow.sklearn
import os
import numpy as np
import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline as SklearnPipeline

from preprocess import load_and_preprocess_heart_disease_data, DATA_FILE

MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "Heart_Disease_Classification_Baselines"

def train_and_log_model(model, model_name, params, 
                        X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series, 
                        preprocessor):
    """Trains a model, logs it with its preprocessor to MLflow, and returns metrics."""
    with mlflow.start_run(run_name=model_name) as run:
        # print(f"Starting MLflow Run for {model_name} (ID: {run.info.run_id})") # Can be verbose

        print(f"Training {model_name}...")
        current_model_pipeline = SklearnPipeline(steps=[('classifier', model)])
        current_model_pipeline.fit(X_train.values if isinstance(X_train, pd.DataFrame) else X_train, y_train)

        y_pred = current_model_pipeline.predict(X_test.values if isinstance(X_test, pd.DataFrame) else X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"{model_name} Metrics - Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_weighted", f1)
        
        full_pipeline_to_log = SklearnPipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', current_model_pipeline.named_steps['classifier'])
        ])
        
        mlflow.sklearn.log_model(full_pipeline_to_log, f"{model_name}_pipeline")
        print(f"Logged {model_name} pipeline (MLflow Run ID: {run.info.run_id}).")
        
        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

def main():
    print(f"Loading and preprocessing data from: {DATA_FILE}")
    X_train_p, X_test_p, y_train_s, y_test_s, fitted_preprocessor = load_and_preprocess_heart_disease_data(data_path=DATA_FILE)

    if X_train_p is None:
        print("Data preprocessing failed. Exiting.")
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Creating MLflow experiment: '{EXPERIMENT_NAME}'")
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Using MLflow experiment: '{EXPERIMENT_NAME}'")

    models_to_train = {
        "LogisticRegression": (LogisticRegression(solver="liblinear", random_state=42, C=1.0, max_iter=200), 
                               {"solver": "liblinear", "C": 1.0, "random_state": 42, "max_iter": 200}),
        "RandomForest": (RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                         {"n_estimators": 100, "max_depth": 10, "random_state": 42}),
        "XGBoost": (XGBClassifier(n_estimators=100, random_state=42, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss"),
                    {"n_estimators": 100, "learning_rate": 0.1, "eval_metric": "logloss", "random_state": 42})
    }

    all_metrics = {}
    for model_name, (model_instance, params) in models_to_train.items():
        print(f"\n--- Training {model_name} ---")
        metrics = train_and_log_model(model_instance, model_name, params, 
                                    X_train_p, y_train_s, X_test_p, y_test_s, 
                                    fitted_preprocessor)
        all_metrics[model_name] = metrics

    print("\n--- Baseline Model Training Summary ---")
    for model_name, metrics in all_metrics.items():
        print(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

if __name__ == "__main__":
    main() 