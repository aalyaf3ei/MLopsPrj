import mlflow
import pandas as pd
import os
from datetime import datetime

PREDICTION_LOG_FILE = "prediction_log.csv"
MONITORING_EXPERIMENT_NAME = "Model_Monitoring_Heart_Disease"
MLFLOW_TRACKING_URI = "mlruns" # Ensure this matches your setup

def run_monitoring_check():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Ensure the experiment exists
    experiment = mlflow.get_experiment_by_name(MONITORING_EXPERIMENT_NAME)
    if experiment is None:
        print(f"Creating new MLflow experiment for monitoring: '{MONITORING_EXPERIMENT_NAME}'")
        mlflow.create_experiment(MONITORING_EXPERIMENT_NAME)
    mlflow.set_experiment(MONITORING_EXPERIMENT_NAME)

    if not os.path.exists(PREDICTION_LOG_FILE):
        print(f"Prediction log file '{PREDICTION_LOG_FILE}' not found. Make some predictions first.")
        return

    try:
        log_df = pd.read_csv(PREDICTION_LOG_FILE)
    except pd.errors.EmptyDataError:
        print(f"Prediction log file '{PREDICTION_LOG_FILE}' is empty. Make some predictions first.")
        return

    if log_df.empty:
        print("No data in prediction log. Make some predictions first.")
        return

    with mlflow.start_run(run_name=f"Monitoring_Check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        print(f"Starting monitoring run for data in '{PREDICTION_LOG_FILE}'")
        mlflow.log_param("log_file_processed", PREDICTION_LOG_FILE)
        mlflow.log_param("number_of_predictions_logged", len(log_df))

        # 1. Prediction distribution
        if 'prediction' in log_df.columns:
            prediction_counts = log_df['prediction'].value_counts(normalize=True)
            for cls, perc in prediction_counts.items():
                mlflow.log_metric(f"prediction_dist_class_{cls}", round(perc, 4))
            print(f"Logged prediction distribution: {prediction_counts.to_dict()}")
        else:
            print("Warning: 'prediction' column not found in log.")

        # 2. Average of a numerical feature (e.g., 'age')
        if 'age' in log_df.columns:
            # Ensure age is numeric, coerce errors for non-numeric strings if any slipped in (should not happen with Pydantic)
            log_df['age'] = pd.to_numeric(log_df['age'], errors='coerce') 
            avg_age = log_df['age'].mean()
            if pd.notna(avg_age):
                mlflow.log_metric("avg_input_age", round(avg_age, 2))
                print(f"Logged average input age: {avg_age:.2f}")
            else:
                 print("Could not calculate average age, possibly all NaNs or no numeric age data.")
        else:
            print("Warning: 'age' column not found in log for average calculation.")
        
        # Add more metrics as needed (e.g., missing value counts, other feature averages)
        
        # As a simple artifact, let's log the current prediction log itself for this monitoring run
        # In a real scenario, you might log a summary report or plots.
        mlflow.log_artifact(PREDICTION_LOG_FILE, artifact_path="prediction_logs_snapshot")
        print(f"Logged '{PREDICTION_LOG_FILE}' as an artifact for this monitoring run.")

        print("Monitoring check complete. Metrics logged to MLflow.")

if __name__ == "__main__":
    run_monitoring_check() 