import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "BestRandomForestHeartDisease"
MLFLOW_TRACKING_URI = "mlruns" 
def transition_model_stages():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        latest_versions = client.get_latest_versions(MODEL_NAME)
        if not latest_versions:
            print(f"No versions found for model '{MODEL_NAME}'. Please train and register a model first.")
            return

   
        target_version = None
        if len(latest_versions) == 1:
            target_version = latest_versions[0]
        else:
            
            latest_versions.sort(key=lambda v: int(v.version), reverse=True)
            target_version = latest_versions[0]
            print(f"Multiple 'latest' versions found. Targeting version: {target_version.version}")

        version_to_manage = target_version.version
        current_stage = target_version.current_stage
        print(f"Managing model '{MODEL_NAME}', version {version_to_manage}. Current stage: '{current_stage}'")

        # 1. Transition to Staging
        if current_stage != "Staging":
            print(f"Transitioning model '{MODEL_NAME}' version {version_to_manage} to 'Staging'...")
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version_to_manage,
                stage="Staging",
                archive_existing_versions=True 
            )
            print(f"Successfully transitioned model version {version_to_manage} to 'Staging'.")
        else:
            print(f"Model version {version_to_manage} is already in 'Staging'.")

      
        print(f"Transitioning model '{MODEL_NAME}' version {version_to_manage} to 'Production'...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version_to_manage,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Successfully transitioned model version {version_to_manage} to 'Production'.")

        print("\nModel stage management complete.")
        updated_version_details = client.get_model_version(MODEL_NAME, version_to_manage)
        print(f"Final stage of model '{MODEL_NAME}' version {version_to_manage}: '{updated_version_details.current_stage}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    transition_model_stages() 