import mlflow
import mlflow.sklearn
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline as SklearnPipeline

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from mlflow.models.signature import infer_signature # Import for signature

from preprocess import load_and_preprocess_heart_disease_data, DATA_FILE

MLFLOW_TRACKING_URI = "mlruns"
TUNING_EXPERIMENT_NAME = "Heart_Disease_RF_Tuning"
BASE_MODEL_EXPERIMENT_NAME = "Heart_Disease_Classification_Baselines" # For referencing preprocessor if needed

# Load and preprocess data once
print(f"Loading and preprocessing data from: {DATA_FILE} for tuning...")
# We need the raw X_train to create an input_example for the full pipeline
# Call once and unpack all necessary components
(
    X_train_p, 
    X_test_p, 
    y_train_s, 
    y_test_s, 
    fitted_preprocessor, 
    X_train_raw_for_example  # This is the raw sample
) = load_and_preprocess_heart_disease_data(data_path=DATA_FILE, return_raw_train_data=True)

if X_train_p is None or X_train_raw_for_example is None: # Check both processed and raw sample
    print("Data preprocessing failed. Exiting tuning script.")
    exit()

print(f"Data ready for tuning. X_train_processed shape: {X_train_p.shape}, X_test_processed shape: {X_test_p.shape}")

def objective(params):
    """
    Objective function for Hyperopt.
    Trains a RandomForest model with given params and logs to a nested MLflow run.
    Returns the negative F1-score for minimization.
    """
    # Extract parameters, converting float outputs from hyperopt to int where necessary
    n_estimators = int(params['n_estimators'])
    max_depth = int(params['max_depth'])
    min_samples_split = int(params['min_samples_split'])
    min_samples_leaf = int(params['min_samples_leaf'])
    criterion = params['criterion']
    class_weight = params.get('class_weight') # Optional

    current_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'criterion': criterion,
        'random_state': 42 # Keep random state for comparability
    }
    if class_weight: # Add if 'balanced' is chosen
        current_params['class_weight'] = class_weight

    # Start a nested MLflow run for this trial
    with mlflow.start_run(nested=True):
        mlflow.log_params(current_params)

        model = RandomForestClassifier(**current_params)
        
      
        model.fit(X_train_p.values if isinstance(X_train_p, pd.DataFrame) else X_train_p, y_train_s)
        
        preds = model.predict(X_test_p.values if isinstance(X_test_p, pd.DataFrame) else X_test_p)
        
        # Using X_test_p as validation set for tuning
        f1 = f1_score(y_test_s, preds, average='weighted')
        accuracy = accuracy_score(y_test_s, preds)

        mlflow.log_metric("f1_weighted_validation", f1)
        mlflow.log_metric("accuracy_validation", accuracy)
        
        
        mlflow.sklearn.log_model(model, "random_forest_model_trial") # Simpler for now for nested

    # Hyperopt minimizes, so return negative of the metric we want to maximize (F1 score)
    return {'loss': -f1, 'status': STATUS_OK, 'model_params': current_params, 'accuracy': accuracy}


# Define the search space for Hyperopt
search_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 25), 
    'max_depth': hp.quniform('max_depth', 5, 25, 1),          
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1), 
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),   
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    
}

def run_hyperparameter_tuning():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(TUNING_EXPERIMENT_NAME)
    if experiment is None:
        print(f"Creating MLflow experiment: '{TUNING_EXPERIMENT_NAME}'")
        mlflow.create_experiment(TUNING_EXPERIMENT_NAME)
    mlflow.set_experiment(TUNING_EXPERIMENT_NAME)

    trials = Trials()
    
    # Parent run for all tuning trials of RandomForest
    with mlflow.start_run(run_name="RandomForest_Hyperopt_Tuning") as parent_run:
        print(f"Parent MLflow Run for RF Tuning: {parent_run.info.run_id}")
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("tuning_library", "Hyperopt")
        mlflow.log_param("max_evaluations", 50) # Log how many trials we plan to run

        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=50, # Number of hyperparameter combinations to try
            trials=trials
        )

        print("\n--- Hyperparameter Tuning Complete ---")
        print("Best parameters found by Hyperopt:")
        
        best_trial = trials.best_trial
        
        print(f"  Raw best_params from fmin: {best_params}") 
        print(f"  Best trial F1 score: {-best_trial['result']['loss']:.4f}")
        print(f"  Best trial accuracy: {best_trial['result']['accuracy']:.4f}")
        print(f"  Best trial parameters (from trials object): {best_trial['result']['model_params']}")

        mlflow.log_params(best_trial['result']['model_params']) 
        mlflow.log_metric("best_f1_weighted_validation", -best_trial['result']['loss'])
        mlflow.log_metric("best_accuracy_validation", best_trial['result']['accuracy'])
        
        print("\nTraining final best model with identified optimal parameters...")
        best_rf_params = best_trial['result']['model_params']
        final_best_model = RandomForestClassifier(**best_rf_params)
        
        final_pipeline = SklearnPipeline(steps=[
            ('preprocessor', fitted_preprocessor),
            ('classifier', final_best_model)
        ])
        
      
        final_best_model.fit(X_train_p.values if isinstance(X_train_p, pd.DataFrame) else X_train_p, y_train_s)


        input_example = X_train_raw_for_example.head(5) # Using the raw X_train sample
        
        # Predictions for signature should come from the full pipeline
        predictions = final_pipeline.predict(input_example)
        signature = infer_signature(input_example, predictions)
        
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="best_random_forest_pipeline",
            signature=signature,                         
            input_example=input_example,                 
            registered_model_name="BestRandomForestHeartDisease" 
        )
        print(f"Logged best RandomForest pipeline to parent run {parent_run.info.run_id} with signature and input example, and registered as 'BestRandomForestHeartDisease'")

if __name__ == "__main__":
    run_hyperparameter_tuning() 