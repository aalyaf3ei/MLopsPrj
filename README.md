# MLOps Project: Heart Disease Prediction

This project demonstrates a simplified MLOps workflow for a heart disease prediction task. It covers experiment tracking, model training, hyperparameter tuning, model registration and lifecycle management, and model deployment with performance monitoring using MLflow and FastAPI.

## Project Objectives Met

-   **Experiment Tracking:** Implemented using MLflow to track experiments for baseline models and hyperparameter tuning, logging parameters, metrics, and artifacts.
-   **Model Training and Tuning:** Developed baseline models (Logistic Regression, Random Forest, XGBoost) and performed hyperparameter tuning for Random Forest using Hyperopt, with all sessions logged in MLflow.
-   **Model Deployment (Custom API):** The best-tuned model, after being promoted to "Production" in the Model Registry, is served via a FastAPI endpoint.
-   **Model Registry & Lifecycle Management:** Models are registered in the MLflow Model Registry. A script (`manage_model_registry.py`) demonstrates transitioning model versions through stages (e.g., from "None" to "Staging" to "Production").
-   **Performance Monitoring:** The FastAPI application logs prediction requests and responses. A script (`monitor_model.py`) processes these logs to calculate basic metrics (e.g., prediction distribution, average input feature values) and logs them to a dedicated MLflow experiment ("Model_Monitoring_Heart_Disease") to track changes over time, simulating drift detection.

## Directory Structure

```
MLopsPrj/
├── data/
│   └── heart_disease_uci.csv       # Dataset
├── mlruns/                           # MLflow tracking data (automatically generated)
├── preprocess.py                   # Script for data loading and preprocessing
├── train_models.py                 # Script for training baseline models
├── tune_random_forest.py           # Script for hyperparameter tuning of Random Forest
├── serve_model.py                  # FastAPI application to serve the model (logs predictions)
├── manage_model_registry.py        # Script to transition model stages (e.g., Staging, Production)
├── monitor_model.py                # Script to process prediction logs and log monitoring metrics
├── prediction_log.csv              # Log file for API predictions (created by serve_model.py)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup Instructions

1.  **Clone the Repository (if applicable)**
    If this were a remote repository, you would clone it. For now, ensure you have the project files.

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    ```
    Activate it:
    -   Windows: `.\venv\Scripts\activate`
    -   macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies**
    With the virtual environment activated, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you encounter issues with `xgboost` or other specific libraries on your system, you might need to consult their individual installation guides for your OS.*

## Running the MLOps Workflow

### 1. Launch the MLflow UI (Optional, but Recommended)

It's helpful to have the MLflow UI running in a separate terminal so you can observe experiments and model registration in real-time.

Navigate to the project root directory (`MLopsPrj`) and run:
```bash
mlflow ui
```
This will start the MLflow tracking server, typically accessible at `http://127.0.0.1:5000` in your web browser.

### 2. Train Baseline Models

This script trains Logistic Regression, Random Forest, and XGBoost classifiers and logs them to MLflow under the "Heart_Disease_Classification_Baselines" experiment.

```bash
python train_models.py
```
-   Check the MLflow UI for the new experiment and its runs.

### 3. Perform Hyperparameter Tuning for Random Forest

This script uses Hyperopt to tune a Random Forest model. It logs trials as nested runs under a parent run in the "Heart_Disease_RF_Tuning" experiment. The best model (full scikit-learn pipeline) is logged and registered in the MLflow Model Registry as "BestRandomForestHeartDisease".

```bash
python tune_random_forest.py
```
-   Observe the tuning process in the MLflow UI.
-   After completion, check the "Heart_Disease_RF_Tuning" experiment for the parent run containing the best parameters and metrics.
-   Navigate to the "Models" section in MLflow UI to see the registered "BestRandomForestHeartDisease" model and its versions (initially in "None" stage).

### 4. Manage Model Lifecycle in Registry

This script transitions the latest version of the "BestRandomForestHeartDisease" model to "Staging" and then to "Production".

```bash
python manage_model_registry.py
```
-   Check the MLflow UI: Go to Models -> BestRandomForestHeartDisease. You should see the latest version now marked as "Production".

### 5. Serve the Production Model with FastAPI

The FastAPI application (`serve_model.py`) is configured to load the model currently in the "Production" stage from the MLflow Model Registry. It also logs every prediction request and response to `prediction_log.csv`.

Start the server (in a new terminal if MLflow UI is running):
```bash
uvicorn serve_model:app --reload
```
-   The API will typically be available at `http://127.0.0.1:8000`.
-   Access `http://127.0.0.1:8000/docs` in your browser for interactive API documentation (Swagger UI).
-   Send several POST requests to `http://127.0.0.1:8000/predict` with input data. Use varied JSON inputs to simulate different incoming data points. Each request will be logged.
    Example JSON body for the `/docs` page or `curl`:
    ```json
    {
      "age": 50,
      "sex": "1",
      "cp": "2",
      "trestbps": 120,
      "chol": 200,
      "fbs": "0",
      "restecg": "0",
      "thalch": 160,
      "exang": "0",
      "oldpeak": 1.0,
      "slope": "2",
      "ca": "0.0",
      "thal": "3.0"
    }
    ```

### 6. Run Performance Monitoring Check

After making some predictions via the API (which populates `prediction_log.csv`), run this script. It reads the log, calculates basic statistics (e.g., prediction distribution, average input age), and logs them as metrics to a new MLflow run in the "Model_Monitoring_Heart_Disease" experiment.

In a separate terminal:
```bash
python monitor_model.py
```
-   **To demonstrate monitoring over time:** 
    1.  Make an initial batch of predictions using the API.
    2.  Run `python monitor_model.py`.
    3.  Make a *second* batch of predictions (perhaps with slightly different data characteristics).
    4.  Run `python monitor_model.py` *again*.
    5.  In the MLflow UI, go to the "Model_Monitoring_Heart_Disease" experiment. Select the two (or more) monitoring runs and compare their metrics (e.g., `avg_input_age`, prediction distributions) to visualize how these characteristics might be changing over time.

## Conclusion

This project provides a foundational example of an MLOps pipeline, covering key stages from experiment tracking to model deployment and basic monitoring. It highlights how MLflow can be used as a central tool for managing the machine learning lifecycle.