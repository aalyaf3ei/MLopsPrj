from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd
import os 
import csv # For logging
from datetime import datetime # For timestamping logs

PREDICTION_LOG_FILE = "prediction_log.csv"
LOG_FILE_HEADER = ['timestamp', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'prediction']

# Create log file with header if it doesn't exist
if not os.path.exists(PREDICTION_LOG_FILE):
    with open(PREDICTION_LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(LOG_FILE_HEADER)

class HeartDiseaseInput(BaseModel):
    age: int | None = None
    sex: str | None = None 
    cp: str | None = None 
    trestbps: float | None = None 
    chol: float | None = None 
    fbs: str | None = None 
    restecg: str | None = None 
    thalch: float | None = None 
    exang: str | None = None
    oldpeak: float | None = None 
    slope: str | None = None 
    ca: str | None = None 
    thal: str | None = None 

app = FastAPI(title="Heart Disease Prediction API", version="1.0")

MLFLOW_TRACKING_URI = "mlruns" 
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "BestRandomForestHeartDisease"
# Let's load the Production model now
MODEL_STAGE = "Production" 
try:
    print(f"Loading model '{MODEL_NAME}' version from stage: '{MODEL_STAGE}'")
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}" 
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None 

@app.on_event("startup")
async def startup_event():
    if model is None:
        print("MLflow model could not be loaded. Predictions will not be available.")
    else:
        print("Application startup: MLflow model is loaded and ready.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Prediction API. Use the /predict endpoint to make predictions."}

@app.post("/predict")
async def predict(data: HeartDiseaseInput):
    if model is None:
        return {"error": "Model not loaded. Cannot make predictions."}
    try:
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        prediction_array = model.predict(input_df)
        prediction_value = prediction_array[0] # Assuming single prediction

        # Log the request and prediction
        with open(PREDICTION_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            # Order of values must match LOG_FILE_HEADER
            log_row = [
                datetime.now().isoformat(),
                input_dict.get('age'), input_dict.get('sex'), input_dict.get('cp'),
                input_dict.get('trestbps'), input_dict.get('chol'), input_dict.get('fbs'),
                input_dict.get('restecg'), input_dict.get('thalch'), input_dict.get('exang'),
                input_dict.get('oldpeak'), input_dict.get('slope'), input_dict.get('ca'),
                input_dict.get('thal'),
                prediction_value
            ]
            writer.writerow(log_row)
     
        return {"prediction": prediction_array.tolist()}
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server with Uvicorn...")
  
    uvicorn.run("serve_model:app", host="0.0.0.0", port=8000, reload=True) 