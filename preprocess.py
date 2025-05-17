import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

DATA_FILE = os.path.join("data", "heart_disease_uci.csv")

def load_and_preprocess_heart_disease_data(data_path=DATA_FILE, test_size=0.2, random_state=42, return_raw_train_data=False):
    """
    Loads, cleans, and preprocesses the Heart Disease UCI dataset.
    'num' column is binarized (0=no disease, 1=disease). Handles missing values.
    Categorical features are one-hot encoded, numerical features are scaled.
    'id' and 'dataset' columns are dropped.
    """
    try:
        df = pd.read_csv(data_path, na_values=['?', ''])
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        if return_raw_train_data:
            return None, None, None, None, None, None
        return None, None, None, None, None
    
    print(f"Loaded {data_path}, original shape: {df.shape}")

    df['target'] = np.where(df['num'] > 0, 1, 0)
    df = df.drop(columns=['num', 'id', 'dataset'])

    numerical_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    for col in numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str) 

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    valid_numerical_features = [f for f in numerical_features if f in df.columns]
    valid_categorical_features = [f for f in categorical_features if f in df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, valid_numerical_features),
            ('cat', categorical_transformer, valid_categorical_features)
        ], 
        remainder='drop'
    )

    X = df.drop('target', axis=1)
    y = df['target']

    if X.empty or y.empty:
        print("Error: Feature set (X) or target (y) is empty before splitting.")
        if return_raw_train_data:
            return None, None, None, None, None, None
        return None, None, None, None, None
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    X_train_raw_sample = X_train.head(5) if return_raw_train_data else None

    X_train_processed_np = preprocessor.fit_transform(X_train)
    X_test_processed_np = preprocessor.transform(X_test)
    
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = None # Fallback if get_feature_names_out fails

    if feature_names is not None:
        X_train_processed = pd.DataFrame(X_train_processed_np, columns=feature_names, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed_np, columns=feature_names, index=X_test.index)
    else: 
        X_train_processed = pd.DataFrame(X_train_processed_np, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed_np, index=X_test.index)

    print(f"Preprocessing complete. X_train_processed shape: {X_train_processed.shape}, X_test_processed shape: {X_test_processed.shape}")

    if return_raw_train_data:
        return X_train_processed, X_test_processed, y_train, y_test, preprocessor, X_train_raw_sample
    else:
        return X_train_processed, X_test_processed, y_train, y_test, preprocessor

if __name__ == '__main__':
    print("Running preprocess.py standalone for testing...")
    # Test without returning raw data
    X_train_df, X_test_df, y_train_s, y_test_s, fitted_preprocessor = load_and_preprocess_heart_disease_data()

    if X_train_df is not None:
        print("\n--- Preprocessing Test Output (Standard) ---")
        print(f"X_train_df.shape: {X_train_df.shape}, y_train_s.shape: {y_train_s.shape}")
        print(f"y_train distribution:\n{y_train_s.value_counts(normalize=True).round(2)}")
        print("------------------------------------------")
    else:
        print("Preprocessing failed during standalone test (Standard).")

    # Test with returning raw data
    print("\nRunning preprocess.py standalone for testing (with raw data return)...")
    results_with_raw = load_and_preprocess_heart_disease_data(return_raw_train_data=True)
    
    if results_with_raw[0] is not None: # Check if X_train_df is not None
        X_train_df_raw, X_test_df_raw, y_train_s_raw, y_test_s_raw, fitted_preprocessor_raw, x_train_sample_raw = results_with_raw
        print("\n--- Preprocessing Test Output (With Raw Sample) ---")
        print(f"X_train_df_raw.shape: {X_train_df_raw.shape}, y_train_s_raw.shape: {y_train_s_raw.shape}")
        print(f"y_train_raw distribution:\n{y_train_s_raw.value_counts(normalize=True).round(2)}")
        print(f"X_train_raw_sample (first 5 rows of X_train before processing):\n{x_train_sample_raw}")
        print("---------------------------------------------------")
    else:
        print("Preprocessing failed during standalone test (With Raw Sample).") 