import pandas as pd
import numpy as np
import os
import argparse
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import plot_importance

def load_and_merge_data(claims_path, options_path):
    """Loads and merges the claims and options datasets."""
    if not (os.path.exists(claims_path) and os.path.exists(options_path)):
        raise FileNotFoundError("Data files not found. Please check the paths.")
        
    print("Loading datasets...")
    claims_df = pd.read_excel(claims_path)
    options_df = pd.read_excel(options_path)

    merged_df = pd.merge(claims_df, options_df, left_on="Truck Number", right_on="Truck")
    print(f"DataFrames merged successfully. Shape: {merged_df.shape}")
    return merged_df

def preprocess_data(df):
    """Encodes categoricals and ensures targets are numerical."""
    print("Preprocessing data...")
    df = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column].astype(str))

    # Ensure target variables are numerical and drop NaNs
    df['Scale Claim Cost'] = pd.to_numeric(df['Scale Claim Cost'], errors='coerce')
    df['Scale Labor Cost'] = pd.to_numeric(df['Scale Labor Cost'], errors='coerce')
    df = df.dropna(subset=['Scale Claim Cost', 'Scale Labor Cost'])
    
    return df

def train_and_evaluate_xgboost(X, y):
    """Trains the XGBoost model, evaluates it, and plots feature importance."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Training XGBoost Regressor...")
    xg_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xg_model.fit(X_train, y_train)

    # Predictions and Metrics
    y_pred = xg_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-Squared (R2): {r2:.4f}")

    # Cross-Validation
    print("\nRunning 5-Fold Cross Validation...")
    cv_scores = cross_val_score(xg_model, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f"Cross-Validated MAE: {np.mean(-cv_scores):.4f}")

    # Plot Importance
    plot_importance(xg_model, max_num_features=10)
    plt.title('Top 10 Feature Importances from XGBoost')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')
    print("\nFeature importance plot saved as 'xgboost_feature_importance.png'.")

if __name__ == "__main__":
    # Setup argument parsing for flexible file paths
    parser = argparse.ArgumentParser(description="Train Fleet Claim Cost Model")
    parser.add_argument("--claims_path", type=str, default="Claim Information file for UNT.xlsx", help="Path to claims data")
    parser.add_argument("--options_path", type=str, default="Option Code Information file for UNT - Final.xlsx", help="Path to options data")
    args = parser.parse_args()

    try:
        # Pipeline Execution
        merged_data = load_and_merge_data(args.claims_path, args.options_path)
        processed_data = preprocess_data(merged_data)
        
        # Define Features and Target
        # Note: Ensure these attribute columns match your actual processed dataframe
        feature_cols = ['Attribute 1', 'Attribute 2', 'Attribute 3', 'Attribute 4', 
                        'Attribute 5', 'Attribute 6', 'Attribute 7', 'Attribute 8', 'Scale Labor Cost']
        
        # Filter to only existing columns to prevent KeyError
        existing_features = [col for col in feature_cols if col in processed_data.columns]
        
        X = processed_data[existing_features]
        y = processed_data['Scale Claim Cost']
        
        train_and_evaluate_xgboost(X, y)
        
    except Exception as e:
        print(f"Pipeline Error: {e}")
