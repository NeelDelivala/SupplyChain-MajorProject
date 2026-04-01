"""
Step 3: Machine Learning Enhancement - Stock Risk Classification
Train RandomForestClassifier to predict stock risk level.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from typing import Optional, Tuple


def create_risk_labels(
    inventory: pd.Series,
    forecast_demand: pd.Series
) -> pd.Series:
    """
    Create risk labels from inventory & forecast demand:
    - High: inventory < forecast demand
    - Medium: inventory < 1.2 * forecast (buffer zone)
    - Low: otherwise
    """
    risk = np.where(
        inventory < forecast_demand,
        'High',
        np.where(inventory < 1.2 * forecast_demand, 'Medium', 'Low')
    )
    return pd.Series(risk, index=inventory.index)


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select and encode features for ML model.
    Default features: inventory, demand (units_sold), demand_forecast, price, discount, promotion, seasonality.
    """
    default_features = [
        'inventory_level', 'units_sold', 'demand_forecast',
        'price', 'discount_pct', 'is_promotion', 'seasonality_factor'
    ]
    cols = feature_cols or [c for c in default_features if c in df.columns]
    
    X = df[cols].copy()
    
    # Encode categorical if any (e.g. store_id, region) - add if present
    for col in ['store_id', 'region']:
        if col in df.columns and col not in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(df[col].astype(str))
            cols = list(X.columns)
    
    # Create labels
    y = create_risk_labels(df['inventory_level'], df['demand_forecast'])
    
    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, dict]:
    """
    Train RandomForestClassifier and evaluate.
    Returns (model, metrics_dict).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importance': dict(zip(X.columns, model.feature_importances_)),
    }
    
    return model, metrics


def save_model(model, filepath: str) -> None:
    """Save trained model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """Load trained model from disk."""
    return joblib.load(filepath)


def run_ml_pipeline(
    df: pd.DataFrame,
    model_path: Optional[str] = None
) -> Tuple[RandomForestClassifier, dict, pd.DataFrame]:
    """
    Full ML pipeline: prepare features, train, evaluate, save.
    Returns (model, metrics, df_with_predictions).
    """
    X, y = prepare_features(df)
    
    model, metrics = train_model(X, y)
    
    # Add predictions to dataframe
    df_result = df.copy()
    df_result['ml_risk_prediction'] = model.predict(X)
    
    if model_path:
        save_model(model, model_path)
    
    return model, metrics, df_result
