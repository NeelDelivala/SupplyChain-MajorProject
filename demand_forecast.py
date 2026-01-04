"""
Demand Forecasting Model
This file shows how to integrate your Jupyter notebook code into the Flask app.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from statsmodels.tsa.arima.model import ARIMA
# from prophet import Prophet

def run_forecast(df, forecast_horizon=30, model_type='auto'):
    """
    Run demand forecasting on the uploaded data

    Parameters:
    - df: pandas DataFrame with columns [Date, SKU_ID, Region, Units_Sold]
    - forecast_horizon: number of days to forecast
    - model_type: 'auto', 'arima', 'prophet', 'xgboost', 'lstm'

    Returns:
    - results dict with metrics, forecasts, and visualizations
    """

    # Step 1: Prepare data
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # For demo: use one SKU
    if 'SKU_ID' in df.columns:
        sku = df['SKU_ID'].iloc[0]
        df_sku = df[df['SKU_ID'] == sku].copy()
    else:
        df_sku = df.copy()

    # Create time features
    df_sku['day_of_week'] = df_sku['Date'].dt.dayofweek
    df_sku['month'] = df_sku['Date'].dt.month
    df_sku['day'] = df_sku['Date'].dt.day

    # Step 2: Train-test split
    train_size = int(len(df_sku) * 0.8)
    train_df = df_sku[:train_size]
    test_df = df_sku[train_size:]

    # Step 3: Build model based on type
    if model_type == 'auto' or model_type == 'xgboost':
        # Simple Random Forest for demo
        features = ['day_of_week', 'month', 'day']
        X_train = train_df[features]
        y_train = train_df['Units_Sold']
        X_test = test_df[features]
        y_test = test_df['Units_Sold']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        r2 = r2_score(y_test, predictions)

    # TODO: Add ARIMA, Prophet, LSTM implementations from your notebooks

    # Step 4: Generate future forecast
    # (For demo, we'll just extrapolate)
    last_date = df_sku['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                  periods=forecast_horizon)

    # Step 5: Prepare results
    results = {
        'status': 'success',
        'rows': len(df),
        'columns': list(df.columns),
        'preview': df.head(10).to_dict('records'),
        'forecast_horizon': forecast_horizon,
        'model_type': model_type,
        'metrics': {
            'mape': round(mape, 2),
            'rmse': round(rmse, 2),
            'mae': round(mae, 2),
            'r2': round(r2, 2)
        },
        'forecast': {
            'dates': [str(d.date()) for d in future_dates],
            'values': [float(np.mean(y_train))] * forecast_horizon  # Placeholder
        }
    }

    return results

# Copy your Jupyter notebook functions here and call them from run_forecast()
