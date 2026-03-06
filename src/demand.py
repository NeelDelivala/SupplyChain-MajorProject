import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class ARIMADemandForecaster:
    def __init__(self, seasonal_period=7):  # Weekly seasonality for daily data
        self.models = {}
        self.forecasts = {}
        self.seasonal_period = seasonal_period
        self.metrics = {}
    
    def load_data(self, file_path):
        """Load and preprocess your Demand.csv"""
        self.df = pd.read_csv(r"C:\Users\neeld\OneDrive\Desktop\IDK2\Demand.csv")
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(['SKU_ID', 'Warehouse_ID', 'Date'])
        self.df.set_index('Date', inplace=True)
        print(f"Loaded {len(self.df)} records for {self.df['SKU_ID'].nunique()} SKUs across {self.df['Warehouse_ID'].nunique()} warehouses")
    
    def prepare_ts_data(self):
        """Group by SKU_ID + Warehouse_ID for multivariate time series"""
        self.groups = self.df.groupby(['SKU_ID', 'Warehouse_ID'])
        self.series = {}
        for (sku, wh), group in self.groups:
            key = f"{sku}_{wh}"
            ts = group['Units_Sold'].asfreq('D', fill_value=0)  # Daily frequency
            self.series[key] = ts
        print(f"Prepared {len(self.series)} time series")
    
    def auto_arima_fit(self, ts, max_p=5, max_d=2, max_q=5, max_P=2, max_D=1, max_Q=2, stepwise=True):
        """Auto ARIMA model selection using pmdarima"""
        model = pm.auto_arima(
            ts,
            start_p=0, start_q=0, max_p=max_p, max_d=max_d, max_q=max_q,
            seasonal=True, m=self.seasonal_period,
            start_P=0, max_P=max_P, max_D=max_D, max_Q=max_Q,
            stepwise=stepwise, suppress_warnings=True,
            error_action='ignore', trace=False,
            scoring='mse'
        )
        return model
    
    def train_models(self, test_size=30):
        """Train ARIMA models for all SKU-Warehouse combinations"""
        self.test_size = test_size
        for key, ts in self.series.items():
            if len(ts) < 100:  # Skip short series
                continue
                
            # Split train/test
            train = ts.iloc[:-test_size]
            test = ts.iloc[-test_size:]
            
            # Auto-fit ARIMA
            model = self.auto_arima_fit(train)
            self.models[key] = model
            
            # Forecast test period for validation
            forecast = model.predict(n_periods=test_size)
            mae = mean_absolute_error(test, forecast)
            self.metrics[key] = {'mae': mae, 'order': model.order, 'seasonal_order': model.seasonal_order}
            
            print(f"{key}: ARIMA{model.order}x{model.seasonal_order} MAE={mae:.2f}")
    
    def forecast_future(self, horizon=30):
        for key, model in self.models.items():
            # FIXED: Handle pmdarima predict return format
            forecast_result = model.predict(n_periods=horizon)
            
            # Try confidence intervals (pmdarima format)
            try:
                conf_int_result = model.predict(n_periods=horizon, return_conf_int=True)
                if isinstance(conf_int_result, tuple) and len(conf_int_result) == 2:
                    forecast_ci, conf_int = conf_int_result
                    lower_ci = conf_int[:, 0]
                    upper_ci = conf_int[:, 1]
                else:
                    lower_ci = forecast_result * 0.9  # Fallback
                    upper_ci = forecast_result * 1.1
            except:
                lower_ci = forecast_result * 0.9  # 90% CI fallback
                upper_ci = forecast_result * 1.1
            
            dates = pd.date_range(
                start=self.series[key].index[-1] + timedelta(days=1),
                periods=horizon, freq='D'
            )
            
            self.forecasts[key] = pd.DataFrame({
                'date': dates,
                'forecast': forecast_result,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci
            })
    
    def plot_forecast(self, sku_wh_key, save_path=None):
        """Interactive Plotly forecast visualization"""
        if sku_wh_key not in self.forecasts:
            print(f"No forecast for {sku_wh_key}")
            return
        
        ts = self.series[sku_wh_key]
        fc = self.forecasts[sku_wh_key]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Demand Forecast', 'Residuals Diagnostics'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Forecast plot
        fig.add_trace(
            go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Historical',
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=fc['date'], y=fc['forecast'], mode='lines', name='Forecast',
                      line=dict(color='orange', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=fc['date'], y=fc['upper_ci'], fill=None,
                      line=dict(color='orange', width=0),
                      showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=fc['date'], y=fc['lower_ci'], fill='tonexty',
                      fillcolor='rgba(255,165,0,0.2)', line=dict(color='orange', width=0),
                      name='95% CI'),
            row=1, col=1
        )
        
        # Residuals plot (last 60 days)
        fitted = self.models[sku_wh_key].fittedvalues
        residuals = ts.iloc[-60:] - fitted.iloc[-60:]
        fig.add_trace(
            go.Scatter(x=residuals.index, y=residuals.values, mode='lines',
                      name='Residuals', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title=f"Demand Forecast: {sku_wh_key}",
                         xaxis_title="Date", yaxis_title="Units Sold")
        fig.update_xaxes(rangeslider_visible=False)
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
    
    def save_models(self, path='arima_models.pkl'):
        """Save trained models"""
        joblib.dump(self.models, path)
        joblib.dump(self.forecasts, path.replace('.pkl', '_forecasts.pkl'))
        print(f"Models saved to {path}")
    
    def get_summary(self):
        """Forecast summary table"""
        summary = []
        for key in self.forecasts:
            fc = self.forecasts[key]
            summary.append({
                'SKU_WH': key,
                'Avg_Forecast': fc['forecast'].mean(),
                'Total_Forecast_30d': fc['forecast'].sum(),
                'MAE': self.metrics[key]['mae']
            })
        return pd.DataFrame(summary).round(2)

# Flask API Integration Example
from flask import Flask, jsonify, request
app = Flask(__name__)
forecaster = ARIMADemandForecaster()

@app.route('/train', methods=['POST'])
def train():
    file_path = request.json['file_path']
    forecaster.load_data(file_path)
    forecaster.prepare_ts_data()
    forecaster.train_models()
    forecaster.forecast_future(30)
    forecaster.save_models()
    return jsonify({'status': 'trained', 'models': list(forecaster.models.keys())})

@app.route('/forecast/<sku_wh>')
def get_forecast(sku_wh):
    fc = forecaster.forecasts.get(sku_wh, {})
    return jsonify(fc.to_dict('records'))

@app.route('/summary')
def summary():
    return forecaster.get_summary().to_json()

if __name__ == '__main__':
    # USAGE: Train on your Demand.csv
    forecaster = ARIMADemandForecaster()
    forecaster.load_data('Demand.csv')
    forecaster.prepare_ts_data()
    forecaster.train_models(test_size=30)
    forecaster.forecast_future(30)
    
    # View results
    print(forecaster.get_summary())
    forecaster.plot_forecast(list(forecaster.models.keys())[0])  # First model
    
    print("ARIMA forecasting pipeline complete!")
