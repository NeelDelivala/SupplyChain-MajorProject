from src.demand import ARIMADemandForecaster
import os

print("Training master ARIMA models...")
forecaster = ARIMADemandForecaster()
forecaster.load_data('Demand.csv')  # YOUR master dataset
forecaster.prepare_ts_data()
forecaster.train_models(test_size=30)
forecaster.forecast_future(30)
forecaster.save_models('models/arima_models.pkl')

print("Models saved! Ready for user uploads.")
print(forecaster.get_summary())