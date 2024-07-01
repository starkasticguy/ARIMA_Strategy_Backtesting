import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class RollingPredictor:
    def __init__(self, data, best_orders, initial_window_size):
        self.data = data
        self.best_orders = best_orders
        self.initial_window_size = initial_window_size

    def rolling_forecast(self):
        predictions = {}
        for column, order in self.best_orders.items():
            predictions[column] = self._rolling_forecast_for_column(self.data[column], order)

        return predictions

    def _rolling_forecast_for_column(self, series, order):
        rolling_predictions = []
        for i in range(self.initial_window_size, len(series)):
            train = series[:i]
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            pred = model_fit.forecast()[0]
            rolling_predictions.append(pred)

        return rolling_predictions

    def future_forecast(self, steps=5):
        future_predictions = {}
        for column, order in self.best_orders.items():
            future_predictions[column] = self._future_forecast_for_column(self.data[column], order, steps)

        return future_predictions

    def _future_forecast_for_column(self, series, order, steps):
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
