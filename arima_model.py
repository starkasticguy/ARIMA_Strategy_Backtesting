import itertools
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

class ARIMAModel:
    def __init__(self, series):
        self.series = series

    def grid_search_arima_order(self, series):
        p = d = q = range(0, 3)
        pdq = list(itertools.product(p, d, q))
        best_aic = float('inf')
        best_order = None
        for param in pdq:
            try:
                model = ARIMA(series, order=param)
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = param
            except:
                continue
        return best_order

    def rolling_forecast(self, window_size=252):
        predictions = []
        for start in range(len(self.series) - window_size):
            train = self.series[start:start + window_size]
            test = self.series[start + window_size:start + window_size + 1]
            order = self.grid_search_arima_order(train)
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            predictions.append(forecast.iloc[0])
        return predictions

    def forecast_next_days(self, steps=5):
        order = self.grid_search_arima_order(self.series)
        model = ARIMA(self.series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
do