import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import itertools

class OrderSelector:
    def __init__(self, data):
        self.data = data

    def select_order(self):
        best_orders = {}
        best_models = {}
        for column in self.data.columns:
            best_order, best_model = self._select_best_arima_order_for_column(self.data[column])
            best_orders[column] = best_order
            best_models[column] = best_model

        return best_orders, best_models

    def _select_best_arima_order_for_column(self, series):
        p = d = q = range(0, 3)
        pdq = list(itertools.product(p, d, q))
        best_aic = float('inf')
        best_order = None
        best_model = None

        for param in pdq:
            try:
                model = ARIMA(series, order=param)
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = param
                    best_model = model_fit
            except:
                continue

        return best_order, best_model
