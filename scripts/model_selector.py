from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import itertools
import matplotlib.pyplot as plt

class OrderSelector:
    def __init__(self, data):
        self.data = data

    def adf_test(self, series):
        result = adfuller(series)
        return result[1]

    def determine_d(self):
        d = 0
        temp_series = self.data.copy()
        while self.adf_test(temp_series) > 0.05:
            temp_series = temp_series.diff().dropna()
            d += 1
        return d

    def plot_acf_pacf(self, series, lags=40):
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        acf_values = acf(series, nlags=lags)
        pacf_values = pacf(series, nlags=lags)
        ax[0].stem(acf_values)
        ax[0].set_title('ACF')
        ax[1].stem(pacf_values)
        ax[1].set_title('PACF')
        plt.tight_layout()
        plt.show()

    def suggest_initial_p_q(self, series):
        self.plot_acf_pacf(series)
        return range(0, 4), range(0, 4)

    def grid_search_arima(self, p_range, d, q_range):
        best_aic = np.inf
        best_order = None
        best_model = None

        for p, q in itertools.product(p_range, q_range):
            try:
                model = ARIMA(self.data, order=(p, d, q))
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = (p, d, q)
                    best_model = model_fit
            except:
                continue

        return best_order, best_model

    def select_order(self):
        d = self.determine_d()
        differenced_series = self.data.diff(d).dropna()
        p_range, q_range = self.suggest_initial_p_q(differenced_series)
        best_order, best_model = self.grid_search_arima(p_range, d, q_range)
        return best_order, best_model
