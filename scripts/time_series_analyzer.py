import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt

class TimeSeriesAnalyzer:
    def __init__(self, data):
        self.data = data

    def plot_decomposition(self):
        period = 5  # Set the period to 5 business days (weekly seasonality for business days)
        for column in self.data.columns:
            decomposition = seasonal_decompose(self.data[column], model='additive', period=period)
            decomposition.plot()
            plt.title(f'Decomposition of {column} Price')
            plt.show()

    def check_stationarity(self):
        results = {}
        for column in self.data.columns:
            result = adfuller(self.data[column].dropna())
            print(f'ADF Statistic for {column}: {result[0]}')
            print(f'p-value for {column}: {result[1]}')
            results[column] = result[1] < 0.05
        return results

    def plot_acf_pacf(self, lags=40):
        for column in self.data.columns:
            fig, ax = plt.subplots(2, 1, figsize=(12, 8))
            acf_values = acf(self.data[column].dropna(), nlags=lags)
            pacf_values = pacf(self.data[column].dropna(), nlags=lags)

            ax[0].stem(range(len(acf_values)), acf_values, basefmt=" ")
            ax[0].set_title(f'Autocorrelation Function (ACF) for {column}')
            ax[0].set_xlabel('Lags')
            ax[0].set_ylabel('ACF')

            ax[1].stem(range(len(pacf_values)), pacf_values, basefmt=" ")
            ax[1].set_title(f'Partial Autocorrelation Function (PACF) for {column}')
            ax[1].set_xlabel('Lags')
            ax[1].set_ylabel('PACF')

            plt.tight_layout()
            plt.show()
