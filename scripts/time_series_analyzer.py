import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt

class TimeSeriesAnalyzer:
    def __init__(self, data):
        self.data = data
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a pandas DatetimeIndex")
        self.data = self.data.asfreq('D')  # Set frequency to daily
        self.data = self.data.interpolate()  # Interpolate missing values

    def decompose(self, model='additive'):
        decomposition = seasonal_decompose(self.data, model=model)
        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.residual = decomposition.resid  # Correct attribute name
        return decomposition

    def plot_decomposition(self):
        decomposition = self.decompose()
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
        decomposition.observed.plot(ax=ax1, legend=False)
        ax1.set_ylabel('Observed')
        decomposition.trend.plot(ax=ax2, legend=False)
        ax2.set_ylabel('Trend')
        decomposition.seasonal.plot(ax=ax3, legend=False)
        ax3.set_ylabel('Seasonal')
        decomposition.resid.plot(ax=ax4, legend=False)  # Correct attribute name
        ax4.set_ylabel('Residual')
        plt.tight_layout()
        plt.show()

    def check_stationarity(self):
        result = adfuller(self.data.dropna())
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        return result[1] < 0.05

    def plot_acf_pacf(self, lags=40):
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        acf_values = acf(self.data.dropna(), nlags=lags)
        pacf_values = pacf(self.data.dropna(), nlags=lags)
        ax[0].stem(acf_values)
        ax[0].set_title('ACF')
        ax[1].stem(pacf_values)
        ax[1].set_title('PACF')
        plt.tight_layout()
        plt.show()
