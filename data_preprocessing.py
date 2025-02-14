import pandas as pd
from statsmodels.tsa.stattools import adfuller


def load_and_preprocess_data(ticker, start_date, end_date):
    import yfinance as yf
    data = yf.download(ticker, start=start_date, end=end_date)
    print(data.head())
    # Ensure the date index is a datetime type and has a frequency
    data.index = pd.to_datetime(data.index)
    data = data.asfreq('D')

    # Fill missing values using forward fill and backward fill
    data = data.ffill().bfill()

    # Ensure the series is stationary
    adf_result = adfuller(data['Close'])
    if adf_result[1] > 0.05:
        data['Close'] = data['Close'].diff().dropna()

    # Double-check and fill any remaining NaN values
    data = data.ffill().bfill()
    print("Hello?")
    print(data.head())
    return data
