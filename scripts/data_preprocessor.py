import pandas as pd

class DataPreprocessor:
    @staticmethod
    def preprocess(data):
        data.dropna(inplace=True)
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a pandas DatetimeIndex")
        data = data.asfreq('D')  # Set frequency to daily
        data = data.interpolate()  # Interpolate missing values
        return data
