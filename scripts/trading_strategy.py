import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from scripts.data_preprocessor import IndianBusinessCalendar

# Define a custom business day
indian_bday = CustomBusinessDay(calendar=IndianBusinessCalendar())

class TradingStrategy:
    def __init__(self, predictions, actual, future_predictions):
        self.predictions = {k: pd.Series(v, index=actual.index[-len(v):]) for k, v in predictions.items()}
        self.actual = actual
        # Create a date range for future predictions using Indian business days
        future_index = pd.date_range(start=actual.index[-1] + indian_bday, periods=len(list(future_predictions.values())[0]), freq=indian_bday)
        self.future_predictions = {k: pd.Series(v, index=future_index) for k, v in future_predictions.items()}

    def backtest(self):
        signals = {k: np.sign(v - self.actual[k].shift(1)) for k, v in self.predictions.items()}
        returns = {k: s * self.actual[k].pct_change().shift(-1) for k, s in signals.items()}
        cumulative_return = {k: (r + 1).cumprod() for k, r in returns.items()}

        self.buy_signals = {k: self.actual[k][s > 0] for k, s in signals.items()}
        self.sell_signals = {k: self.actual[k][s < 0] for k, s in signals.items()}
        self.volatility = {k: r.rolling(window=20).std() * np.sqrt(252) for k, r in returns.items()}

        return cumulative_return

    def future_return(self):
        future_dod_change = {k: v.pct_change().dropna() for k, v in self.future_predictions.items()}
        return future_dod_change

    def last_5_days(self):
        last_5_days_prices = {k: v[-5:] for k, v in self.actual.items()}
        last_5_days_dod_change = {k: v.pct_change().dropna()[-5:] for k, v in self.actual.items()}
        return last_5_days_prices, last_5_days_dod_change
