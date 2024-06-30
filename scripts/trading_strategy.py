import numpy as np
import pandas as pd


class TradingStrategy:
    def __init__(self, predictions, actual, future_predictions):
        self.predictions = pd.Series(predictions, index=actual.index)
        self.actual = actual
        # Create a date range for future predictions
        future_index = pd.date_range(start=actual.index[-1] + pd.Timedelta(days=1), periods=len(future_predictions),
                                     freq='D')
        self.future_predictions = pd.Series(future_predictions, index=future_index)

    def backtest(self):
        signals = np.sign(self.predictions - self.actual.shift(1))
        returns = signals * self.actual.pct_change().shift(-1)
        cumulative_return = (returns + 1).cumprod()

        self.buy_signals = self.actual[signals > 0]
        self.sell_signals = self.actual[signals < 0]
        self.volatility = returns.rolling(window=20).std() * np.sqrt(252)  # 20-day rolling volatility

        return cumulative_return

    def future_return(self):
        future_returns = self.future_predictions.pct_change().dropna()
        return future_returns

    def last_5_days(self):
        last_5_days_prices = self.actual[-5:]
        last_5_days_returns = self.actual.pct_change().dropna()[-5:]
        return last_5_days_prices, last_5_days_returns
