from statsmodels.tsa.arima.model import ARIMA


class RollingPredictor:
    def __init__(self, data, model_order, window_size):
        self.data = data
        self.model_order = model_order
        self.window_size = window_size
        self.predictions = []

    def rolling_forecast(self):
        if len(self.data) <= self.window_size:
            print("Not enough data for the initial window size.")
            return []

        for i in range(len(self.data) - self.window_size):
            train = self.data[i:i + self.window_size]
            model = ARIMA(train, order=self.model_order)
            model_fit = model.fit()
            pred = model_fit.forecast(steps=1).iloc[0]  # Use iloc to access the first element
            self.predictions.append(pred)

        return self.predictions

    def future_forecast(self, steps=5):
        model = ARIMA(self.data, order=self.model_order)
        model_fit = model.fit()
        future_predictions = model_fit.forecast(steps=steps)
        return future_predictions
