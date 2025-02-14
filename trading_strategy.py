class ARIMAStrategy:
    def __init__(self, data, forecaster):

        self.data = data
        self.forecaster = forecaster
        self.position = None
        self.forecasts = self.forecaster.rolling_forecast()
        self.data['Forecast'] = [None] * (len(self.data) - len(self.forecasts)) + self.forecasts

    def next(self):
        forecast = self.data['Forecast'].iloc[-1]
        current_price = self.data['Close'].iloc[-1]
        if forecast > current_price:
            self.position = 'long'
        elif forecast < current_price:
            self.position = 'short'
        else:
            self.position = None
