# Import necessary libraries
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from pmdarima import auto_arima
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Import Stock Data
ticker = 'AAPL'  # Example: Apple Inc.
data = yf.download(ticker, start='2020-01-01', end='2024-01-01')

# Display the first few rows of the dataset
print("Data Sample:")
print(data.head())


# Step 2: Define the ARIMA-based Strategy
class ARIMAStrategy(Strategy):
    def init(self):
        self.model = None  # Placeholder for ARIMA model
        self.predictions = []  # Store predictions

    def next(self):
        # Fit the ARIMA model on the available data
        if len(self.data) < 20:  # Ensure we have enough data points
            return

        # Fit ARIMA model and make a prediction for the next day
        try:
            self.model = auto_arima(self.data['Close'][:len(self.data)], seasonal=False, stepwise=True)
            forecast = self.model.predict(n_periods=1)
            predicted_price = forecast[0]
            print(f"Predicted Price for {self.data.index[-1] + pd.Timedelta(days=1)}: {predicted_price}")

            # Buy or sell based on prediction vs current price
            if predicted_price > self.data['Close'][-1]:  # Buy signal
                self.buy()
            elif predicted_price < self.data['Close'][-1]:  # Sell signal
                self.sell()
        except Exception as e:
            print(f"Error in ARIMA modeling: {e}")
            return


# Step 3: Run the Backtest
bt = Backtest(data, ARIMAStrategy, commission=0.002, exclusive_orders=True)
stats = bt.run()
bt.plot()

# Step 4: Analyze Results
print("Backtest Statistics:")
print(stats)

# Step 5: Visualize Predictions vs Actual Prices (for last few days)
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Actual Prices', color='blue')
plt.title('Actual Prices of AAPL')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()