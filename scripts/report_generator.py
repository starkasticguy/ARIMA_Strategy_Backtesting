import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ReportGenerator:
    def __init__(self, backtest_results, buy_signals, sell_signals, volatility, actual, predictions, future_predictions,
                 future_returns, last_5_days_prices, last_5_days_returns):
        self.backtest_results = backtest_results
        self.buy_signals = buy_signals
        self.sell_signals = sell_signals
        self.volatility = volatility
        self.actual = actual
        self.predictions = predictions
        self.future_predictions = future_predictions
        self.future_returns = future_returns
        self.last_5_days_prices = last_5_days_prices
        self.last_5_days_returns = last_5_days_returns

    def calculate_metrics(self):
        returns = self.backtest_results.pct_change().dropna()
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        annualized_return = cumulative_returns.iloc[-1] ** (252 / len(cumulative_returns)) - 1

        metrics = {
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'Annualized Return': annualized_return
        }

        return metrics

    def generate_report(self):
        if self.backtest_results.empty:
            print("No backtest results to plot.")
            return

        metrics = self.calculate_metrics()

        # Plot Cumulative Return and Volatility
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return', color=color)
        ax1.plot(self.backtest_results, color=color, label='Cumulative Return')
        ax1.scatter(self.buy_signals.index, self.backtest_results.loc[self.buy_signals.index], marker='^', color='g',
                    label='Buy Signal')
        ax1.scatter(self.sell_signals.index, self.backtest_results.loc[self.sell_signals.index], marker='v', color='r',
                    label='Sell Signal')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Volatility', color=color)
        ax2.plot(self.volatility, color=color, label='Volatility')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        plt.title('Backtest Cumulative Return with Buy/Sell Signals and Volatility')
        fig.tight_layout()
        plt.grid(True)
        plt.show()

        # Plot Real vs Predicted Closing Prices
        plt.figure(figsize=(12, 6))
        plt.plot(self.actual, label='Actual')
        plt.plot(self.predictions, label='Predicted')
        plt.plot(self.future_predictions, label='Future Predictions', color='purple')
        plt.title('Real vs Predicted Closing Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Returns Timeseries
        actual_returns = self.actual.pct_change().dropna()
        predicted_returns = self.predictions.pct_change().dropna()
        plt.figure(figsize=(12, 6))
        plt.plot(actual_returns, label='Actual Returns')
        plt.plot(predicted_returns, label='Predicted Returns')
        plt.plot(self.future_predictions.pct_change().dropna(), label='Future Returns', color='brown')
        plt.title('Returns Timeseries')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print Metrics
        print("Performance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")

        # Print Last 5 Days Prices and Returns
        last_5_days_data = pd.DataFrame({
            'Last 5 Days Prices': self.last_5_days_prices,
            'Last 5 Days Returns': self.last_5_days_returns
        })
        print("\nLast 5 Days Prices and Returns:")
        print(last_5_days_data)

        # Print Future Predictions and Returns
        future_data = pd.DataFrame({
            'Future Predictions': self.future_predictions,
            'Future Returns': self.future_returns
        })
        print("\nFuture 5 Days Predictions and Returns:")
        print(future_data)

        return metrics
