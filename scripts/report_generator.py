import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ReportGenerator:
    def __init__(self, backtest_results, buy_signals, sell_signals, volatility, actual, predictions, future_predictions,
                 future_dod_change, last_5_days_prices, last_5_days_dod_change):
        self.backtest_results = backtest_results
        self.buy_signals = buy_signals
        self.sell_signals = sell_signals
        self.volatility = volatility
        self.actual = actual
        self.predictions = predictions
        self.future_predictions = future_predictions
        self.future_dod_change = future_dod_change
        self.last_5_days_prices = last_5_days_prices
        self.last_5_days_dod_change = last_5_days_dod_change

    def calculate_metrics(self):
        metrics = {}
        for key in self.backtest_results.keys():
            returns = self.backtest_results[key].pct_change(fill_method=None).dropna()
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            cumulative_returns = (1 + returns).cumprod()
            max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
            annualized_return = cumulative_returns.iloc[-1] ** (252 / len(cumulative_returns)) - 1

            metrics[key] = {
                'Sharpe Ratio': sharpe_ratio,
                'Maximum Drawdown': max_drawdown,
                'Annualized Return': annualized_return
            }

        return metrics

    def generate_report(self):
        if not self.backtest_results:
            print("No backtest results to plot.")
            return

        metrics = self.calculate_metrics()

        for key in self.backtest_results.keys():
            # Plot Cumulative Return and Volatility
            fig, ax1 = plt.subplots(figsize=(12, 6))

            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Cumulative Return', color=color)
            ax1.plot(self.backtest_results[key], color=color, label='Cumulative Return')
            ax1.scatter(self.buy_signals[key].index, self.backtest_results[key].loc[self.buy_signals[key].index],
                        marker='^', color='g', label='Buy Signal')
            ax1.scatter(self.sell_signals[key].index, self.backtest_results[key].loc[self.sell_signals[key].index],
                        marker='v', color='r', label='Sell Signal')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Volatility', color=color)
            ax2.plot(self.volatility[key], color=color, label='Volatility')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.legend(loc='upper right')

            plt.title(f'Backtest Cumulative Return with Buy/Sell Signals and Volatility for {key}')
            fig.tight_layout()
            plt.grid(True)
            plt.show()

            # Plot Real vs Predicted Closing Prices
            plt.figure(figsize=(12, 6))
            plt.plot(self.actual[key], label='Actual')
            plt.plot(self.predictions[key], label='Predicted')
            plt.plot(self.future_predictions[key], label='Future Predictions', color='purple')
            plt.title(f'Real vs Predicted {key} Prices')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot Returns Timeseries
            actual_returns = self.actual[key].pct_change(fill_method=None).dropna()
            predicted_returns = self.predictions[key].pct_change(fill_method=None).dropna()
            plt.figure(figsize=(12, 6))
            plt.plot(actual_returns, label='Actual Returns')
            plt.plot(predicted_returns, label='Predicted Returns')
            plt.plot(self.future_dod_change[key], label='Future Returns', color='brown')
            plt.title(f'Returns Timeseries for {key}')
            plt.xlabel('Date')
            plt.ylabel('Returns')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Print Metrics
            print(f"Performance Metrics for {key}:")
            for metric_key, value in metrics[key].items():
                print(f"{metric_key}: {value:.2f}")

            # Print Last 5 Days Prices and DoD Change
            last_5_days_data = pd.DataFrame({
                f'Last 5 Days {key} Prices': self.last_5_days_prices[key],
                f'Last 5 Days {key} DoD Change': self.last_5_days_dod_change[key]
            })
            print(f"\nLast 5 Days {key} Prices and DoD Change:")
            print(last_5_days_data)

            # Print Future Predictions and DoD Change
            future_data = pd.DataFrame({
                f'Future {key} Predictions': self.future_predictions[key],
                f'Future {key} DoD Change': self.future_dod_change[key]
            })
            print(f"\nFuture 5 Days {key} Predictions and DoD Change:")
            print(future_data)

        return metrics
