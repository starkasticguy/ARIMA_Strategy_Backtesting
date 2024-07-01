import pandas as pd  # Import pandas
from pandas.tseries.offsets import CustomBusinessDay
from scripts.data_fetcher import StockDataFetcher
from scripts.data_preprocessor import DataPreprocessor, IndianBusinessCalendar
from scripts.time_series_analyzer import TimeSeriesAnalyzer
from scripts.model_selector import OrderSelector
from scripts.rolling_predictor import RollingPredictor
from scripts.trading_strategy import TradingStrategy
from scripts.report_generator import ReportGenerator
import matplotlib.pyplot as plt

# Define a custom business day
indian_bday = CustomBusinessDay(calendar=IndianBusinessCalendar())

def main(ticker, start_date, end_date, initial_window_size=100):  # Reduced window size
    # Fetch data
    fetcher = StockDataFetcher(ticker)
    data = fetcher.fetch_data(start_date, end_date)

    # Preprocess data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(data)

    print(f"Length of processed data: {len(processed_data)}")

    # Analyze time series
    analyzer = TimeSeriesAnalyzer(processed_data)
    analyzer.plot_decomposition()
    is_stationary = analyzer.check_stationarity()
    analyzer.plot_acf_pacf()

    # Select model order
    order_selector = OrderSelector(processed_data)
    best_orders, best_models = order_selector.select_order()

    print(f"Best ARIMA orders: {best_orders}")
    for key, model in best_models.items():
        print(f"\n{key} Model Summary:")
        print(model.summary())

    # Rolling prediction
    predictor = RollingPredictor(processed_data, best_orders, initial_window_size)
    predictions = predictor.rolling_forecast()
    future_predictions = predictor.future_forecast(steps=5)

    print(f"Number of predictions: {len(predictions['Close'])}")  # Example for Close prices
    print(predictions)
    print(f"Future predictions: {future_predictions}")

    # Trading strategy
    if len(predictions) == 0:
        print("No predictions made. Exiting.")
        return

    strategy = TradingStrategy(predictions, processed_data[initial_window_size:], future_predictions)
    cumulative_return = strategy.backtest()
    future_dod_change = strategy.future_return()
    last_5_days_prices, last_5_days_dod_change = strategy.last_5_days()

    print(f"Cumulative return length: {len(cumulative_return['Close'])}")  # Example for Close prices
    print(cumulative_return)

    # Generate report
    report = ReportGenerator(
        cumulative_return,
        strategy.buy_signals,
        strategy.sell_signals,
        strategy.volatility,
        processed_data[initial_window_size:],
        {k: pd.Series(v, index=processed_data[initial_window_size:].index) for k, v in predictions.items()},
        strategy.future_predictions,
        future_dod_change,
        last_5_days_prices,
        last_5_days_dod_change
    )
    metrics = report.generate_report()

    print("Generated Report:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    ticker = "KOTAKBANK.NS"
    start_date = "2024-01-01"
    end_date = "2024-07-01"

    main(ticker, start_date, end_date)
