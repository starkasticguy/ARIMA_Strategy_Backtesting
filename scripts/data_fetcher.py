import yfinance as yf

class StockDataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker

    def fetch_data(self, start_date, end_date):
        stock = yf.download(self.ticker, start=start_date, end=end_date)
        return stock[['Open', 'High', 'Low', 'Close']]
