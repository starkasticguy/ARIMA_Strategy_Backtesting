import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from statsmodels.tsa.arima.model import ARIMA
import itertools

class IndianBusinessCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('Republic Day', month=1, day=26, observance=nearest_workday),
        Holiday('Independence Day', month=8, day=15, observance=nearest_workday),
        Holiday('Gandhi Jayanti', month=10, day=2, observance=nearest_workday),
        Holiday('Diwali', month=11, day=4, observance=nearest_workday),  # Example date
        Holiday('Holi', month=3, day=29, observance=nearest_workday),  # Example date
        # Add more holidays as per the Indian calendar
    ]

# Define a custom business day
indian_bday = CustomBusinessDay(calendar=IndianBusinessCalendar())

class DataPreprocessor:
    def preprocess(self, data):
        data = data.asfreq(indian_bday)
        data = data.interpolate(method='linear')
        data = self.arima_impute(data)
        return data

    def arima_impute(self, data):
        for column in data.columns:
            missing_dates = data[data[column].isna()].index
            for date in missing_dates:
                before_date = data[column][:date].dropna()
                if len(before_date) < 10:  # Ensure there is enough data to fit ARIMA
                    continue

                best_order = self.select_best_arima_order(before_date)
                if best_order is None:
                    continue

                model = ARIMA(before_date, order=best_order)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=1)
                data.at[date, column] = forecast[0]

        return data

    def select_best_arima_order(self, series):
        p = d = q = range(0, 3)
        pdq = list(itertools.product(p, d, q))
        best_aic = np.inf
        best_order = None

        for param in pdq:
            try:
                model = ARIMA(series, order=param)
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = param
            except:
                continue

        return best_order
