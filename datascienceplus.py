# @TODO Compare accuracy of predictions by starting the prediction 100 days ago for the next 100 days, and compare it
#  to the actual stock price

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime

# Ford Motor Company
ticker = 'F'

# 2D data structure with labeled axes ex. Date/Closing Price
data = pd.DataFrame()


# Get Google stock data on Close
# Dictionary[] because DataFrame can hold many different stocks, which are shown in different columns on DataReader
data[ticker] = wb.DataReader(ticker, data_source='quandl', start=datetime(2016, 1, 1), end=datetime(2017, 1, 1))[
    'Close']

# What is logarithmic return? Return on investment. Logarithmic method used to avoid quirks of arithmetic method.
# This because in arithmetic, a stock increase of 15% then a decrease of 15% would have a net change of -2.25%
# instead of 0
# data.pct_change() returns the % change of the current day from the previous one
# log() is the natural logarithm
log_returns = np.log(1 + data.pct_change())

stochastic_drift = log_returns.mean() - (0.5 * log_returns.var())

standard_deviation = log_returns.std()

x = np.random.rand(10, 2)

z = norm.ppf(np.random.rand(10, 2))

days_to_predict = 365
nr_predictions = 5

# np.exp(x) is equivalent to e^(x) where e is Euler's number (root of natural logarithm)
# Here because drift and std are calculated from log_returns above, which is the natural logarithm
# norm.ppf() is the Percent Point Function, the inverse of Cumulative Distribution Function
daily_returns = np.exp(stochastic_drift.values + standard_deviation.values * norm.ppf(np.random.rand(days_to_predict,
                                                                                                     nr_predictions)))

# Current market price to make best possible predictions for future
S0 = data.iloc[-1]

price_list = np.zeros_like(daily_returns)

price_list[0] = S0

for t in range(1, days_to_predict):
    price_list[t] = price_list[t - 1] * daily_returns[t]

plt.figure(figsize=(10, 6))
plt.plot(price_list)
plt.xlabel("Nr. of days from today")
plt.ylabel("Price in USD")
plt.title("Price Prediction")

plt.show()

real_data = pd.DataFrame()
real_data[ticker] = wb.DataReader(ticker, data_source='quandl', start=datetime(2017, 1, 1),
                                  end=datetime(2018, 1, 1))['Close']
