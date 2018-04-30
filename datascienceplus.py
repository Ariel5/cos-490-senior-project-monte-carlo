import numpy
import pandas
from pandas_datareader import data as pandas_datareader
import matplotlib.pyplot as pyplot
from scipy.stats import norm
from datetime import datetime

# Ford Motor Company
ticker = 'F'

days_to_predict = 365
nr_predictions = 5

# 2D data structure with labeled axes ex. Date/Closing Price
# data = pandas.DataFrame()

# @TODO implement argparse to select CSV vs quandl data source
# Dictionary[] because DataFrame can hold many different stocks, which are shown in different columns on DataReader
# data[ticker] = pandas_datareader.DataReader(ticker, data_source='quandl', start=datetime(2007, 1, 1), end=datetime(2017, 1, 1))[
#     'Close']

# @TODO implement save to disk
# past_stock = open('past-stock.csv', 'w')
# data[ticker].to_csv(past_stock)

data = pandas.read_csv('past-stock.csv', header=None, index_col=0, dtype={'a':numpy.str, 'b': numpy.float16})

# What is logarithmic return? Return on investment. Logarithmic method used to avoid quirks of arithmetic method.
# This because in arithmetic, a stock increase of 15% then a decrease of 15% would have a net change of -2.25%
# instead of 0
# data.pct_change() returns the % change of the current day from the previous one
# log() is the natural logarithm
log_returns = numpy.log(1 + data.pct_change())

# stochastic_drift = log_returns.mean() - (0.5 * log_returns.var())
# For some reason needs to be converted to numpy.array. Is: Series
# Best apporximation for future log return
stochastic_drift = log_returns.mean() - (0.5 * log_returns.var())

# The higher the s_d the harder the prediction
standard_deviation = log_returns.std()

# So far nothing is random - all data is gained from the given info
# We have all the required info for the Brownian motion

# PPF - Inverse Cumulative Distribution Function
# CDF - probability that X is smaller or equal to some value x P(X<=x) = F(x)
# Inverse CDF tell you what value of x would return a specific probability value p
# F^(-1)(p) = x
# std_dev distance between the event and the mean
# Returns an array of random numbers with dimensions given in rand(), which will "seed" the daily_returns
brownian_random_variable = norm.ppf(numpy.random.rand(days_to_predict, nr_predictions))




# numpy.exp(x) is equivalent to e^(x) where e is Euler's number (root of natural logarithm)
# Here because drift and std are calculated from log_returns above, which is the natural logarithm
# norm.ppf() is the Percent Point Function, the inverse of Cumulative Distribution Function
brownian_motion = numpy.exp(stochastic_drift.values + standard_deviation.values * brownian_random_variable)

# @TODO delete
# daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(t_intervals, iterations)))

# Returns last-day's stock price from the data given
# iloc means item at location, 0 means first
# @Important Dependent on Stock Price API. For example, Yahoo! Finance sorts oldest to newest / Quandl is the opposite
latest_price_from_data = data.iloc[0]

# Returns an array with same dimensions as the argument
# Price list should to have same dimensions as the Brownian log returns we predicted
# It cannot be larger as then there are no prediction values to indicate the price
# It can be shorter, but then you are wasting computing power on a longer Brownian Motion calculation that
#   you don't need
price_prediction = numpy.zeros_like(brownian_motion)

# Start prediction from where the real (past) data left off
# Note that price_list[0] is an entire row with (nr_iterations) nr. of columns
price_prediction[0] = latest_price_from_data

# Returns the full price_list with our predictions
# Populates the rest of the price_predictions array by multiplying the previous entry with the Brownian Motion
#    i.e. the daily log_return predicted for the future
for t in range(1, days_to_predict):
    price_prediction[t] = price_prediction[t - 1] * brownian_motion[t]

# Code to plot the price_prediction list
pyplot.figure(figsize=(10, 6))
pyplot.plot(price_prediction)
pyplot.xlabel("Nr. of days from today")
pyplot.ylabel("Price in USD")
pyplot.title("Price Prediction")
pyplot.ylim(10, 13.5)

pyplot.show()

# @TODO implement argparse to select CSV vs quandl data source
# Real Stock Price
# real_data = pandas.DataFrame()
# real_data[ticker] = pandas_datareader.DataReader(ticker, data_source='quandl', start=datetime(2017, 1, 1),
#                                   end=datetime(2018, 1, 1))['Close']
#
# real_data[ticker].to_csv(
#     open('real-stock.csv', 'w')
# )

# @TODO implement % difference between prediction and reality
# @TODO multiprocess
real_data = pandas.read_csv('real-stock.csv', header=None, index_col=0, dtype={'a':numpy.str, 'b': numpy.float16})

# Code to plot the price_prediction list
pyplot.figure(figsize=(10, 6))
pyplot.plot(real_data)
pyplot.xlabel("Nr. of days from today")
pyplot.ylabel("Price in USD")
pyplot.title("Real Stock Price")
pyplot.ylim(10, 13.5)

pyplot.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Welcome to Brownian Motion, Monte-Carlo simulation of future stock price.\nAuthor: Ariel Lubonja")

    parser.add_argument('-o', '--online-source', help="Switches from CSV-file reading to online-data receiving mode. "
                                                      "Default: CSV for testing and performance purposes")
