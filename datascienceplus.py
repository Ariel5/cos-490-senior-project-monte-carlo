import numpy
import pandas
from pandas_datareader import data as pandas_datareader
import matplotlib.pyplot as pyplot
from scipy.stats import norm


# @TODO document machine-learning wouldn't help because of the huge number of test samplings needed to do
#   For each ticker you need to optimize


# Prefix to specify whether the saved file contains past stock data, prediction data, or real data to compare accuracy.
# Refer to save_to_csv argument help description
def save_as_csv(data, filename, prefix):
    csv_file = open(prefix + filename + '.csv', 'w')
    if type(data) != numpy.ndarray:
        data.to_csv(csv_file)
    else:
        numpy.savetxt(prefix + filename + '.csv', data, delimiter=',')


# Returns 2D data structure with labeled axes in the format: Date/Closing Price
def data_source(ticker, start_date, end_date, online_source, load_csv='past-stock.csv'):
    # Choose whether to get data from online source or CSV file
    if (online_source):
        # Dictionary[] because DataFrame (pandas object) can hold many different stocks,
        #   which are shown in different columns on DataReader
        data = pandas_datareader.DataReader(ticker, data_source='quandl',
                                            start=datetime(start_date.year, start_date.month, start_date.day),
                                            end=datetime(end_date.year, end_date.month, end_date.day))['Close']
    else:
        data = pandas.read_csv(load_csv, header=None, index_col=0, dtype={'a': numpy.str, 'b': numpy.float16})

    return data


# Makes Brownian Motion predictions based on past data. Refer to each line's comment for more details
def statistical_wizardry(data, days_to_predict, nr_predictions):
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

    # Depending on source of data, stochastic_drift can either be of type pandas.Series (online) or numpy.float64 (CSV).
    #   This is because CSV files are parsed directly into type float
    if type(stochastic_drift) == numpy.float64:
        # numpy.exp(x) is equivalent to e^(x) where e is Euler's number (root of natural logarithm)
        # Here because drift and std are calculated from log_returns above, which is the natural logarithm
        # norm.ppf() is the Percent Point Function, the inverse of Cumulative Distribution Function
        brownian_motion = numpy.exp(stochastic_drift + standard_deviation * brownian_random_variable)
    else:
        brownian_motion = numpy.exp(stochastic_drift.values + standard_deviation.values * brownian_random_variable)

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

    return price_prediction


# Compares prediction to reality and gives prediction accuracy (if the program isn't used to predict data into the
#   future, as then there obviously is no real-world data to compare accuracy against.
# Prediction starts from last day of past-data given as arguments to the statistics prediction function hence
#   comparison_start_date is parameter and end_date is actual argument.
def prediction_accuracy(ticker, comparison_start_date, comparison_end_date, prediction_data):
    comparison_data = data_source(ticker, comparison_start_date, comparison_end_date, True)

    mean_comparison = [prediction_data.mean(), comparison_data.mean()]
    standard_deviation_comparison = [prediction_data.std(), comparison_data.std()]

    return comparison_data, mean_comparison, standard_deviation_comparison


# Calculates difference in percentage between two numbers. This is different from Percentage Change,
#   as the order of numbers in Percentage Difference does not matter
def percentage_difference(num1, num2):
    # Round absolute number to 2 decimal places
    return round((abs(num1 - num2)/((num1 + num2)/2))*100, 2)


# Function that uses matplotlib to visualize data in a chart
def plot_chart(data, title):
    pyplot.figure(figsize=(10, 6))
    pyplot.plot(data)
    pyplot.xlabel("Nr. of days from today")
    pyplot.ylabel("NASDAQ Index")
    pyplot.title(title)
    # pyplot.ylim(40, 48)

    pyplot.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Welcome to Brownian Motion, Monte-Carlo simulation of future stock "
                                                 "price. "
                                                 "This program reads past stock price data and uses Brownian Motion "
                                                 "and Monte Carlo style of simulations to predict a future stock price. "
                                                 "!!!DISCLAIMER!!! "
                                                 "Please note that this prediction algorithm is not Wunderwaffe, "
                                                 "neither is it omniscient. The prediction is purely "
                                                 "statistics-based, and you should not base your investment decisions "
                                                 "purely on the results of this piece of software. "
                                                 "It doesn't take into account the economic, political or social "
                                                 "factors, a company's marketing, public perception, or changing "
                                                 "trends. Brownian Motion prediction is the best statistic-only "
                                                 "based prediction out there and even it can wildly vary from reality "
                                                 "since it assumes past trends will continue into the future. "
                                                 "Author: Ariel Lubonja")

    parser.add_argument('-o', '--online_source', default=False,
                        help="Switches from CSV-file reading to online-data receiving mode. "
                             "Need to specify the NASDAQ index you want. "
                             "Online data sourced from Quandl. Takes precedence over load_csv."
                             "!!!IMPORTANT!!! You should not use this option too "
                             "frequently as it is possible that the Finance API will deny you "
                             "access. The system enforces a Fair-Use policy. It might be a good "
                             "idea to download the data once, and use the \"save_to_csv\" "
                             "option to save it locally. Please enter -o True to activate. "
                             "Default: KO (Coca-Cola) from CSV for testing and performance "
                             "purposes")
    parser.add_argument('-t' '--ticker', default='KO',
                        help="Enter NASDAQ ticker. Works only with online_source. Default: KO (Coca-Cola)")
    parser.add_argument('-l' '--load_csv', default='past-stock.csv',
                        help="Enter CSV file to load data from. Default: past-stock.csv")
    parser.add_argument('-sd', '--start_date', default='2007-01-01',
                        help="Select source data start date. Only effective for online retrieval. Format YYYY-MM-DD "
                             "Default: 2007-01-01")
    parser.add_argument('-ed', '--end_date', default='2017-01-01',
                        help="Select source data end date. Only effective for online retrieval. Format YYYY-MM-DD "
                             "Default: 2017-01-01")
    parser.add_argument('-s', '--save_to_csv', help="Allows you to save real stock & prediction data to file, as well "
                                                    "as the prediction data. You can then use it to save "
                                                    "Need to specify filename. It will be preceded by past- for past "
                                                    "stock price data, real- for current data, and prediction- for "
                                                    "prediction data. "
                                                    "!!!IMPORTANT!!! You should not use the online_source option too "
                                                    "frequently as it is possible that the Finance API will deny you "
                                                    "access. The system enforces a Fair-Use policy. It might be a good "
                                                    "idea to download the data once, and use this option to save it "
                                                    "locally. "
                                                    "!!!WARNING!!! This command will create the files if they don't "
                                                    "exist, but will also delete them without prompt if they do! "
                                                    "Default: False")
    parser.add_argument('-d', '--days_to_predict', default=365,
                        help="Enter the number of days to predict in the future. "
                             "In order to show prediction accuracy, it is necessary for "
                             "this number not to exceed the current date, as then there "
                             "will be no real stock price data to compare accuracy agaist. "
                             "Default: 365")
    parser.add_argument('-n', '--number_of_predictions', default='10',
                        help="Enter the number of predictions you want to see. "
                             "Recommended < 100 to fit memory limitations. "
                             "Default: 10")
    args = vars(parser.parse_args())

    # Read command line args
    online_source = False if (args['online_source'] == None) else True
    ticker = args['t__ticker']
    load_csv = args['l__load_csv']

    # Convert string into datetime
    from datetime import datetime

    start_date = datetime.strptime(args['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(args['end_date'], '%Y-%m-%d')

    # Not used a default in add_argument because I want to use Coca-Cola for demonstration purposes.
    # I have already downloaded and saved the data in CSV in case of network error in presentation.
    save_to_csv = False if (args['save_to_csv'] == None) else args['save_to_csv']
    days_to_predict = int(args['days_to_predict'])
    number_of_predictions = int(args['number_of_predictions'])

    # Function calls

    # Get past_data in user-specified way
    past_data = data_source(ticker, start_date, end_date, online_source, load_csv)
    # Calculate price_prediction with log_returns on which to base future prediction
    price_prediction = statistical_wizardry(past_data, days_to_predict, number_of_predictions)
    # Display chart with prediction data
    plot_chart(price_prediction, ticker + " Price Prediction")


    # See if it's possible to get a prediction accuracy reading based on given parameters
    # See comment on prediction_accuracy()
    from datetime import timedelta

    comparison_end_date = end_date + timedelta(days=days_to_predict)
    # See comment on prediction_accuracy()
    is_prediction_accuracy_possible = (comparison_end_date < datetime.today())

    # Needs to be main() scoped, as it is used in the following if statement
    comparison_data = 0

    if is_prediction_accuracy_possible:
        # So prediction_accuracy() doesn't get called twice - it is both network and CPU intensive
        return_value = prediction_accuracy(ticker, end_date, comparison_end_date, price_prediction)
        comparison_data = return_value[0]
        print("Prediction accuracy:\n")
        print("Mean:\t" + "Prediction: " + str(round(return_value[1][0], 2)) + r' / ' +
              "Real: " + str(round(return_value[1][1], 2)) + r'  --  Difference: ' +
              str(percentage_difference(return_value[1][0], return_value[1][1])) + ' %')
        print("Standard Deviation:\t" + "Prediction: " + str(round(return_value[2][0], 2)) + r' / ' +
              "Real: " + str(round(return_value[2][1], 2)) + r'  --  Difference: ' +
              str(percentage_difference(return_value[2][0], return_value[2][1])) + ' %')

        plot_chart(comparison_data, ticker + " Real Data")
    else:
        print("Cannot compute prediction accuracy because you are requesting the program to compute future stock price." \
              "Real-world data is therefore not available to make this prediction.\nSkipping...")

    # Save data to CSV if user gives such parameter
    # Yes, I know this following if statement can be technically "simplified", but note that save_to_csv is not
    #   True or False - it is string or False. See args['save_to_csv']
    if save_to_csv != False:
        # If Not False, save_to_csv will store the filename
        save_as_csv(past_data, save_to_csv, 'past-')
        save_as_csv(price_prediction, save_to_csv, 'prediction-')
        if is_prediction_accuracy_possible:
            save_as_csv(comparison_data, save_to_csv, 'real-')
