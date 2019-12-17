from pandas import read_csv, DataFrame
from datetime import datetime
from matplotlib import pyplot
from pandas import concat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate, sqrt

def parser(date):
	return datetime.strptime(date, '%Y %m %d %H')

def load_data():

	# Argument parse_dates : bool or list of int or names or list of lists or dict, default: False
	# The behavior is as follows:
	#		boolean. If True -> try parsing the index.
	#		list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3 each as a separate date column.
	#		list of lists. e.g. If [[1, 3]] -> combine columns 1 and 3 and parse as a single date column.
	#		dict, e.g. {‘foo’ : [1, 3]} -> parse columns 1, 3 as date and call result ‘foo’

	# This concatenated string forming a date is passed to function specified in argument date_parser
	dataset = read_csv('data/raw.csv', parse_dates = [['year', 'month', 'day', 'hour']], index_col = 0, date_parser = parser)
	print(dataset.head(5))
	# Right now, the column 'No' serves as dataset index, we're gonna change it so the index is the new date field
	# First, the column 'No' is dropped.
	dataset.drop('No', axis=1, inplace=True)

	# manually specify column names
	dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
	# make column date the new index
	dataset.index.name = 'date'

	# mark all NA values with 0
	dataset['pollution'].fillna(0, inplace=True)
	# drop the first 24 hours
	dataset = dataset[24:]
	# summarize first 5 rows
	print(dataset.head(5))
	# save clean dataset to file
	dataset.to_csv('data/pollution.csv')

"""
Frame a time series as a supervised learning dataset.
Arguments:
	data: Sequence of observations as a list or NumPy array.
	n_in: Number of lag observations as input (X).
	n_out: Number of observations as output (y).
	dropnan: Boolean whether or not to drop rows with NaN values.
Returns:
	Pandas DataFrame of series framed for supervised learning.
"""
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

	# Gets number of columns in original dataset
	n_vars = 1 if type(data) is list else data.shape[1]
	
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)

	return agg

def plot_features(dataset):

	# Get dataframe as Numpy array
	values = dataset.values
	# specify columns to plot
	groups = [0, 1, 2, 3, 5, 6, 7]
	i = 1
	# Plot each column as subplot
	pyplot.figure()
	for group in groups:
		pyplot.subplot(len(groups), 1, i)
		pyplot.plot(values[:, group])
		pyplot.title(dataset.columns[group], y=0.5, loc='right')
		i += 1
	pyplot.show()

def feature_transformation(dataset):

	values = dataset.values
	# Encode wind direction
	encoder = LabelEncoder()
	values[:,4] = encoder.fit_transform(values[:,4])
	# ensure all data is float
	values = values.astype('float32')
	# normalize features
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)
	# frame as supervised learning
	reframed = series_to_supervised(scaled, 1, 1)
	# drop columns we don't want to predict
	reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
	print(reframed.head())

	return reframed

def model_training(train_X, train_y, test_X, test_y):

	# design network
	model = Sequential()
	model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mae', optimizer='adam')
	# fit network
	history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
	# plot history
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	pyplot.show()

	return model


def main():
	# load dataset
	dataset = read_csv('data/pollution.csv', header=0, index_col=0)
	
	# Change to true if it is the first execution
	firstExecution = False

	if firstExecution:
		# Call data loading function
		load_data()
	else:
		# Load processed dataset
		dataset = read_csv('data/pollution.csv', header=0, index_col=0)

	# Plot features
	#plot_features(dataset)

	values = dataset.values
	# integer encode direction
	encoder = LabelEncoder()
	values[:,4] = encoder.fit_transform(values[:,4])
	# ensure all data is float
	values = values.astype('float32')
	# normalize features
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)
	# frame as supervised learning
	reframed = series_to_supervised(scaled, 1, 1)
	# drop columns we don't want to predict
	reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
	print(reframed.head())
	 
	# split into train and test sets
	values = reframed.values
	n_train_hours = 365 * 24
	train = values[:n_train_hours, :]
	test = values[n_train_hours:, :]
	# split into input and outputs
	train_X, train_y = train[:, :-1], train[:, -1]
	test_X, test_y = test[:, :-1], test[:, -1]
	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

	model = model_training(train_X, train_y, test_X, test_y)
	 
	# make a prediction
	yhat = model.predict(test_X)
	test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
	# invert scaling for forecast
	inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]
	# invert scaling for actual
	test_y = test_y.reshape((len(test_y), 1))
	inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0]
	# calculate RMSE
	rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
	print('Test RMSE: %.3f' % rmse)
	
main()