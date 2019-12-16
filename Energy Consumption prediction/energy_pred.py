import pandas as pd
#import vaex as pd
#from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datetime import datetime, date
from numpy import sin, cos, pi
import numpy as np
import xgboost as xgb
#import time
import pickle
#import calendar
from matplotlib import pyplot


# Global variable enconding
#start = time.time()
# Variable creation for season computing. 
Y = 2000
seasons = [('0', (date(Y,  1,  1),  date(Y,  3, 20))), # Winter
	           ('1', (date(Y,  3, 21),  date(Y,  6, 20))), # Spring
	           ('2', (date(Y,  6, 21),  date(Y,  9, 22))), # Summer
	           ('3', (date(Y,  9, 23),  date(Y, 12, 20))), # Autumn
	           ('0', (date(Y, 12, 21),  date(Y, 12, 31)))] # Winter

# Function for data fetching. Works for both train and test data (specified by data_type argument)
def fetch_data(data_type, encode_categorical_data = True):

	data_path = 'data/' + data_type + '.csv'
	weather_path = 'data/weather_' + data_type + '.csv'

	building_metadata = pd.read_csv('data/building_metadata.csv')
	weather = pd.read_csv(weather_path)
	data = pd.read_csv(data_path, nrows = 10000000)

	data = pd.merge(data, building_metadata, on='building_id')
	data = pd.merge(data, weather, left_on=['site_id', 'timestamp'], right_on=['site_id', 'timestamp'], how='left')

	if encode_categorical_data:
		data = encode_timestamp(data)
		data = encode_categorical(data)

	return data

# Performs variable enconding for categorical fields. 
# One hot enconding is done by default
def encode_categorical(data, enconding='one_hot_enconding'):

	if enconding == 'label_encoding':
		le = preprocessing.LabelEncoder()
		le.fit(data['primary_use'])

		encoded = le.transform(data['primary_use'])

		data = data.drop('primary_use', axis = 1)

		data['primary_use'] = encoded

	elif enconding == 'one_hot_enconding':
		encoded_data = pd.get_dummies(data['primary_use'], drop_first = False)
		data = data.drop('primary_use', axis = 1)
		data = pd.concat([data, encoded_data], axis=1)

	return data

# Get season of datetime variable
def get_season(date_var):

	date_var = date(date_var.year, date_var.month, date_var.day)
	date_var = date_var.replace(year=Y)

	return next(season for season, (start, end) in seasons 
		if start <= date_var <= end)

# Feature transformation and creation from timestamp.
# Features created:
# 			- sinHoD & cosHoD: cyclical encoding of hour of day
#			- sinDoW & cosDoW: cyclical encoding of day of week
#			- sinDoM & cosDoM: cyclical encoding of day of month
#			- sin_season & cos_season: cyclical encoding of season of the year
def encode_timestamp(data):
	
	sinHoD = []
	cosHoD = []
	sinDoM = []
	cosDoM = []
	sinDoW = []
	cosDoW = []
	sin_season = []
	cos_season = []

	days_in_month = 31
	hours_in_day = 24
	days_in_week = 7
	seasons_in_year = 4

	for i in range(len(data)):

		d = datetime.strptime(data['timestamp'][i], '%Y-%m-%d %H:%M:%S')

		sinHoD.append(sin(2*pi*d.hour/hours_in_day))
		cosHoD.append(cos(2*pi*d.hour/hours_in_day))
		sinDoM.append(sin(2*pi*d.day/days_in_month))
		cosDoM.append(cos(2*pi*d.day/days_in_month))
		sinDoW.append(sin(2*pi*d.weekday()/days_in_week))
		cosDoW.append(cos(2*pi*d.weekday()/days_in_week))
		sin_season.append(sin(2*pi*float(get_season(d))/seasons_in_year))
		cos_season.append(cos(2*pi*float(get_season(d))/seasons_in_year))

	data['sinHoD'] = sinHoD
	data['cosHoD'] = cosHoD
	data['sinDoM'] = sinDoM
	data['cosDoM'] = cosDoM
	data['sinDoW'] = sinDoW
	data['cosDoW'] = cosDoW
	data['sin_season'] = sin_season
	data['cos_season'] = cos_season

	data = data.drop('timestamp', axis = 1)

	return data

def grid_search():
	depth_list = [35]
	n_trees_list = [30]
	learning_rate_list = [0.1, 0.3, 0.5, 0.8]

	X = fetch_data('train')
	y = X['meter_reading']
	X = X.drop(['meter_reading'], axis = 1)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	for n_trees in n_trees_list:
		for depth in depth_list:
			for lr in learning_rate_list:
				model = xgb.XGBRegressor(objective ='reg:squarederror', n_jobs = 5, max_depth = depth, n_estimators = n_trees, learning_rate=lr, verbosity=1)
				model.fit(X_train, y_train)
				y_pred = model.predict(X_test)
				score = rmsle(y_test, y_pred)
				print('Test: Score for depth = ' + str(depth) + '; n_trees = ' + str(n_trees) + '; RMSLE = ' + str(score))
				y_pred = model.predict(X_train)
				score = rmsle(y_train, y_pred)
				print('Train: Score for depth = ' + str(depth) + '; n_trees = ' + str(n_trees) + '; RMSLE = ' + str(score))


# XGBoost training function
def model_training():

	train = fetch_data('train')

	model = xgb.XGBRegressor(objective ='reg:squarederror', n_jobs = 5, max_depth = 30, n_estimators = 50, verbosity=2)

	print('Starting model training...')
	
	y = train['meter_reading']
	X = train.drop(['meter_reading'], axis = 1)
	model.fit(X, y)

	print('Done')

	model.save_model('model/xgboost_complete.model')

# Compares data schema used in trained and enforces test data schema to be of the same form
# Important function if the predictions are computed iteratively 
def complete_schema(base, current):

	base_columns = list(base.columns.values)
	current_columns = list(current.columns.values)
	for column_name in base_columns:
		if column_name not in current_columns:
			current[column_name] = np.zeros((len(current)))

	current = current.reindex(columns=base_columns)

	return current

# Computes predictions
# Currently doing iterative prediction (test set too big to fit in memory)
def compute_predictions(train):

	# Variable declaration
	data_type = 'test'
	predictions = pd.DataFrame(columns = ['row_id', 'meter_reading'])
	train = train.drop(['meter_reading', 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], axis = 1)

	# Building and weather data load
	data_path = 'data/' + data_type + '.csv'
	weather_path = 'data/weather_' + data_type + '.csv'

	building_metadata = pd.read_csv('data/building_metadata.csv')
	weather = pd.read_csv(weather_path)

	# Model import
	model = xgb.XGBRegressor(objective ='reg:squarederror', n_jobs = 5, max_depth = 30, n_estimators = 50, verbosity=2)
	model.load_model('model/xgboost_complete.model')

	print('Starting prediction...')

	# Iterative data load and prediction
	for test in pd.read_csv('data/test.csv', chunksize=5000000):
		
		test = pd.merge(test, building_metadata, on='building_id')
		test = pd.merge(test, weather, left_on=['site_id', 'timestamp'], right_on=['site_id', 'timestamp'], how='left')

		test = encode_timestamp(test)
		test = encode_categorical(test)
		test = test.sort_values('row_id')

		row_id = test['row_id']
		test = test.drop(['row_id'], axis = 1)

		test = complete_schema(train, test)
		
		y_pred = model.predict(test)

		test['meter_reading'] = y_pred
		test['row_id'] = row_id
		new_predictions = test[['row_id', 'meter_reading']]

		predictions = pd.concat([predictions, new_predictions])

	print('Done')

	return predictions

#Calculate feature importance using XGBoost and plots it
def feature_importance(model):

	xgb.plot_importance(model)
	pyplot.show()

# Compute rmsle of predicted samples
def rmsle(real, predicted):
	sum = 0.0
	real = real.to_list()

	for x in range(len(predicted)):
		if predicted[x] < 0 or real[x] < 0:
			continue
		p = np.log(predicted[x]+1)
		r = np.log(real[x]+1)
		sum = sum + (p - r)**2

	return (sum/len(predicted))**0.5

def main():

	#print('Loading data...')
	#train = load_data()
	#train.to_csv('complete_training_set.csv')
	train_schema = pd.read_csv('complete_training_set.csv', nrows=10)
	# Calculate feature importance
	#feature_importance(model)
	#model_training()
	predictions = compute_predictions(train_schema)
	predictions.to_csv('submission.csv', index=False)
	#grid_search()

main()