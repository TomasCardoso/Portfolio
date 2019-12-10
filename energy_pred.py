import pandas as pd
#import vaex as pd
from sklearn import preprocessing
from datetime import datetime, date
from numpy import sin, cos, pi
from numpy import loadtxt
from xgboost import XGBClassifier
import time
import calendar
#from matplotlib import pyplot


# Global variable enconding
start = time.time()
# Variable creation for season computing. 
Y = 2000
seasons = [('0', (date(Y,  1,  1),  date(Y,  3, 20))), # Winter
	           ('1', (date(Y,  3, 21),  date(Y,  6, 20))), # Spring
	           ('2', (date(Y,  6, 21),  date(Y,  9, 22))), # Summer
	           ('3', (date(Y,  9, 23),  date(Y, 12, 20))), # Autumn
	           ('0', (date(Y, 12, 21),  date(Y, 12, 31)))] # Winter

def load_data(encode_categorical_data = True):

	building_metadata = pd.read_csv('data/building_metadata.csv')
	train  = pd.read_csv('data/train.csv')
	weather_train = pd.read_csv('data/weather_train.csv')
	train = pd.merge(train, building_metadata, on='building_id')
	building_metadata = None
	train = pd.merge(train, weather_train, left_on=['site_id', 'timestamp'], right_on=['site_id', 'timestamp'], how='left')
	weather_train = None
	print('Finished loading data. Time elapsed: ' + str(time.time() - start) + ' s')

	#weather_test = pd.read_csv('data/weather_test.csv')
	#test  = pd.read_csv('data/train.csv')

	if encode_categorical_data:
		train = encode_timestamp(train)
		train = encode_categorical(train)
		print('Finished encoding data. Time elapsed: ' + str(time.time() - start) + ' s')
	return train

# Performs variable enconding for categorical fields. 
# One hot enconding is done by default
def encode_categorical(data, enconding='one_hot_enconding'):

	if enconding == 'label_encoding':
		le = preprocessing.LabelEncoder()
		le.fit(data['primary_use'])

		encoded = le.transform(data['primary_use'])

		data.drop('primary_use', axis = 1)

		data['primary_use'] = encoded

	elif enconding == 'one_hot_enconding':
		encoded_data = pd.get_dummies(data['primary_use'], drop_first = False)
		data.drop('primary_use', axis = 1)
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

	data.drop('timestamp', axis = 1)

	return data

def feature_importance(data):
	y = data['meter_reading']
	X = data
	X.drop('meter_reading', axis = 1)

	model = XGBClassifier()

	for i in range(5):
		init = int((i)*(len(X)/5))
		end = int((i+1)*(len(X)/5))
		model.fit(X[init:end], y[init:end])

	#model.fit(X, y)
	print(X.info())
	print(model.feature_importances_)

def main():

	train = load_data()
	print('Saving data...' + str(time.time() - start) + ' s')
	train.to_csv('complete_training_set.csv')
	print('Done' + str(time.time() - start) + ' s')	
	#train = pd.read_csv('complete_training_set.csv')
	#feature_importance(train)
	
	
main()