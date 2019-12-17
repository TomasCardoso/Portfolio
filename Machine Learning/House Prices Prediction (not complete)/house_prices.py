import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import math
import csv

class tuning_result:

    def __init__(self, no_of_trees, rmse):
        self.no_of_trees = no_of_trees
        self.rmse = rmse

def getData(path, test = False):

    ######################## Load csv file as pandas dataframe ################

    df = pd.read_csv(path)

    if not test:
        ######## Get house prices and remove respective column ####################
        price = df['SalePrice'].values
        price = price.T
        df = df.drop(columns = 'SalePrice')
    else:
        price = []

    ################## One hot encode categorical columns #####################

    for column in df:
        if 'object' in str(df[column].dtype):
            #Gets the one hot encoding
            hot_encoding = pd.get_dummies(df[column], prefix = column, dummy_na = True)
            #Drops the categorical column
            df = df.drop(columns = column)
            #Concatenates one hot encoded columns to dataframe
            df = pd.concat([df, hot_encoding], axis = 1)

    df = df.astype(float) # Convert all entries in dataframe to float64

    df = df.fillna(0)


    ########## Convert dataframe to numpy array and drop id values ############
    record = np.array(df.to_records())

    features = np.zeros(shape = [len(record), len(record[0])-1])

    for i in range(len(record)):
        for j in range(len(record[0])-1):
            features[i,j] = record[i][j+1]

    return features, price

def enforceSameness():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    count = 0

    for column in test:
        if test[column].nunique() == train[column].nunique():
            count += 1

    print(count)

    return train, test

def calc_rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())

def predict(clf, X, y):

    prices = clf.predict(X)

    rmse = calc_rmse(np.log(prices), np.log(y))

    return rmse, prices

def getBestParameters(tuning_results):

    tuning_results.sort(key=lambda tuning_results: tuning_results.rmse)

    return tuning_results[0].no_of_trees

def main():

    ##################### Get train data ###############################

    features, price = getData('data/train.csv')

    ##################### Cross-validation #############################

    no_of_trees = [100]
    max_depths = [2, 4, 6]

    if True:
        eval_fold_fraction = int(np.floor((len(features) / 5)))
        no_of_folds = 5
        train_fold = np.empty(shape = [0, len(features[0])])
        train_fold_prices = []
        tuning_results = []
        for i in range(no_of_folds):

            inc = 0
            print('Starting eval fold ', i)

            folds = np.split(features, no_of_folds)
            fold_prices = np.split(price, no_of_folds)

            for j in range(no_of_folds):
                if j != i:
                    train_fold = np.concatenate((train_fold, folds[j]))
                    train_fold_prices = np.concatenate((train_fold_prices, fold_prices[j]))

            eval_fold = folds[i]
            eval_fold_prices = fold_prices[i]

            for no in no_of_trees:

                rf = RandomForestRegressor(random_state=0, n_estimators=no)
                rf.fit(train_fold, train_fold_prices)

                rmse, predictions = predict(rf, eval_fold, eval_fold_prices)

                print('For ', no,' trees the rmse is ', rmse)

                if i == 0:
                    result = tuning_result(no, rmse)
                    tuning_results.append(result)
                else:
                    tuning_results[inc].rmse = tuning_results[inc].rmse + rmse
                    inc += 1

        for j in range(len(tuning_results)):
            tuning_results[j].rmse = tuning_results[j].rmse / no_of_folds

        best_no_of_trees = getBestParameters(tuning_results)
    else:
        best_no_of_trees = 100

    rf = RandomForestRegressor(random_state=0, n_estimators=best_no_of_trees)
    rf.fit(features, price)

    test_set, test_set_prices = getData('data/test.csv', True)

    junk = np.zeros(shape=[len(test_set), len(features[0])-len(test_set[0])])

    test_set = np.concatenate((test_set, junk), axis=1)

    predictions = rf.predict(test_set)

    with open('test_result.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'SalePrice'])
        for i in range(len(predictions)):
            writer.writerow([(1461+i), predictions[i]])



main()
