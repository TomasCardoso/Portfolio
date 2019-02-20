import pandas as pd
import numpy as np

def main():


    ######################## Load csv file as pandas dataframe ################

    df = pd.read_csv('data/train.csv')

    ######## Get house prices and remove respective column ####################
    price = df['SalePrice'].values
    price = price.T
    df = df.drop(columns = 'SalePrice')

    ################## One hot encode categorical columns #####################

    for column in df:
        if 'object' in str(df[column].dtype):
            #Gets the one hot encoding
            hot_encoding = pd.get_dummies(df[column], prefix = column, dummy_na = True)
            #Drops the categorical column
            df = df.drop(columns = column)
            #Concatenates one hot encoded columns to dataframe
            df = pd.concat([df, hot_encoding], axis = 1)

    # Convert all entries in dataframe to float64
    df = df.astype(float)

    ########## Convert dataframe to numpy array and drop id values ############
    record = np.array(df.to_records())

    features = np.zeros(shape = [len(record), len(record[0])-1])

    for i in range(len(record)):
        for j in range(len(record[0])-1):
            features[i,j] = record[i][j+1]


    #Ok so right now there's an array with the prices and and an array with the features



main()
