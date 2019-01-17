import pandas


def main():
    dataset = pandas.read_csv('asia_reviews.csv')

    print(type(dataset))
    print(dataset.iloc[:,2])

main()
