import pandas


def main():
    dataset = pandas.read_csv('asia_reviews.csv')

    print(type(dataset))
    print(dataset.iloc[:,2])

    attributes = {'Category 1' : 'Art Galleries',
                  'Category 2' : 'Dance Clubs',
                  'Category 3' : 'Juice Bars',
                  'Category 4' : 'Restaurants',
                  'Category 5' : 'Museums'}

main()
