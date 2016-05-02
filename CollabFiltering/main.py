import pandas
import recommender

# Open the file containing the training data
trainD = pandas.read_csv(
    "C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning/CollabFiltering"
    "/Data/TrainingRatings.txt",
    sep=",",
    header=None,
    names=["movieId", "userId", "rating"])

testD = pandas.read_csv(
    "C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning/CollabFiltering"
    "/Data/TestingRatings.txt",
    sep=",",
    header=None,
    names=["movieId", "userId", "rating"])

recommender = Recommender(trainD)
recommender.execute(testD)

# if __name__ == "__main__":
# do stuff
