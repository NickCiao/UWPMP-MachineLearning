import pandas
import numpy
from nbspam import *

trainD = createDF("C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning"
    "/NaiveBayes/Data/train.txt")
print("training set loaded...")

testD = createDF("C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning"
    "/NaiveBayes/Data/test.txt")
print("test set loaded...")

spamFilter = NaiveBayesSpamFilter(trainD)
spamFilter.execute(testD)

# if __name__ == "__main__":
# do stuff
