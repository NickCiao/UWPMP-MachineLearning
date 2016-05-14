import pandas
import numpy
from nbspam import *

def createDF(fileName):
    with open(fileName) as f:
        content = f.readlines()

    d = []
    for line in content:
        Id = ""
        Class = ""
        Word = ""
        for i, val in enumerate(line.split()):
            if i == 0:
                Id = val
            elif i == 1:
                Class = val
            elif i % 2 == 0:
                Word = val
            else:
                d.append({
                    'Id':Id,
                    'Class':Class,
                    'Word':Word,
                    'Count':val})

    return pandas.DataFrame(d)

trainD = createDF("C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning"
    "/NaiveBayes/Data/train.txt")
print("training set loaded...")

testD = createDF("C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning"
    "/NaiveBayes/Data/test.txt")
print("test set loaded...")

spamFilter = Naive(trainD)
recommender.execute(testD)




# if __name__ == "__main__":
# do stuff
