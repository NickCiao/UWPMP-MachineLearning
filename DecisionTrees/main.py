# Download liac-arff package with pip
import arff
import collections
import numpy
import pandas
import sys
# Import my ID3 implementation
from ID3 import *

print("loading training data...")
rawTrainD = open(
    "C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning"
    "/DecisionTrees/Data/training_subsetD.arff")

arffData = arff.load(rawTrainD)
attributeDict = collections.OrderedDict(arffData['attributes'])
attributeNames = list(attributeDict.keys())

df = pandas.DataFrame(
    numpy.array(arffData['data']),
    columns=attributeNames)
print("finished loading training data!")


print("Training Decision tree...")
decisionTree = DecisionTree(df, 'Class')
print("Finished training decision tree!")


print("loading test data...")
rawTestD = open(
    "C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning/DecisionTrees"
    "/Data/testingD.arff")

arffData = arff.load(rawTestD)
testData = pandas.DataFrame(
            numpy.array(arffData['data']),
            columns=attributeNames)
print("Finished loading test data!")


print("Predicting...")
decisionTree.predict(arff.load(rawTestD))
# if __name__ == "__main__":
# do stuff
