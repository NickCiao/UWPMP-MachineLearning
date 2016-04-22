# Download liac-arff package with pip
import arff
# Import my ID3 implementation
from ID3 import *

# Open the file containing the training data

rawTrainD = open(
    "C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning"
    "/DecisionTrees/Data/training_subsetD.arff")

# rawTrainD = open("C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning
# /DecisionTrees"
# "/Data/training_subsetD.arff")

# Learn a decision tree using training data.
decisionTree = train(arff.load(rawTrainD))

# Open the file containing the test data
rawTestD = open(
    "C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning/DecisionTrees"
    "/Data/testingD.arff")

# Predict the testData
decisionTree.predict(arff.load(rawTestD))

# if __name__ == "__main__":
# do stuff
