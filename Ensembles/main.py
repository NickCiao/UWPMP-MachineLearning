import numpy
import pandas
import sys
# Import my ID3 implementation
from ID3 import *

columns = [
    "PregnancyCount",
    "PlasmaGlucose",
    "DiastolicBP",
    "TricepsSkinFold",
    "2HrSerumInsulin",
    "BMI",
    "DiabetesPedigreeFx",
    "Age",
    "Class"]

trainD = pandas.read_csv(
    "C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning"
    "/Ensembles/Data/pima-indians-diabetes.train",
    sep=",",
    header=None,
    names=columns)
print("Training data loaded!")


decisionTree = DecisionTree(trainD, 'Class')


testD = pandas.read_csv(
    "C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning"
    "/Ensembles/Data/pima-indians-diabetes.test",
    sep=",",
    header=None,
    names=columns)
print("Test data loaded!")


print("Predicting...")
decisionTree.predict(testD)
# if __name__ == "__main__":
# do stuff
