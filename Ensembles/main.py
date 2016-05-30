import numpy
import pandas
import sys
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from scipy import stats
import bvd

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

testD = pandas.read_csv(
    "C:/Users/Nicholas/Documents/Repos/UWPMP-MachineLearning"
    "/Ensembles/Data/pima-indians-diabetes.test",
    sep=",",
    header=None,
    names=columns)
print("Test data loaded!")

numOfEstimators = 50

# Create an ensemble of decision trees
dTree = tree.DecisionTreeClassifier(criterion='entropy')
bagging = BaggingClassifier(
    dTree,
    n_estimators=numOfEstimators,
    max_samples=len(trainD),
    bootstrap=True)

xDF = trainD.drop('Class', 1, inplace=False)
yDF = trainD['Class']

# Train
bagging.fit(xDF.values, yDF.values)

# Predict
tx = testD.drop('Class', 1, inplace=False)
preds = bagging.predict_proba(tx.values)

# Calculate bias-variance
ty = testD['Class']
bvd.biasVarZeroOne(ty.values, preds, numOfEstimators)


