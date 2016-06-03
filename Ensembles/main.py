import numpy
import pandas
import sys
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
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

numOfEstimators = 30

# QUESTION 1
# Create an ensemble of decision trees
dTree = tree.DecisionTreeClassifier(criterion='entropy')
bagging = BaggingClassifier(
    dTree,
    n_estimators=numOfEstimators,
    max_samples=len(trainD),
    bootstrap=True)

xDF = trainD.drop('Class', 1, inplace=False)
yDF = trainD['Class']

# # Train
bagging.fit(xDF.values, yDF.values)

# # Predict
tx = testD.drop('Class', 1, inplace=False)
preds = bagging.predict_proba(tx.values)

# # Calculate bias-variance
ty = testD['Class']
bvd.biasVarZeroOne(ty.values, preds, numOfEstimators)

# QUESTION 4 PART 1
# Create a few SVM classifiers
rbfsvm = svm.SVC(kernel='rbf', probability=True)
rbfsvm = rbfsvm.fit(xDF.values, yDF.values)
result = rbfsvm.score(tx.values, ty.values)
print("rbf kernel score:{0}".format(result))

linsvm = svm.SVC(kernel='linear', probability=True)
linsvm = linsvm.fit(xDF.values, yDF.values)
result = linsvm.score(tx.values, ty.values)
print("linear kernel score:{0}".format(result))

sigmoidsvm = svm.SVC(kernel='sigmoid', probability=True)
sigmoidsvm = sigmoidsvm.fit(xDF.values, yDF.values)
result = sigmoidsvm.score(tx.values, ty.values)
print("sigmoid kernel score:{0}".format(result))

# QUESTION 4 PART 2
# Create a naive bayes classifier
mnb = MultinomialNB()
y_pred = mnb.fit(xDF.values, yDF.values).score(tx.values, ty.values)
print("Naive Bayes score:{0}".format(y_pred))

# QUESTION 4 PART 3
preds = rbfsvm.predict_proba(tx.values)
bvd.biasVarZeroOne(ty.values, preds, 1)

preds = linsvm.predict_proba(tx.values)
bvd.biasVarZeroOne(ty.values, preds, 1)

preds = sigmoidsvm.predict_proba(tx.values)
bvd.biasVarZeroOne(ty.values, preds, 1)

