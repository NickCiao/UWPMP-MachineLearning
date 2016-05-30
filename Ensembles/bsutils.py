import pandas
import collections
import numpy
import math
import sys
from scipy import stats
from sklearn import tree

'''
    Bootstrap Utilities
'''
def createNBootstrapSets(n, df):
    trainingSets = []
    for i to range(0, n-1):
        trainingSets[i] = df.sample(n=df.size, replace=True)

    return trainingSets

'''
    X is the training set. [<example1>,<example2>,...]
    Y is known output.
'''
def trainNTrees(n, X, Y, maxTreeDepth=None):
    trainingSets = createNBootstrapSets(n, X)
    trees = []

    for i in range(0, n-1):
        trainD = trainingSets[i]
        dTree = tree.DecisionTreeClassifier(
            criterion='entropy',
            max_depth=maxTreeDepth)

        trees[i] = decisionTree.fit(X.values, Y.values)

    return trees
