import pandas
import collections
import numpy
import math
from scipy.stats import chisquare


def train(arffData):
    attributeDict = collections.OrderedDict(arffData['attributes'])
    attributeNames = list(attributeDict.keys())

    # Put the arffData into a pandas DataFrame
    df = pandas.DataFrame(
        numpy.array(arffData['data']),
        columns=attributeNames)

    # Remove the Class attribute before calling ID3
    attributeNames.remove('Class')

    # Train!
    return DecisionTree(_id3(df, attributeDict, attributeNames))


# ID3 implementation.  Returns a decision tree object.
def _id3(examples, attributeDict, attributeNames):
    # Create a root node for the tree
    root = Node()
    numRows = len(examples)

    # If all our training examples fall under one
    # category, then this is a leaf node and we're done.
    for category in attributeDict['Class']:
        # If the current example
        if len(examples[examples['Class'] == category]) == numRows:
            root.label = category
            return root

    print(len(attributeNames))
    # Select the best attribute to split on.
    attribute = _chooseBestAttribute(examples, attributeNames)

    # Get the most frequently occurring class in examples
    mostFrequentClass = examples['Class'].value_counts().idxmax()

    # Perform the chi square test
    isWorthBranching = _chiSquareHelper(examples, attribute, 0.95)

    # If it isn't worth branching.  Create a leaf.
    if isWorthBranching:
        root.attribute = attribute
    else:
        root.label = mostFrequentClass
        return root

    # Create a shallow copy of the attributeNames list, then
    # remove the attribute we're currently splitting on.
    reducedAttributeSet = list(attributeNames)
    reducedAttributeSet.remove(attribute)

    for value in examples[attribute].unique():
        exampleSubset = examples[examples[attribute] == value]
        # print(len(attributeNames), attribute, value, len(exampleSubset))

        if len(exampleSubset) == 0:
            root.branches[value] = Node(label=mostFrequentClass)
        else:
            root.branches[value] = _id3(
              exampleSubset,
              attributeDict,
              reducedAttributeSet)

    return root


# Given a set of observations and attributes, find
# the "best" attribute to split on.
def _chooseBestAttribute(examples, attributeNames):
    bestIG = 0
    splitCandidate = None

    for a in attributeNames:
        setEntropy = _calculateEntropy(examples)
        infoGain = _calculateInformationGain(examples, a, setEntropy)

        if infoGain > bestIG:
            bestIG = infoGain
            splitCandidate = a

    return splitCandidate


def _chiSquareHelper(examples, attribute, cutoff):
    # List of possible attribute values
    V = examples[attribute].unique()

    # Create two lists:
    return True


def _calculateInformationGain(examples, attribute, setEntropy):
    pV = examples[attribute].value_counts(normalize=True, dropna=False)

    weightedSubsetEntropy = 0
    for v in pV.keys():
        subset = examples[examples[attribute] == v]
        weightedSubsetEntropy += pV[v] * _calculateEntropy(subset)

    return setEntropy - weightedSubsetEntropy


# Calculates the entropy for a given set of observations.
def _calculateEntropy(examples):
    pV = examples.Class.value_counts(normalize=True, dropna=False)
    result = 0

    for v in pV.keys():
        pv = pV[v]
        result += -(pv * math.log(pv, 2))

    return result


class DecisionTree(object):

    def __init__(self, node):
        self.rootNode = node

    # Outputs results
    def predict(self, testData):
        attributeDict = collections.OrderedDict(testData['attributes'])
        attributeNames = list(attributeDict.keys())

        # Put the arffData into a pandas DataFrame
        testData = pandas.DataFrame(
            numpy.array(testData['data']),
            columns=attributeNames)

        # Add a new column to the testData.
        testData['PredictedClass'] = pandas.Series(None, index=testData.index)

        for i in range(0, len(testData)):
            example = testData.iloc[[i]]
            predictedClass = self._classify(example, self.rootNode, i)
            testData.set_value(i, 'PredictedClass', predictedClass)

        correctCount = len(
            testData[testData['Class'] == testData['PredictedClass']])
        totalRows = len(testData)

        print(testData[testData.Class == testData.PredictedClass])
        print(testData.Class.dtype, testData.PredictedClass.dtype)
        accuracy = float(correctCount) / float(totalRows)
        print("Accuracy: ", accuracy)

    def _classify(self, example, root, i):
        # Base case
        if root.attribute is None:
            return root.label
        else:
            val = example[root.attribute][i]
            return self._classify(example, root.branches[val], i)


class Node(object):

    def __init__(self, attribute=None, label=None):
        self.attribute = attribute
        self.label = label
        self.branches = dict()
