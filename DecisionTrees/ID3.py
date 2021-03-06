import pandas
import numpy
import math
import sys
from scipy import stats

class DecisionTree(object):

    def __init__(self, trainData, yColName):
        self.trainData = trainData
        self.yColName = yColName
        attributeNames = list(trainData.columns.values).remove(yColName)
        self.rootNode = self._id3(trainData, attributeNames)


    # ID3 implementation.  Returns a decision tree object.
    def _id3(self, trainData, attributeNames):
        # Create a root node for the tree
        root = Node()
        numRows = len(trainData)

        # If all our training examples fall under one
        # category, then this is a leaf node and we're done.
        for category in trainData[self.yColName].unique():
            # If the current example
            if len(trainData[trainData[self.yColName] == category]) == numRows:
                root.label = category
                return root

        # Select the best attribute to split on.
        attribute = _chooseBestAttribute(trainData, attributeNames)

        # Get the most frequently occurring class in trainData
        mostFrequentClass = trainData[self.yColName].value_counts().idxmax()

        # Perform the chi square test
        isWorthBranching = _chiSquareHelper(trainData, attribute, 0.1)

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

        for value in trainData[attribute].unique():
            exampleSubset = trainData[trainData[attribute] == value]

            if len(exampleSubset) == 0:
                root.branches[value] = Node(label=mostFrequentClass)
            else:
                root.branches[value] = _id3(
                  exampleSubset,
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
        V = examples[attribute].unique()

        testStatistic = 0
        p = len(examples[examples[self.yColName] == "True"])
        n = len(examples[examples[self.yColName] == "False"])
        dof = 0
        for v in V:
            if v is not numpy.NaN and v is not None:
                dof += 1
                subset = examples[examples[attribute] == v]
                p_i = len(subset[subset[self.yColName] == "True"])
                n_i = len(subset[subset[self.yColName] == "False"])
                pprime = p*((p_i+n_i)/(p+n))
                nprime = n*((p_i+n_i)/(p+n))
                testStatistic += (math.pow(p_i - pprime, 2)/pprime) + (math.pow(n_i - nprime, 2)/nprime)

        pValue = 1 - stats.chi2.cdf(testStatistic, dof)
        return pValue <= cutoff


    def _calculateInformationGain(examples, attribute, setEntropy):
        pV = examples[attribute].value_counts(normalize=True, dropna=False)

        weightedSubsetEntropy = 0
        for v in pV.keys():
            if v is not numpy.NaN:
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


    # Outputs results
    def predict(self, testData):
        attributeDict = collections.OrderedDict(testData['attributes'])
        attributeNames = list(attributeDict.keys())

        # Put the arffData into a pandas DataFrame
        testData = pandas.DataFrame(
            numpy.array(testData['data']),
            columns=attributeNames)

        # Add a new column to the testData.
        testData['PredictedClass'] = pandas.Series("", index=testData.index)

        for i in range(0, len(testData)):
            example = testData.iloc[[i]]
            predictedClass = self._classify(example, self.rootNode, i)

            testData.set_value(i, 'PredictedClass', predictedClass)

        correctCount = len(
            testData[testData[self.yColName] == testData['PredictedClass']])
        totalRows = len(testData)

        print(testData[testData.Class == testData.PredictedClass])
        accuracy = float(correctCount) / float(totalRows)
        #print(testData)
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
