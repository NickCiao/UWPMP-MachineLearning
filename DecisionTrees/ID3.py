import pandas
import collections
import numpy
import math
from scipy.stats import chisquare

#
def train(examples):
	attributeDict = collections.OrderedDict(examples['attributes'])
	attributeNames = list(attributeDict.keys())

	# Put the examples into a pandas DataFrame
	df = pandas.DataFrame(
		numpy.array(examples['data']),
		columns = attributeNames)

	# Remove the Class attribute before calling ID3
	attributeNames.remove('Class')

	# Train!
	return DecisionTree(_id3(df, attributeDict, attributeNames))


# ID3 implementation.  Returns a decision tree object.
def _id3(examples, attributeDict, attributeNames):

	print(len(attributeNames))

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

	# Select the best attribute to split on.
	attribute = _chooseBestAttribute(examples, attributeNames)

	# Get the most frequently occurring class in examples
	mostFrequentClass = examples['Class'].value_counts().idxmax()

	# Perform the chi square test
	isWorthBranching = _chiSquareHelper(examples, attribute, cutOff)

	# If it isn't worth branching.  Create a leaf.
	if isWorthBranching:
		root.label = mostFrequentClass
		return root
	else:
		root.attribute = attribute

	# Create a shallow copy of the attributeNames list, then
	# remove the attribute we're currently splitting on.
	reducedAttributeSet = list(attributeNames)
	reducedAttributeSet.remove(attribute)

	for value in attributeDict[attribute]:
		exampleSubset = examples[examples[attribute] == value]
		if len(exampleSubset) == 0:
			root.branches[value] = Node(label = mostFrequentClass)
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
		infoGain = _calculateInformationGain(examples, a)

		if  infoGain > bestIG:
			bestIG = infoGain
			splitCandidate = a

	return a


def _chiSquareHelper(examples, attribute, cutoff):
	# List of possible attribute values
	V = examples[attribute].unique()

	# Create two lists:
	for v in V:
		p_prime =


def _calculateInformationGain(examples, attribute):
	pV = examples[attribute].value_counts(normalize=True)
	setEntropy = _calculateEntropy(examples)
	weightedSubsetEntropy = 0

	for v in pV.keys():
		subset = examples[examples[attribute] == v]
		weightedSubsetEntropy += pV[v] * _calculateEntropy(subset)

	return setEntropy - weightedSubsetEntropy


# Calculates the entropy for a given set of observations.
def _calculateEntropy(examples):
	pV = examples.Class.value_counts(normalize=True)
	result = 0

	for v in pV.keys():
		pv = pV[v]
		result += pv * math.log(pv, 2)

	return -result


class DecisionTree(object):

	def __init__(self, node):
		self.rootNode = node


	# Outputs results
	def predict(self, testData):
		# Add a new column to the testData.
		testData['PredictedClass'] = pandas.Series(None, index=testData.index)

		for i in len(testData) - 1:
			example = testData.iloc[[i]]
			predictedClass = _classify(example, self.rootNode)
			testData.set_value(i, 'PredictedClass', predictedClass)

		correctCount = \
			len(testData[testData['Class'] == testData['PredictedClass']])
		totalRows = len(testData)

		accuracy = correctCount / totalRows
		print("Accuracy: " + str(accuracy))


	def _classify(example, root):
		# Base case
		if root.attribute == None:
			return rootNode.label
		else:
			splitAttribute = root.attribute
			return _blargh(example, root.branches[splitAttribute])

class Node(object):

	def __init__(self, attribute = None, label = None):
		self.attribute = attribute
		self.label = label
		self.branches = dict()

