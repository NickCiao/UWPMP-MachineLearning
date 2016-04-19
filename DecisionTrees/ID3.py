import pandas
import collections
import numpy
import math

#
def train(examples):
	attributeDict = collections.OrderedDict(examples['attributes'])
	attributeNames = attributeDict.keys()

	# Put the examples into a pandas DataFrame
	df = pandas.DataFrame(
		numpy.array(arffObject['data']),
		columns = attributeNames)

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
		# If the current exa
		if len(examples[examples['Class'] == category]) == numRows
			root.label = category
			return root

	# Select the best attribute to split on.
	attribute = _chooseBestAttribute(examples, attributes)
	root.attribute = attribute

	# Create a shallow copy of the attributeNames list, then
	# remove the attribute we're currently splitting on.
	reducedAttributeSet = list(attributeNames)
	reducedAttributeSet.remove(attribute)

	for value in attributeDict[attribute]:
		exampleSubset = examples[examples[attribute] == value]
		if len(exampleSubset) == 0:
			mostFrequentClass = examples['Class'].value_counts().idxmax()
			root.branches[value] = Node(label = mostFrequentClass)
		else:
			root.branches[value] = _id3(
				exampleSubset,
				attributeDict,
				reducedAttributeSet)
	return root


# Given a set of observations and attributes, find
# the "best" attribute to split on.
def _chooseBestAttribute(examples, attributes):
	bestIG = 0
	splitCandidate = None

	for a in attributes
		infoGain = _calculateInformationGain(examples, a)

		if  infoGain > bestIG:
			bestIG = infoGain
			splitCandidate = a

	return a


def _calculateInformationGain(examples, attribute):
	pV = examples[attribute].value_counts(normalize=True)
	setEntropy = _calculateEntropy(examples)
	weightedSubsetEntropy = 0

	for v in pV:
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
		pass


class Node(object):

	def __init__(self, attribute = None, label = None):
		self.attribute = attribute
		self.label = label
		self.branches = dict()

