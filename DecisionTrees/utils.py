import pandas
import numpy
import collections

# A deserialized arff object looks like this:
# xor_dataset = {
#     'description': 'XOR Dataset',
#     'relation': 'XOR',
#     'attributes': [
#         ('input1', 'REAL'),
#          ('input2', 'REAL'),
#          ('y', 'REAL'),
#      ],
#      'data': [
#          [0.0, 0.0, 0.0],
#          [0.0, 1.0, 1.0],
#          [1.0, 0.0, 1.0],
#          [1.0, 1.0, 0.0]
#      ]
#  }
def convertArffToDataFrame(arffObject):

	attributeDict = collections.OrderedDict(arffObject['attributes'])
	attributeNames = attributeDict.keys()

	df = pandas.DataFrame(
		numpy.array(arffObject['data']),
		columns = attributeNames)

	return df

# from scipy.io import arff
# arff_train = arff.loadarff('training_subsetD.arff')

# arff_train[0] # has data in Numpy NDarray
# arff_train[1] # has metadata