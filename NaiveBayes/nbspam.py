import pandas
import numpy
import math

class NaiveBayesSpamFilter(object):

    def __init__(self, trainingData):
        self.trainData = trainingData

    def execute(self, testData):
        self.testData = testData
