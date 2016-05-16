import pandas
import numpy
import math
import util

def createDF(fileName):
    with open(fileName) as f:
        content = f.readlines()

    d = []
    for line in content:
        Id = ""
        Class = ""
        Word = ""
        for i, val in enumerate(line.split()):
            if i == 0:
                Id = val
            elif i == 1:
                Class = val
            elif i % 2 == 0:
                Word = val
            else:
                d.append({
                    'Id':Id,
                    'Class':Class,
                    'Word':Word,
                    'Count':int(val)})

    return pandas.DataFrame(d)

class NaiveBayesSpamFilter(object):

    def __init__(self, trainingData):
        self.trainData = trainingData
        self.pV = trainingData['Class'].value_counts(normalize=True)
        self.sizeOfVocab = len(trainingData['Word'].unique())
        self.probabilities = self._precomputeProbabilities()

    def execute(self, testData):
        self.testData = testData

        result = []
        Ids = testData['Id'].unique()
        progressCounter = 0

        # Make a prediction for each emailId in the test set
        for emailId in Ids:
            actualClass = testData['Class'].values[0]

            example = testData[testData['Id'] == emailId]
            predictedClass = self.predict(example)
            result.append({
                'ActualClass':actualClass,
                'PredictedClass':predictedClass
            })

            # Print the progress
            util.printProgress(progressCounter, len(Ids))

        resultDF = pandas.DataFrame(result)

        # Calculate accuracy
        countCorrect = len(resultDF[
            resultDF['ActualClass'] == resultDF['PredictedClass']])
        totalRows = len(resultDF)
        print("Accuracy: {0}".format(countCorrect/totalRows))

    def predict(self, example):
        pSpam = self._computeP("spam", example)
        pHam = self._computeP("ham", example)

        if pHam > pSpam:
            return "ham"
        else:
            return "spam"


    def _computeP(self, v, example):
        view = self.probabilities[self.probabilities['Class'] == v]
        valid = example['Word'].isin(view['Word'])
        p = 1

        for index, row in example[valid].iterrows():
            # Fetch the value from our pre-computed probabilities
            value = self.probabilities[
                (self.probabilities['Word'] == row['Word']) &
                (self.probabilities['Class'] == v)]

            # Warning!
            if len(value) > 1:
                print(">1 instance for p({0}|{1})".format(row['Word'], v))

            logP = math.log(float(value['P']))
            p = p * logP

        return p * self.pV[v]


    def _precomputeProbabilities(self):
        result = []
        progressCounter = 0

        # Unique pairs of Word/Class
        uniquePairs = self.trainData.loc[:, ['Class','Word']].drop_duplicates()

        for index, row in uniquePairs.iterrows():
            Class = row['Class']
            Word = row['Word']

            df = self.trainData[
                (self.trainData['Class'] == Class) &
                (self.trainData['Word'] == Word)]
            count = df['Count'].sum()

            df = self.trainData[self.trainData['Class'] == Class]
            total = df['Count'].sum()

            result.append({
                'Class':Class,
                'Word':Word,
                'P': (count + 1)/(total + self.sizeOfVocab)
            })

            # Print the progress
            progressCounter += 1
            util.printProgress(progressCounter, len(uniquePairs))

        return pandas.DataFrame(result)




