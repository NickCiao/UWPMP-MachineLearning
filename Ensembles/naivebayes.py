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

class NaiveBayes(object):

    def __init__(self, trainingData):
        self.trainData = trainingData
        self.pV = trainingData['Class'].value_counts(normalize=True)
        self.sizeOfVocab = len(trainingData['Word'].unique())
        self.wordCountsByClass = trainingData.groupby(['Class']).sum()['Count']
        self.probabilities = self._precomputeProbabilities()

    def execute(self, testData):
        self.testData = testData
        result = []
        Ids = testData['Id'].unique()
        progressCounter = 0

        print("Predicting...")
        # Make a prediction for each emailId in the test set
        for emailId in Ids:
            example = testData[testData['Id'] == emailId]
            actualClass = example['Class'].values[0]
            predicted = self.predict(example)

            if predicted['pHam'] > predicted['pSpam']:
                predictedClass = "ham"
            else:
                predictedClass = "spam"

            result.append({
                'ActualClass':actualClass,
                'PredictedClass':predictedClass,
                'pHam':predicted['pHam'],
                'pSpam':predicted['pSpam']
            })

            # Print the progress
            progressCounter += 1
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

        return {"pHam":pHam, "pSpam":pSpam}


    def _computeP(self, v, example):
        view = self.probabilities[self.probabilities['Class'] == v]
        # valid = example[example['Word'].isin(view['Word'])]
        valid = example
        p = 0

        for index, row in valid.iterrows():
            word = row['Word']
            # Fetch the value from our pre-computed probabilities
            value = self.probabilities[
                (self.probabilities['Word'] == word) &
                (self.probabilities['Class'] == v)]['P']

            if len(value) == 0:
                val = math.log(
                    1/(self.wordCountsByClass[v] + self.sizeOfVocab))
            else:
                val = float(value.values[0])

            p += (val*row['Count'])

        return p + math.log(self.pV[v])


    def _precomputeProbabilities(self):
        print("Pre-Computing likelihoods...")
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
            likelihood = (float(count) + 1)/(float(total) + self.sizeOfVocab)

            result.append({
                'Class':Class,
                'Word':Word,
                'P': math.log(likelihood)
            })

            # Print the progress
            progressCounter += 1
            util.printProgress(progressCounter, len(uniquePairs))

        return pandas.DataFrame(result)




