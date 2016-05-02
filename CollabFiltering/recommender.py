import pandas
import numpy

class Recommender(object):

    def __init__(self, trainingData):
        self.trainingData = trainingData
        self.testData = None
        self.avgVotes = None


    def execute(self, testData):
        self.testData = testData

        # Create a new column for testData to hold predicted values.
        self.testData['predictedRating'] = self.testData['rating'].map(lambda x: x)

        # Precompute everyone's average votes.
        self._precomputeAvgVotes()

        # Go down each row in the test data, predict the rating for each row.
        for i in range(0, len(testD)):
            row = testD.iloc[[i]]

            # Make the prediction
            predictedVote = self._predictVote(row['userId'], row['movieId'])

            # Set the value
            self.testData.set_value(i, 'predictedRating', predictedVote)

            # Print some kind of progress indicator
            print("{0:8.3f}%".format(i/len(testD)))

        # Calculate Mean Absolute Error
        print(self._computeMAE())

        # Calculate Root Mean Square Error
        print(self._computeRSME())


    # Predicts the vote of user A on Item j.
    def _predictVote(self, user, item):
        # Compute the average vote for the active user.
        V_a = self.avgVotes[user]

        nonZeroWeights = _computePearsonCoefficients(user)
        k = _computeNormalizer(nonZeroWeights)

        weightedNN = 0
        for i in nonZeroWeights.keys():
            V_ij = self._getVote(i, item)
            weightedNN += nonZeroWeights[i] * (V_ij - self.avgVotes[i])

        return V_a + k*weightedNN


    # Returns a dictionary whose key consists of users against which
    # userA has a nonzero pearson correlation coefficient.
    # The value is the pearson correlation coefficient.
    def _computePearsonCoefficients(self, userA):
        pass


    def _computeMAE(self):
        pass


    def _computeRSME(self):
        pass


    def _computeNormalizer(self, nonZeroWeights):
        pass


    # Given a user, computes their average vote.
    def _computeAvgVote(self, user):

        pass


    # Retrieves the set of items on which user has voted.
    def _getVotedSet(self, user):
        pass


    # Helper to get the vote for user i on item j.
    def _getVote(self, i, j):
        return self.trainData[
            (self.trainData['userId'] == i) &
            (self.trainData['movieId'] == item)]['rating']


    # Precomputes the average vote for every user in the
    # training set and stores it as an instance variable.
    def _precomputeAvgVotes(self):
        # self.avgVotes =
        pass