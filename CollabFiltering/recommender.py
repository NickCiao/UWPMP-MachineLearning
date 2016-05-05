import pandas
import numpy
import math

class Recommender(object):

    def __init__(self, trainingData):
        self.trainData = trainingData
        self.testData = None
        self.avgVotes = None


    def execute(self, testData):
        self.testData = testData

        # Create a new column for testData to hold predicted values.
        self.testData['predictedRating'] = self.testData['rating'].map(
            lambda x: x)

        # Precompute everyone's average votes.
        print("Starting precomputeAvgVotes...")
        self._precomputeAvgVotes()
        print("Finished precomputeAvgVotes...")

        print("Prediction started...")
        # Iterate over each row in the test data,
        # predict the rating for each row.
        for i in range(0, len(self.testData)):
            row = self.testData.iloc[[i]]

            # Make the prediction
            predictedVote = self._predictVote(row['userId'][i], row['movieId'][i])

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
        Vbar_a = self.avgVotes[user]

        # Dataframe containing all votes for item
        itemVotes = self.trainData[
            (self.trainData['movieId'] == item) &
            (self.trainData['userId'] != user)]

        # Users who have voted on the current prediction item
        usersWhoVotedForItem = itemVotes['userId'].unique()

        nonZeroWeights = self._computePearsonCoefficients(
            user,
            usersWhoVotedForItem)

        k = self._computeNormalizer(nonZeroWeights)

        nonZeroWeights['rating_adjusted'] = nonZeroWeights.apply(
            lambda row: self._getVote(row['userId'], item) - self.avgVotes[row['userId']],
            axis=1)

        summationTerm = (nonZeroWeights['r'] * nonZeroWeights['rating_adjusted']).sum()

        return Vbar_a + (k * summationTerm)


    # Returns a dataframe containing userIds and pearson correlation
    # coefficients.  Non-zero values only.
    def _computePearsonCoefficients(self, userA, usersWhoVotedForItem):
        # Subset of training data consisting of user A's votes.
        subsetA = self.trainData[self.trainData['userId'] == userA]
        movies = subsetA['movieId'].unique()

        # All votes that overlap with userA's votes
        overlappingVotes = self.trainData[
            (self.trainData['movieId'].isin(movies)) &
            (self.trainData['userId'] != userA)]

        similarUsers = overlappingVotes['userId'].unique()

        # The intersection between similarUsers and usersWhoVotedForItem
        usersToCompare = numpy.intersect1d(usersWhoVotedForItem, similarUsers)

        # Get rows associated with userIds in usersToCompare
        workingSet = self.trainData[
            self.trainData['userId'].isin(usersToCompare)]


        return workingSet

        # Users who have voted on the current prediction item
        usersWhoVotedForItem = self.trainData[
            (self.trainData['movieId'] == item) &
            (self.trainData['userId'] != userA)]['userId'].unique()


    def _computeMAE(self):
        pass


    def _computeRSME(self):
        pass


    def _computeNormalizer(self, nonZeroWeights):
        return 1/nonZeroWeights['r'].abs().sum()


    # Helper to get the vote for user i on item j.
    def _getVote(self, user_i, item_j):
        return self.trainData[
            (self.trainData['userId'] == user_i) &
            (self.trainData['movieId'] == item_j)]['rating']


    # Precomputes the average vote for every user in the training set.
    def _precomputeAvgVotes(self):
        self.avgVotes = self.trainData.groupby('userId')['rating'].mean()