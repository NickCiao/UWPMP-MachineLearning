import pandas
import numpy
import math

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

        # Iterate over each row in the test data, predict the rating for each row.
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
        Vbar_a = self.avgVotes[user]

        nonZeroWeights = _computePearsonCoefficients(user)
        k = _computeNormalizer(nonZeroWeights)

        weightedNN = 0
        for i in nonZeroWeights.keys():
            V_ij = self._getVote(i, item)
            weightedNN += nonZeroWeights[i] * (V_ij - self.avgVotes[i])

        return Vbar_a + k*weightedNN


    # Returns a dictionary whose key consists of users against which
    # userA has a nonzero pearson correlation coefficient.
    # The value is the pearson correlation coefficient.
    def _computePearsonCoefficients(self, userA):
        # Grab every userId except for userA
        allIds = self.trainData['userId'].unique()
        otherUserIds = numpy.delete(allIds, numpy.where(allIds == userA))

        # Dataframe slice for userA
        userAData = self.trainData[self.trainData['userId'] == userA]

        # Get active user's average vote
        Vbar_a = self.avgVotes[userA]

        # Compute Pearson Coefficients
        for user in otherUserIds:
            #Dataframe slice for userB
            userBData = self.trainData[self.trainData['userId' == user]]
            intersection = pandas.merge(
                userAData, userBData, how='inner', on=['movieId'], suffixes=('_a', '_b'))

            # add column that represents (V_aj - Vbar_a)
            intersection['rating_adjusted_a']  = intersection['rating_a'].subtract(Vbar_a)

            # add column that represents (V_ij - Vbar_i)
            Vbar_i = self.avgVotes[user]
            intersection['rating_adjusted_b'] = intersection['rating_b'].subtract(Vbar_i)

            intersection['numeratorProduct'] = intersection.apply(
                lambda row: row['rating_adjusted_a'] * row['rating_adjusted_b'],
                axis=1,
                raw=True)

            numerator = intersection['numeratorProduct'].sum()
            denominator = math.sqrt(
                (intersection['rating_adjusted_a']**2).sum() *
                (intersection['rating_adjusted_b']**2).sum())

            r =  numerator/denominator


    def _computeMAE(self):
        pass


    def _computeRSME(self):
        pass


    def _computeNormalizer(self, nonZeroWeights):
        pass


    # Helper to get the vote for user i on item j.
    def _getVote(self, user_i, item_j):
        return self.trainData[
            (self.trainData['userId'] == user_i) &
            (self.trainData['movieId'] == item_j)]['rating']


    # Precomputes the average vote for every user in the
    # training set and stores it as an instance variable.
    def _precomputeAvgVotes(self):
        distinctUserIds = self.trainData['userId'].unique()

        for userId in distinctUserIds:
            ratings = self.trainData[trainData['userId'] == userId]['rating']
            self.avgVotes[userId] = ratings.sum()/len(ratings)