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

            print("userId: {0:n}, movieId: {1:n}, predicted rating: {2:n}, actual rating: {3:n}".format(
                row['userId'][i],
                row['movieId'][i],
                predictedVote,
                row['rating'][i]))
            # Print some kind of progress indicator
            print("{0:10.6f}%".format(i/len(self.testData)))

        # Calculate Mean Absolute Error
        print(self._computeMAE())

        # Calculate Root Mean Square Error
        print(self._computeRSME())


    # Predicts the vote of user A on Item j.
    def _predictVote(self, user, item):
        # Retrieve the average vote for the active user.
        Vbar_a = self.avgVotes[user]

        # Get all the rows corresponding to votes for this item.
        itemVotes = self.trainData[
            (self.trainData['movieId'] == item) &
            (self.trainData['userId'] != user)]

        # Users who have voted on the current prediction item
        usersWhoVotedForItem = itemVotes['userId'].unique()

        nonZeroWeights = self._computePearsonCoefficients(
            user,
            usersWhoVotedForItem)

        # This is a hack.  It's possible to get no non-zero weights.
        # Should have a better way of handling this
        if len(nonZeroWeights) == 0:
            return Vbar_a

        k = self._computeNormalizer(nonZeroWeights)

        itemRatings = self.trainData[
            (self.trainData['userId'].isin(nonZeroWeights.index.values)) &
            (self.trainData['movieId'] == item)]

        weightedAdjustedRating = itemRatings.apply(
            lambda row: nonZeroWeights[numpy.int64(row['userId'])]*(row['rating'] - self.avgVotes[numpy.int64(row['userId'])]),
            axis=1)
        itemRatings = itemRatings.assign(weightedAdjustedRating=weightedAdjustedRating)

        summationTerm = itemRatings['weightedAdjustedRating'].sum()

        return Vbar_a + (k * summationTerm)


    # Returns a dataframe containing userIds and pearson correlation
    # coefficients.  Non-zero values only.
    def _computePearsonCoefficients(self, userA, usersWhoVotedForItem):
        # Inner function to compute each w(a, i)
        def computePearsonCoefficient(group):
            numerator = (group['normalizedRating_a']*group['normalizedRating_b']).sum()
            denominator = math.sqrt((group['normalizedRating_a']**2).sum())*math.sqrt((group['normalizedRating_b']**2).sum())

            # Case where there is only one vote
            if denominator == 0:
                result = 0
            else:
                result = numerator/denominator
            return result

        # Get user A's votes.
        subsetA = self.trainData[self.trainData['userId'] == userA]
        movies = subsetA['movieId'].unique()

        # Compute (V_aj - Vbar_a)
        Vbar_a = self.avgVotes[userA]
        normalizedRating = subsetA.apply(
            lambda row: row['rating'] - Vbar_a,
            axis=1,
            raw=True)

        subsetA = subsetA.assign(normalizedRating=normalizedRating)

        # Compute Pearson Coefficients only for users who satisfy
        # the following conditions:
        # 1 - The user has at least one movieId overlap with userA
        # 2 - The user has voted for the current prediction item
        # 3 - The user is not userA
        subsetB = self.trainData[
            (self.trainData['movieId'].isin(movies)) &
            (self.trainData['userId'].isin(usersWhoVotedForItem)) &
            (self.trainData['userId'] != userA)]

        #print(subsetB)

        # Compute (V_ij - Vbar_i)
        normalizedRating = subsetB.apply(
            lambda row: row['rating'] - self.avgVotes[numpy.int64(row['userId'])],
            axis=1,
            raw=True)

        #print(normalizedRating)

        subsetB = subsetB.assign(normalizedRating=normalizedRating)

        merged = pandas.merge(
            subsetA,
            subsetB,
            how='inner',
            on=['movieId'],
            suffixes=('_a', '_b'))

        # Group subsetB by userIds
        grouped = merged.groupby('userId_b')

        # Compute pearson coefficient for each group
        result = grouped.apply(computePearsonCoefficient)
        #print(result)
        return result[result != 0]


    def _computeMAE(self):
        return (self.testData['predictedRating'] - self.testData['rating']).abs().mean()


    def _computeRSME(self):
        return math.sqrt(((self.testData['predictedRating'] - self.testData['rating'])**2).mean())


    def _computeNormalizer(self, nonZeroWeights):
        val = nonZeroWeights.abs().sum()
        if val == 0:
            return 0
        else:
            return 1/nonZeroWeights.abs().sum()


    # Helper to get the vote for user i on item j.
    def _getVote(self, user_i, item_j):
        return self.trainData[
            (self.trainData['userId'] == user_i) &
            (self.trainData['movieId'] == item_j)]['rating']


    # Precomputes the average vote for every user in the training set.
    def _precomputeAvgVotes(self):
        self.avgVotes = self.trainData.groupby('userId')['rating'].mean()