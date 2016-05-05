# Returns a dataframe containing userIds and pearson correlation
    # coefficients.  Non-zero values only.
    def _computePearsonCoefficients(self, userA, item):
        # Grab every userId except for userA
        allIds = self.trainData['userId'].unique()
        otherUserIds = numpy.delete(allIds, numpy.where(allIds == userA))

        # Dataframe slice for userA
        userAData = self.trainData[self.trainData['userId'] == userA]

        # Get active user's average vote
        Vbar_a = self.avgVotes[userA]

        # temp arrays
        chosenUsers = []
        chosenR = []

        # Compute Pearson Coefficients
        for user in otherUserIds:
            # Dataframe slice for userB
            userBData = self.trainData[self.trainData['userId'] == user]
            intersection = pandas.merge(
                userAData,
                userBData,
                how='inner',
                on=['movieId'],
                suffixes=('_a', '_b'))

            if len(intersection) == 0:
                continue

            # add column that represents (V_aj - Vbar_a)
            intersection['rating_adjusted_a'] = intersection['rating_a'].subtract(Vbar_a)

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

            # Only care about non-zero weights
            if r != 0:
                chosenUsers.append(user)
                chosenR.append(r)

        return pandas.DataFrame({'userId': chosenUsers, 'r': chosenR})