import pandas
import collections
import numpy
import math
import sys

'''
classes:  a vector containing the actual classes of the test examples,
          represented as integers; classes[i] is the class of test example i

preds:    an array where preds[i][j] is the class predicted for example i
          by the classifier learned on training set j; classes are represented
          as integers, consistent with those used in the classes vector

ntrsets:  the number of training sets used

loss:     pointer to a float where the average loss is returned

bias:     pointer to a float where the average bias is returned

var:      pointer to a float where the net variance is returned

varp:     pointer to a float where the average contribution to variance
          from unbiased examples is returned

varn:     pointer to a float where the average contribution to variance
          from biased examples is returned

varc:     pointer to a float where the average contribution to variance from
          biased examples is returned, with the variance from each example
          weighted by the probability that the class predicted for the example
          is the optimal prediction, given that it is not the main
          prediction. (In multiclass domains, the net variance is equal to
          varp - varc, not varp - varn.)
'''
def biasVarZeroOne(classes, preds, ntrsets):
    ntestexs = len(preds)

    loss = 0.0
    bias = 0.0
    varp = 0.0
    varn = 0.0
    varc = 0.0

    for e in range(0, ntestexs-1):
        lossx, biasx, varx = biasVarZeroOneX(classes[e], preds[e], ntrsets)
        loss += lossx
        bias += biasx
        if biasx != 0.0:
            varn += varx
            varc += 1.0
            varc -= lossx
        else:
            varp += varx

    loss = loss / ntestexs
    bias = bias/ ntestexs
    var = loss - bias
    # Unused
    varp = varp / ntestexs
    varn = varn / ntestexs
    varc = varc / ntestexs

    formatStr = "n_estimators:{0}, loss:{1}, variance:{2}, bias:{3}"
    print(formatStr.format(ntrsets, loss, var, bias))


'''
classx:   the actual class of the example, represented as an integer

predsx:   a vector where predsx[j] is the class predicted for the
          example by the classifier learned on training set j; classes are
          represented as integers, consistent with classx

ntrsets:  the number of training sets used
'''
def biasVarZeroOneX(classx, predsx, ntrsets):
    nmax = 0.0
    majclass = -1

    # Figure out which class had the most votes
    for c in range(0, len(predsx)-1):
        if predsx[c] > nmax:
            majclass = c
            nmax = predsx[c]

    # lossx: The probability of the ensemble incorrectly predicting this example.
    lossx = 1.0 - predsx[classx]
    # biasx: The 0/1 bias for this example.
    biasx = 1.0 if majclass != classx else 0.0
    # varx: 1.0 - the probability of the ensemble's prediction for this example.
    varx = 1.0 - predsx[majclass]

    return (lossx, biasx, varx)
