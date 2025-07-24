import math
import numpy as np
import random
import symreg as sr

def test_pysr():
    X = 2 * np.random.randn(5, 100)
    y = 2.5382 * np.cos(X[3, :]) + X[0, :] ** 2 - 0.5

    variables = []
    
    for i in range(0, 5):
        variables.append(sr.Variable("x" + str(i), X[i, :]))

    paramsValue = [] #[0, 1, -0.5, 2.5382]
    operatorDepth = {"cos": 1, "*": 1, "+": 1}
    extraExpressions = []
    verbose = False
    callback = lambda x: None
    discreteParams = True
    timeout = 30 * 60
    keepBestLimit = 10000

    model = sr.SymbolicRegressor(variables,
                                 [sr.UnaryOperator.cos()],
                                 [sr.BinaryOperator.times(), sr.BinaryOperator.plus()],
                                 2,
                                 paramsValue,
                                 operatorDepth,
                                 extraExpressions,
                                 verbose,
                                 callback,
                                 discreteParams,
                                 timeout,
                                 keepBestLimit)

    r = model.fit(y)

    print(r.expression.optStr())
    
    assert(r.loss < model.epsLoss)
