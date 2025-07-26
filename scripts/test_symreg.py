import math
import numpy as np
import random
import sympy
from sympy import Float, Rational
import symreg as sr

def symStr(e, eps = 1e-4):
    return e.replace(lambda x: isinstance(x, (Float, Rational)), lambda x: Float(round(float(x) / eps) * eps))

def test_pysr():
    np.random.seed(0)

    X = 2 * np.random.randn(5, 100)
    y = 2.5382 * np.cos(X[3, :]) + X[0, :] ** 2 - 0.5

    variables = []
    
    for i in range(0, 5):
        variables.append(sr.Variable("x" + str(i), X[i, :]))

    paramsValue = [] #[0, 1, -0.5, 2.5382]
    operatorDepth = {"cos": 1, "*": 1, "+": 1}
    extraExpressions = [sr.Expression.bin(sr.BinaryOperator.plus(),
                                          sr.Expression.un(sr.UnaryOperator.cos(), sr.Expression.var(variables[3])),
                                          sr.Expression.bin(sr.BinaryOperator.times(),
                                                            sr.Expression.var(variables[0]),
                                                            sr.Expression.var(variables[0])))]
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
    print(r.expression.symStr())
    print(symStr(sympy.expand_trig(sympy.sympify(r.expression.symStr()))))
    #print(r.expression.dot())

    assert(r.loss < model.epsLoss)
