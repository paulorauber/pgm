"""Classification based on simple probabilistic graphical models"""

import numpy as np

from model.factor import RandomVar

from learning.parameter import UniformDirichlet
from inference.exact import VariableElimination


class NaiveBayes:
    """Multinomial Naive Bayes implementation.

    This implementation is inefficient. It serves mainly as a
    proof of concept that Naive Bayes can be seen in the framework of proba-
    bilistic graphical models.
    """
    def __init__(self):
        self.classes_ = None
        self.scope_ = None

        self.bn_ = None

    def fit(self, X, y):
        """Fit a Multinomial Naive Bayes model to the data.

        Parameters:
        -----------
        X : two-dimensional np.array or python matrix of integers
            Matrix representing the observations. It is assumed that
            `X[:, i]` is a sample from a discrete random variable $X_i$ that
            takes values between `0` and `X[:, i].max()`
        y : one-dimensional np.array or python list of integers
            Array representing the classes assigned to each observation
        """
        X = np.asarray(X, dtype=np.int)
        if X.min() < 0:
            raise Exception('Invalid samples')

        self.classes_, y = np.unique(y, return_inverse=True)

        C = RandomVar('Y', len(self.classes_))

        scope = []
        for i in range(X.shape[1]):
            scope.append(RandomVar('X{0}'.format(i), X[:, i].max() + 1))

        graph = {v: set() for v in scope}
        graph[C] = set(scope)

        scope.append(C)

        Xy = np.concatenate([X, y.reshape(-1, 1)], axis=1)

        self.bn_ = UniformDirichlet(scope).fit_predict(Xy, graph)
        self.scope_ = scope

        return self

    def predict(self, X):
        """Predict classes for observations in `X` given the learned model"""
        if X.shape[1] != len(self.scope_) - 1:
            raise Exception('Invalid observations')

        ypred = np.zeros(X.shape[0], np.int)

        varelim = VariableElimination(self.bn_)

        for i, x in enumerate(X):
            evidence = []
            for j in range(len(x)):
                evidence.append((self.scope_[j], x[j]))

            c = varelim.maximum_a_posteriori(evidence)[0][1]

            ypred[i] = self.classes_[c]

        return ypred

    def predict_proba(self, X):
        """Predict probability of an observation belonging to each class.
        Return a matrix whose lines correspond to the observations and whose
        columns represent the probability of the observation belonging to each
        class, in the order in which they appear in `self.classes_`"""
        if X.shape[1] != len(self.scope_) - 1:
            raise Exception('Invalid observations')

        proba = np.zeros((X.shape[0], len(self.classes_)), np.float)

        varelim = VariableElimination(self.bn_)

        for i, x in enumerate(X):
            evidence = []
            for j in range(len(x)):
                evidence.append((self.scope_[j], x[j]))

            proba[i] = varelim.posterior([self.scope_[-1]], evidence).values

        return proba

    def score(self, X, y):
        """Score prediction given `X` wrt correct prediction `y`"""
        return ((self.predict(X) == y).sum())/float(len(y))
