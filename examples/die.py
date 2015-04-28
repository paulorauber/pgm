import numpy as np
import contextlib

from model.factor import RandomVar
from model.factor import Factor
from model.factor import CPD
from model.gd import BayesianNetwork
from inference.approximate import ForwardSampler
from learning.parameter import ExpectationMaximization
from learning.structure import LikelihoodScore


def die():
    # Parameters
    # d1_ = [0.2, 0.0, 0.5, 0.1, 0.1, 0.1]
    # d2_ = [0.2, 0.3, 0.1, 0.05, 0.05, 0.3]
    d1_ = [0.1, 0.9]
    d2_ = [0.6, 0.4]
    n_samples = 1000

    n_iterations = 20
    n_restarts = 5
    verbose = 2

    # Model creation
    if len(d1_) != len(d2_):
        raise Exception('The die should have the same cardinality')

    h = RandomVar('h', 2)
    o1 = RandomVar('o1', len(d1_))
    o2 = RandomVar('o2', len(d2_))

    f_h = CPD([h], [0.5, 0.5])
    f_o1_h = Factor([o1, h])
    f_o2_h = Factor([o2, h])

    for i in range(len(f_o1_h.values)):
        o_, h_ = f_o1_h.itoa(i)
        f_o1_h.values[i] = d1_[o_] if h_ == 0 else d2_[o_]
        f_o2_h.values[i] = d2_[o_] if h_ == 0 else d1_[o_]

    f_o1_h = CPD(f_o1_h.scope, f_o1_h.values)
    f_o2_h = CPD(f_o2_h.scope, f_o2_h.values)

    bn = BayesianNetwork([f_h, f_o1_h, f_o2_h])

    # Sampling from true model
    fs = ForwardSampler(bn)
    fs.sample(n_samples)
    scope, X = fs.samples_to_matrix()

    em = ExpectationMaximization(scope, known_cpds=[f_h],
                                 n_iterations=n_iterations,
                                 n_restarts=n_restarts, alpha=10.0,
                                 verbose=verbose)

    print('True log-likelihood (no missing variables):')
    print(em.expected_log_likelihood(X, bn))

    print('Maximum log-likelihood (no missing variables):')
    ls = LikelihoodScore(scope)
    ls.fit(X, bn.graph())
    print(ls.score)

    # Hiding variable
    X[:, scope.index(h)] = -1

    print('True expected log-likelihood (missing variables):')
    print(em.expected_log_likelihood(X, bn))

    bn_pred = em.fit_predict(X, bn.graph())
    print('Best expected log-likelihood (missing variables)')
    print(em.expected_log_likelihood(X, bn_pred))

    # Estimation results
    print('Results:')
    f_o1_h = [f for f in bn_pred.factors if f.scope[0] == o1][0]
    f_o2_h = [f for f in bn_pred.factors if f.scope[0] == o2][0]

    d = np.zeros(o1.k)

    with printoptions(precision=3):
        print('d1: {0}'.format(d1_))

        for i in range(o1.k):
            d[i] = f_o1_h.values[f_o1_h.atoi([i, 0])]
        print('d1 according to o1: {0}'.format(d))

        for i in range(o2.k):
            d[i] = f_o2_h.values[f_o2_h.atoi([i, 1])]
        print('d1 according to o2: {0}'.format(d))

        print('d2: {0}'.format(d2_))
        for i in range(o1.k):
            d[i] = f_o1_h.values[f_o1_h.atoi([i, 1])]
        print('d2 according to o1: {0}'.format(d))

        for i in range(o2.k):
            d[i] = f_o2_h.values[f_o2_h.atoi([i, 0])]
        print('d2 according to o2: {0}'.format(d))


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

if __name__ == "__main__":
    die()
