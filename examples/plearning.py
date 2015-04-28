import numpy as np

from model.factor import RandomVar
from model.factor import CPD
from model.gd import BayesianNetwork
from inference.approximate import ForwardSampler
from learning.parameter import MaximumLikelihood
from learning.parameter import UniformDirichlet


def simple_chain():
    x1 = RandomVar('X1', 2)
    x2 = RandomVar('X2', 2)
    x3 = RandomVar('X3', 2)

    fx1 = CPD([x1], [0.11, 0.89])
    fx2_x1 = CPD([x2, x1], [0.59, 0.22, 0.41, 0.78])
    fx3_x2 = CPD([x3, x2], [0.39, 0.06, 0.61, 0.94])

    bn = BayesianNetwork([fx1, fx2_x1, fx3_x2])
    graph = bn.graph()
    print(bn)

    fs = ForwardSampler(bn)
    fs.sample(1000)
    scope, X = fs.samples_to_matrix()

    mle = MaximumLikelihood(scope)
    print(mle.fit_predict(X, graph))

    ud = UniformDirichlet(scope, alpha=1.0)
    print(ud.fit_predict(X, graph))


def earthquake():
    B = RandomVar('B', 2)
    E = RandomVar('E', 2)
    A = RandomVar('A', 2)
    R = RandomVar('R', 2)

    a_be = CPD(
        [A, B, E], [0.999, 0.01, 0.01, 0.0001, 0.001, 0.99, 0.99, 0.9999])
    r_e = CPD([R, E], [1.0, 0.0, 0.0, 1.0])
    b = CPD([B], [0.99, 0.01])
    e = CPD([E], [0.999, 0.001])

    bn = BayesianNetwork([a_be, r_e, b, e])
    print(bn)

    fs = ForwardSampler(bn)
    fs.sample(1000)
    scope, X = fs.samples_to_matrix()

    mle = MaximumLikelihood(scope)
    print(mle.fit_predict(X, bn.graph()))

    ud = UniformDirichlet(scope, alpha=1.0)
    print(ud.fit_predict(X, bn.graph()))


def traffic():
    A = RandomVar('A', 2)
    T = RandomVar('T', 2)
    P = RandomVar('P', 2)

    fP = CPD([P], [0.99, 0.01])
    fA = CPD([A], [0.9, 0.1])

    fT_AP = CPD([T, P, A], [0.9, 0.5, 0.4, 0.1, 0.1, 0.5, 0.6, 0.9])

    bn = BayesianNetwork([fP, fA, fT_AP])
    print(bn)

    fs = ForwardSampler(bn)
    fs.sample(1000)
    scope, X = fs.samples_to_matrix()

    mle = MaximumLikelihood(scope)
    print(mle.fit_predict(X, bn.graph()))

    ud = UniformDirichlet(scope, alpha=1.0)
    print(ud.fit_predict(X, bn.graph()))


def simple_sampling():
    V1 = RandomVar('V1', 3)
    V2 = RandomVar('V2', 3)
    V3 = RandomVar('V3', 5)

    scope = [V1, V2, V3]
    graph = {V1: {V3}, V2: {V3}, V3: set()}

    X = np.zeros((1000, 3), dtype=np.int)
    X[:, 0:2] = np.random.choice(
        range(3), size=(X.shape[0], 2), p=[0.2, 0.5, 0.3])
    X[:, 2] = X[:, 0] + X[:, 1]

    mle = MaximumLikelihood(scope)
    print(mle.fit_predict(X, graph))

    ud = UniformDirichlet(scope, alpha=1.0)
    print(ud.fit_predict(X, graph))


def main():
    simple_chain()
    earthquake()
    traffic()
    simple_sampling()

if __name__ == "__main__":
    main()
