from model.factor import CPD
from model.factor import RandomVar
from model.gd import BayesianNetwork

from inference.approximate import ForwardSampler

from learning.classification import NaiveBayes


def one_predictive_var():
    Y = RandomVar('Y', 2)

    X1 = RandomVar('X1', 2)
    X2 = RandomVar('X2', 2)
    X3 = RandomVar('X3', 2)

    f_X1_Y = CPD([X1, Y], [1.0, 0.0, 0.0, 1.0])
    f_X2_Y = CPD([X2, Y], [0.5, 0.5, 0.5, 0.5])
    f_X3_Y = CPD([X3, Y], [0.5, 0.5, 0.5, 0.5])

    f_Y = CPD([Y], [0.5, 0.5])

    bn = BayesianNetwork([f_Y, f_X1_Y, f_X2_Y, f_X3_Y])

    # Training the model
    fs = ForwardSampler(bn)
    fs.sample(1000)
    scope, X = fs.samples_to_matrix()

    y = X[:, -1]
    X = X[:, 0:-1]

    nb = NaiveBayes()
    nb.fit(X, y)

    # Evaluating the model

    fs = ForwardSampler(bn)
    fs.sample(10)
    _, X = fs.samples_to_matrix()

    print(nb.score(X[:, 0:-1], X[:, -1]))
    print(nb.predict_proba(X[:, 0:-1]))


def no_predictive_var():
    Y = RandomVar('Y', 2)

    X1 = RandomVar('X1', 2)
    X2 = RandomVar('X2', 2)
    X3 = RandomVar('X3', 2)

    f_X1_Y = CPD([X1, Y], [0.5, 0.5, 0.5, 0.5])
    f_X2_Y = CPD([X2, Y], [0.5, 0.5, 0.5, 0.5])
    f_X3_Y = CPD([X3, Y], [0.5, 0.5, 0.5, 0.5])

    f_Y = CPD([Y], [0.5, 0.5])

    bn = BayesianNetwork([f_Y, f_X1_Y, f_X2_Y, f_X3_Y])

    # Training the model
    fs = ForwardSampler(bn)
    fs.sample(100)
    scope, X = fs.samples_to_matrix()

    y = X[:, -1]
    X = X[:, 0:-1]

    nb = NaiveBayes()
    nb.fit(X, y)

    fs = ForwardSampler(bn)
    fs.sample(1000)
    _, X = fs.samples_to_matrix()

    # Evaluating the model
    fs = ForwardSampler(bn)
    fs.sample(10)
    _, X = fs.samples_to_matrix()

    print(nb.score(X[:, 0:-1], X[:, -1]))
    print(nb.predict_proba(X[:, 0:-1]))


def almost_predictive_vars():
    Y = RandomVar('Y', 2)

    X1 = RandomVar('X1', 2)
    X2 = RandomVar('X2', 2)
    X3 = RandomVar('X3', 2)

    f_X1_Y = CPD([X1, Y], [0.8, 0.7, 0.2, 0.3])
    f_X2_Y = CPD([X2, Y], [0.6, 0.3, 0.4, 0.7])
    f_X3_Y = CPD([X3, Y], [0.3, 0.7, 0.7, 0.3])

    f_Y = CPD([Y], [0.5, 0.5])

    bn = BayesianNetwork([f_Y, f_X1_Y, f_X2_Y, f_X3_Y])

    # Training the model
    fs = ForwardSampler(bn)
    fs.sample(1000)
    scope, X = fs.samples_to_matrix()

    y = X[:, -1]
    X = X[:, 0:-1]

    nb = NaiveBayes()
    nb.fit(X, y)

    print(nb.bn_)

    # Evaluating the model

    fs = ForwardSampler(bn)
    fs.sample(10000)
    _, X = fs.samples_to_matrix()

    print(nb.score(X[:, 0:-1], X[:, -1]))
    print(nb.predict_proba(X[:, 0:-1]))


def main():
    one_predictive_var()
    no_predictive_var()
    almost_predictive_vars()

if __name__ == "__main__":
    main()
