import numpy as np
from itertools import product

from model.factor import Factor
from model.factor import CPD
from model.gd import BayesianNetwork
from inference.exact import VariableElimination

from collections import OrderedDict


class OccurrenceCounter:

    """Accumulates sufficient statistics from a data set

    This class does not implement ideal caching. For instance, the counts could
    be independent of the order of the scopes, which would also require changes
    to BayesianNetworkEstimator subclasses.
    """

    def __init__(self, scope, maxlen=10000):
        self.scope = scope
        self.maxlen = maxlen

        self.X = None
        self.stats = OrderedDict()
        self.last_scopes = None

    def count_occurences(self, scopes):
        var_index = {v: i for (i, v) in enumerate(self.scope)}

        for scope in scopes:
            for x in self.X:
                M = self.stats[scope]
                M.values[M.atoi(x[[var_index[v] for v in M.scope]])] += 1

    def fit(self, X, graph):
        self.X = np.array(X)
        self.stats = OrderedDict()

        new_scopes = set()
        parents = find_parents(self.scope, graph)
        for v in self.scope:
            scope = tuple([v] + sorted(parents[v]))
            self.stats[scope] = Factor(scope)
            new_scopes.add(scope)

        self.last_scopes = new_scopes
        self.count_occurences(new_scopes)

        return self

    def refit(self, graph):
        parents = find_parents(self.scope, graph)

        new_scopes = set()
        for v in self.scope:
            scope = tuple([v] + sorted(parents[v]))
            new_scopes.add(scope)

        old_scopes = set(self.stats.keys())

        remove_c = len(old_scopes | new_scopes) - self.maxlen
        if remove_c > 0:
            for key in self.stats.keys():
                if key not in new_scopes:
                    del self.stats[key]
                    remove_c -= 1

                if remove_c <= 0:
                    break

        self.last_scopes = new_scopes

        new_scopes = new_scopes - old_scopes
        for scope in new_scopes:
            self.stats[scope] = Factor(scope)

        self.count_occurences(new_scopes)

        return self


def find_parents(scope, graph):
    parents = {v: set() for v in scope}
    for p, children in graph.items():
        for child in children:
            parents[child].add(p)

    return parents


class BayesianNetworkEstimator(object):

    def __init__(self, scope, maxlen):
        self.scope = scope
        self.maxlen = maxlen

        self.bn = None
        self.oc = None

    def fit(self, X, graph):
        self.oc = OccurrenceCounter(self.scope, self.maxlen)
        self.oc.fit(X, graph)
        self._fit_bn()
        return self

    def refit(self, graph):
        self.oc.refit(graph)
        self._fit_bn()
        return self

    def fit_predict(self, X, graph):
        return self.fit(X, graph).bn


class MaximumLikelihood(BayesianNetworkEstimator):

    def __init__(self, scope, maxlen=10000):
        super().__init__(scope, maxlen)

    def _fit_bn(self):
        Ms = [self.oc.stats[scope] for scope in self.oc.last_scopes]
        self.bn = BayesianNetwork([M.to_cpd() for M in Ms])


class UniformDirichlet(BayesianNetworkEstimator):

    def __init__(self, scope, alpha=1.0, maxlen=10000):
        super().__init__(scope, maxlen)
        self.alpha = alpha

    def _fit_bn(self):
        Ms = [self.oc.stats[scope] for scope in self.oc.last_scopes]

        cpds = []
        for M in Ms:
            cpds.append(M.add_scalar(self.alpha / len(M.values)).to_cpd())

        self.bn = BayesianNetwork(cpds)


class ExpectationMaximization:

    """Performs parameter estimation for data sets with missing data using a
    expectation maximization algorithm.

    This implementation is extremely inefficient. At least, the inference step
    should be performed using belief propagation.

    It is also possible to optimize for deterministically hidden variables.
    This implementation deals with missing data in the most general case.

    Parameters
    ----------
    scope: list of RandomVariable
        Random variables in corresponding order to the columns in the data
        matrix to be fitted.
    known_cpds: list of CPD
        Conditional probability distributions that are known a priori. These
        CPDs are not changed during the optimization. Must match with the graph
        given to `self.fit`.
    n_iterations: int
        Number of iterations of expectation maximization.
    n_restarts: int
        The number of times the optimization is repeated from the beginning,
        (very) likely using different initial parameters. In the end, only the
        maximum log-likelihood model is kept.
    alpha: float
        Equivalent sample size for initializing the parameters for each
        (unknown) CPD according to a Dirichlet distribution.
    verbose: int
        Verbosity level between 0 and 2 (inclusive).

    Attributes
    ----------
    bn: BayesianNetwork
        Bayesian network representing the parameters fitted from the data.
        Only available after calling `self.fit`.

    Examples
    --------
    See examples/die.py

    """

    def __init__(self, scope, known_cpds=None, n_iterations=10, n_restarts=2,
                 alpha=10.0, verbose=1):
        self.scope = scope

        self.known_cpds = known_cpds

        self.n_iterations = n_iterations
        self.n_restarts = n_restarts
        self.alpha = alpha

        self.verbose = verbose

        self.bn = None

    def init(self, graph):
        if self.known_cpds is None:
            self.known_cpds = []
        self.alpha = float(self.alpha)

        known = {cpd.scope[0] for cpd in self.known_cpds}
        self.unknown = set(self.scope) - known

        known_cpds = [CPD(cpd.scope, cpd.values) for cpd in self.known_cpds]
        unknown_cpds = []

        self.parents = find_parents(self.scope, graph)
        for v in self.unknown:
            pa_v = sorted(self.parents[v])
            f = Factor([v] + pa_v)

            val_pa_v = product(*(range(pa.k) for pa in pa_v))
            for assg in val_pa_v:
                dist = np.random.dirichlet([self.alpha / len(f.values)] * v.k)

                assg = list(assg)
                for i in range(v.k):
                    f.values[f.atoi([i] + assg)] = dist[i]

            unknown_cpds.append(CPD(f.scope, f.values))

        self.bn = BayesianNetwork(known_cpds + unknown_cpds)

    def fit(self, X, graph):
        """Find the parameters for a probabilistic graphical model, given a
        graph and a data set that possibly contains missing data.

        After fitting, the model is available as a BayesianNetwork `self.bn`.

        Parameters
        ----------
        X : two-dimensional np.array or python matrix of integers
            Matrix representing the observations. The value `X[i, j]` should
            correspond to the discrete random variable `self.scope[j]` in
            sample element `i`. The number -1 represents a missing value.
        graph: dict from RandomVariables to sets of RandomVariables
            the graph for the probabilistic graphical model
        """
        var_index = {v: i for (i, v) in enumerate(self.scope)}

        best_ll = float('-inf')
        best_bn = None
        for irestart in range(self.n_restarts):
            if self.verbose > 0:
                print('Restart {0}.'.format(irestart + 1))

            self.init(graph)

            known_cpds = [CPD(cpd.scope, cpd.values)
                          for cpd in self.known_cpds]

            M_scopes = []
            for v in self.unknown:
                M_scopes.append([v] + sorted(self.parents[v]))

            for iiteration in range(self.n_iterations):
                ess = [Factor(M_scope) for M_scope in M_scopes]

                for x in X:
                    evidence = []
                    hidden = []
                    for (i, xi) in enumerate(x):
                        if xi == -1:
                            hidden.append(self.scope[i])
                        else:
                            evidence.append((self.scope[i], xi))

                    for M in ess:
                        M_assg = x[[var_index[v] for v in M.scope]]

                        M_h = []
                        for (i, v) in enumerate(M.scope):
                            if M_assg[i] == -1:
                                M_h.append(v)

                        if M_h:
                            ve = VariableElimination(self.bn)
                            f = ve.posterior(M_h, evidence=evidence)

                            Mh_index = [M.scope.index(v) for v in f.scope]

                            for i in range(len(f.values)):
                                f_assg = f.itoa(i)
                                M_assg[Mh_index] = f_assg
                                M.values[M.atoi(M_assg)] += f.values[i]
                        else:
                            M.values[M.atoi(M_assg)] += 1

                self.bn = BayesianNetwork(
                    [M.to_cpd() for M in ess] + known_cpds)

                ll = self.log_likelihood(X, self.bn)
                if self.verbose > 1:
                    print('Iteration {0}. '
                          'Current log-likelihood {1}.'.format(iiteration + 1,
                                                               ll))

                if ll > best_ll:
                    best_ll = ll
                    best_bn = self.bn

        self.bn = best_bn

        return self

    def fit_predict(self, X, graph):
        return self.fit(X, graph).bn

    def log_likelihood(self, X, bn):
        var_index = {v: i for (i, v) in enumerate(self.scope)}

        ll = 0
        for x in X:
            hidden = []
            for (i, xi) in enumerate(x):
                if xi == -1:
                    hidden.append(self.scope[i])

            reduced_scope = [v for v in self.scope if v not in hidden]

            ve = VariableElimination(bn)
            f = ve.posterior(reduced_scope)

            assg = x[[var_index[v] for v in f.scope]]

            ll += np.log(f.values[f.atoi(assg)])

        return ll
