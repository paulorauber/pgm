import numpy as np
from learning.parameter import OccurrenceCounter

from model.gd import topological_sorting
from model.gd import BayesianNetwork

from scipy.special import gammaln


class StructureScore:

    def __init__(self, scope):
        self.scope = scope

        self.X = None
        self.oc = None
        self.score = None

    def fit(self, X, graph):
        self.X = np.array(X)

        self.oc = OccurrenceCounter(self.scope)
        self.oc.fit(X, graph)

        self._fit_score(graph)

        return self

    def refit(self, graph):
        self.oc.refit(graph)
        self._fit_score(graph)
        return self


class LikelihoodScore(StructureScore):

    def __init__(self, scope):
        super().__init__(scope)

    def _fit_score(self, graph):
        Ms = [self.oc.stats[scope] for scope in self.oc.last_scopes]

        self.score = 0
        for M in Ms:
            cpd = M.to_cpd()
            with np.errstate(divide='ignore'):
                log_values = np.log(cpd.values)
            log_values[log_values == -np.inf] = 0
            self.score += np.sum(M.values * log_values)


class BICScore(LikelihoodScore):

    def __init__(self, scope):
        super().__init__(scope)

    def _fit_score(self, graph):
        Ms = [self.oc.stats[scope] for scope in self.oc.last_scopes]

        cpds = []

        self.score = 0
        for M in Ms:
            cpd = M.to_cpd()
            with np.errstate(divide='ignore'):
                log_values = np.log(cpd.values)
            log_values[log_values == -np.inf] = 0

            self.score += np.sum(M.values * log_values)

            cpds.append(cpd)

        bn = BayesianNetwork(cpds)

        self.score -= (np.log(self.X.shape[0]) / 2.) * bn.dimension()


class BayesianScore(StructureScore):

    def __init__(self, scope, alpha=1.0, c=1.0):
        super().__init__(scope)

        self.alpha = alpha
        self.c = 1.0

    def _fit_score(self, graph):
        Ms = [self.oc.stats[scope] for scope in self.oc.last_scopes]

        self.score = 0
        for M in Ms:
            alpha_Xi_x_u = self.alpha / len(M.values)
            self.score += gammaln(M.values + alpha_Xi_x_u).sum()
            self.score -= len(M.values) * gammaln(alpha_Xi_x_u)

            alpha_Xi_u = M.scope[0].k * alpha_Xi_x_u
            Mmarg = M.marginalize(M.scope[0])
            self.score -= gammaln(alpha_Xi_u + Mmarg.values).sum()

            n_parents = np.prod([v.k for v in M.scope[1:]])
            self.score += n_parents * gammaln(alpha_Xi_u)

        n_edges = sum([len(v) for (_, v) in graph.items()])
        self.score += np.log(self.c ** n_edges)


def add_edge(graph):
    variables = set(graph.keys())

    edges = []
    for u in variables:
        vs = variables - graph[u]
        vs.remove(u)

        edges += [(u, v) for v in vs]

    candidates = []
    for (u, v) in edges:
        g = {key: set(value) for (key, value) in graph.items()}
        g[u].add(v)

        try:
            topological_sorting(g)
            if g not in candidates:
                candidates.append(g)
        except:
            continue

    return candidates


def remove_edge(graph):
    variables = set(graph.keys())

    edges = []
    for u in variables:
        edges += [(u, v) for v in graph[u]]

    candidates = []
    for (u, v) in edges:
        g = {key: set(value) for (key, value) in graph.items()}
        g[u].remove(v)

        if g not in candidates:
            candidates.append(g)

    return candidates


def revert_edge(graph):
    variables = set(graph.keys())

    edges = []
    for u in variables:
        edges += [(u, v) for v in graph[u]]

    candidates = []
    for (u, v) in edges:
        g = {key: set(value) for (key, value) in graph.items()}
        g[u].remove(v)
        g[v].add(u)

        try:
            topological_sorting(g)
            if g not in candidates:
                candidates.append(g)
        except:
            continue

    return candidates


def local_search(X, scope, scorer, iterations=100, epsilon=0.1, verbose=1):
    """This search procedure is significantly inefficient."""
    graph = {v: set() for v in scope}

    best_graph, best_score = graph, scorer.fit(X, graph).score

    for i in range(iterations):
        if verbose:
            print('Iteration {0}. Best: {1:0.3f}.'.format(i + 1,
                                                          best_score))

        candidates = add_edge(graph)
        candidates += remove_edge(graph)
        candidates += revert_edge(graph)

        best_c, best_c_score = None, float('-inf')
        for c in candidates:
            score = scorer.refit(c).score

            if score > best_c_score:
                best_c_score = score
                best_c = c

        if best_c_score > best_score:
            best_score = best_c_score
            best_graph = best_c

        if np.random.choice([False, True], p=[epsilon, 1 - epsilon]):
            graph = best_c
        else:
            graph = np.random.choice(candidates)

    return best_graph, best_score


def restarting_local_search(X, scope, scorer, restarts=10, iterations=100,
                            epsilon=0.1, verbose=2):
    best_graph, best_score = None, float('-inf')
    for i in range(restarts):
        if verbose:
            print('Restart: {0}'.format(i + 1))

        b, bs = local_search(X, scope, scorer, iterations, epsilon,
                             verbose=verbose-1)

        if bs > best_score:
            best_graph = b
            best_score = bs

    return best_graph, best_score
