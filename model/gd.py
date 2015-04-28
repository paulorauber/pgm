import numpy as np
from collections import deque
from model.factor import Factor


class GibbsDistribution:

    def __init__(self, factors):
        self.factors = factors

    def reduce(self, evidence):
        factors = []
        for f in self.factors:
            ev = [(var, value) for (var, value) in evidence if var in f.scope]
            factors.append(f.reduce(ev))

        return GibbsDistribution(factors)

    def joint(self, normalize=True):
        f = Factor([], [1.0])
        for fi in self.factors:
            f = f * fi

        return f.normalize() if normalize else f

    def variables(self):
        return set([v for f in self.factors for v in f.scope])

    def __repr__(self):
        return ''.join([f.__repr__() for f in self.factors])


class MarkovNetwork(GibbsDistribution):

    def __init__(self, factors):
        super(MarkovNetwork, self).__init__(factors)


class BayesianNetwork(GibbsDistribution):

    def __init__(self, factors):
        super(BayesianNetwork, self).__init__(factors)

    def graph(self):
        g = {v: set() for f in self.factors for v in f.scope}
        for f in self.factors:
            if len(f.scope) > 0:
                u = f.scope[0]
                for w in f.scope[1:]:
                    g[w].add(u)
        return g

    def topological_sorting(self):
        return topological_sorting(self.graph())

    def joint_probability(self, full_assg_map):
        p = 1
        for f in self.factors:
            assg = [full_assg_map[v] for v in f.scope]
            p *= f.values[f.atoi(assg)]

        return p

    def dimension(self):
        d = 0

        for f in self.factors:
            if len(f.scope) > 0:
                d += (f.scope[0].k - 1) * np.prod([v.k for v in f.scope[1:]])

        return int(d)


def topological_sorting(graph):
    """Code published by Alexey Kachayev"""
    GRAY, BLACK = 0, 1

    order, enter, state = deque(), set(graph), {}

    def dfs(node):
        state[node] = GRAY
        for k in graph.get(node, ()):
            sk = state.get(k, None)
            if sk == GRAY:
                raise ValueError("cycle")
            if sk == BLACK:
                continue
            enter.discard(k)
            dfs(k)
        order.appendleft(node)
        state[node] = BLACK

    while enter:
        dfs(enter.pop())
    return list(order)
