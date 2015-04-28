import numpy as np
from functools import reduce

from model.factor import Factor
from model.factor import CPD
from model.gd import GibbsDistribution


class JointMarginalization:

    def __init__(self, gd):
        self.gd = gd

    def posterior(self, hypothesis, evidence=[]):
        j = self.gd.reduce(evidence).joint()

        marg = [v for v in j.scope if v not in hypothesis]
        for m in marg:
            j = j.marginalize(m)

        return j

    def maximum_a_posteriori(self, hypothesis, evidence=[]):
        joint = self.posterior(hypothesis, evidence)
        return list(zip(joint.scope, joint.itoa(np.argmax(joint.values))))


class VariableElimination:

    def __init__(self, gd):
        self.gd = gd

    def posterior(self, hypothesis, evidence=[], normalize=True):
        gd = self.gd.reduce(evidence)
        elim_variables = set(
            [v for f in gd.factors for v in f.scope if v not in hypothesis])

        gm = MinFillElimination(gd.factors)
        elim_variables = gm.ordering(elim_variables)

        factors = gd.factors

        for v in elim_variables:
            factor = reduce(lambda x, y: x * y,
                            [f for f in factors if v in f.scope],
                            Factor([], np.array([1.0])))
            factor = factor.marginalize(v)
            factors = [f for f in factors if v not in f.scope] + [factor]

        return GibbsDistribution(factors).joint(normalize)

    def maximum_a_posteriori(self, evidence=[]):
        gd = self.gd.reduce(evidence)

        elim_variables = set([v for f in gd.factors for v in f.scope])

        gm = MinFillElimination(gd.factors)
        elim_variables = gm.ordering(elim_variables)

        factors = gd.factors

        phis = []
        for v in elim_variables:
            factor = reduce(lambda x, y: x * y,
                            [f for f in factors if v in f.scope],
                            Factor([], np.array([1.0])))

            phis.append(factor)

            factor = factor.maximize(v)
            factors = [f for f in factors if v not in f.scope] + [factor]

        assgmap = dict()
        for (i, f) in reversed(list(enumerate(phis))):
            assg = [0] * len(f.scope)

            for j, v in enumerate(f.scope):
                if v == elim_variables[i]:
                    assg[j] = -1
                else:
                    assg[j] = assgmap[v]

            assgmap[elim_variables[i]] = f.argmax(assg)

        return [(v, assgmap[v]) for v in elim_variables]


class MinFillElimination():

    def __init__(self, factors):
        self.m = {}
        for f in factors:
            for v in f.scope:
                if v in self.m:
                    self.m[v].update(f.scope)
                else:
                    self.m[v] = set(f.scope)

        for v in self.m.keys():
            self.m[v].remove(v)

    def ordering(self, elim_variables):
        ordering = []

        elim = list(elim_variables)
        m = dict(self.m)

        while elim:
            bestv, bestunc = elim[0], self.unconnected(m, m[elim[0]])

            for v in elim[1:]:
                unconnected = self.unconnected(m, m[v])

                if len(unconnected) < len(bestunc):
                    bestv = v
                    bestunc = unconnected

            ordering.append(bestv)
            elim.remove(bestv)

            del m[bestv]
            for v, n in m.items():
                if bestv in n:
                    n.remove(bestv)

            for u, v in bestunc:
                m[u].add(v)
                m[v].add(u)

        return ordering

    def unconnected(self, m, variables):
        unc = set()
        for u in variables:
            for v in variables:
                if u != v and v not in m[u]:
                    unc.add((min(u, v), max(u, v)))

        return unc

    def __repr__(self):
        return self.m.__repr__()


class ExpectedUtility:

    def __init__(self, influence_diagram):
        self.id = influence_diagram

    def expected_utility(self, decision_factors):
        eu = 0.0

        cf = self.id.chance_factors
        df = decision_factors

        for uf in self.id.utility_factors:
            gd = GibbsDistribution(cf + df + [uf])
            ve = VariableElimination(gd)

            eu += ve.posterior([], normalize=False).values[0]

        return eu

    def optimal_decision_rule(self, scope):
        cf = self.id.chance_factors
        uf = Factor([], [0.0])
        for f in self.id.utility_factors:
            uf += f

        gd = GibbsDistribution(cf + [uf])
        ve = VariableElimination(gd)

        mu = ve.posterior(scope, normalize=False)
        assg_map = [scope.index(v) for v in mu.scope]
        ind = mu.scope.index(scope[0])

        rule = Factor(scope, np.zeros(np.prod([v.k for v in scope])))
        n = int(np.prod([v.k for v in scope[1:]]))

        for i in range(n):
            assg = rule.itoa(i)

            assg_mu = np.array(assg)[assg_map]
            assg_mu[ind] = -1

            assg_max = [mu.argmax(assg_mu)] + list(assg[1:])

            rule.values[rule.atoi(assg_max)] = 1.0

        return CPD(rule.scope, rule.values)
