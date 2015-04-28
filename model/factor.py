import numpy as np
import operator


class RandomVar:

    def __init__(self, name, k=2):
        self.name = name
        self.k = k

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return self.name.__hash__()

    def __lt__(self, other):
        return self.name < other.name


class Factor:

    """Representation of a factor over discrete random variables"""

    def __init__(self, scope, values=None):
        self.scope = scope

        if values is None:
            self.values = np.zeros(np.prod(self.scope_dimensions()))
        else:
            self.values = np.array(values)
            if self.values.size != np.prod(self.scope_dimensions()):
                raise Exception('Incorrect table given scope')

    def scope_dimensions(self):
        return [v.k for v in self.scope]

    def itoa(self, i):
        return np.unravel_index(i, self.scope_dimensions())

    def atoi(self, assg):
        return np.ravel_multi_index(assg, self.scope_dimensions())

    def operation(self, oper, factor):
        """Perform a binary operation `oper` between this factor and `factor`.

        This method generalizes factor products to other binary operations.

        Parameters
        ----------
        oper : callable
            Binary operator on real numbers.
        factor : Factor
            Factor whose operation we want to compute with `self`.

        Returns
        -------
        Factor
            Factor corresponding to the binary operation between `self` and
            `factor`.

        """
        if len(factor.scope) == 0:
            return Factor(list(self.scope),
                          oper(self.values, factor.values[0]))
        if len(self.scope) == 0:
            return Factor(list(factor.scope),
                          oper(factor.values, self.values[0]))

        scope_both = [v for v in self.scope if v in factor.scope]
        scope_self = [v for v in self.scope if v not in scope_both]
        scope_factor = [v for v in factor.scope if v not in scope_both]

        scope = scope_self + scope_both + scope_factor
        output = Factor(scope, np.zeros(np.prod([v.k for v in scope])))

        assg_self = [output.scope.index(v) for v in self.scope]
        assg_f = [output.scope.index(v) for v in factor.scope]

        for i in range(output.values.size):
            assg = np.array(output.itoa(i))

            i_self = self.atoi(assg[assg_self])
            i_f = factor.atoi(assg[assg_f])

            output.values[i] = oper(self.values[i_self], factor.values[i_f])

        return output

    def __mul__(self, factor):
        return self.operation(operator.mul, factor)

    def __add__(self, factor):
        return self.operation(operator.add, factor)

    def marginalize(self, var):
        scope = [v for v in self.scope if v != var]
        if len(scope) == 0:
            return Factor([], np.array([sum(self.values)]))

        output = Factor(scope, np.zeros(np.prod([v.k for v in scope])))

        assg_marg = [self.scope.index(v) for v in scope]
        for i in range(self.values.size):
            j = output.atoi(np.array(self.itoa(i))[assg_marg])

            output.values[j] += self.values[i]

        return output

    def maximize(self, var):
        scope = [v for v in self.scope if v != var]
        if len(scope) == 0:
            return Factor([], np.array([max(self.values)]))

        output = Factor(scope, np.zeros(np.prod([v.k for v in scope])))
        output.values.fill(-np.infty)

        assg_marg = [self.scope.index(v) for v in scope]
        for i in range(self.values.size):
            j = output.atoi(np.array(self.itoa(i))[assg_marg])

            if self.values[i] > output.values[j]:
                output.values[j] = self.values[i]

        return output

    def argmax(self, partial_assg):
        assg = np.array(partial_assg, dtype=int)
        ind = np.where(assg == -1)[0][0]

        assg[ind] = 0

        maximum = float('-inf')
        imaximum = 0
        for i in range(self.scope[ind].k):
            assg[ind] = i
            if self.values[self.atoi(assg)] > maximum:
                imaximum = i
                maximum = self.values[self.atoi(assg)]

        return imaximum

    def observe(self, evidence):
        j = Factor(list(self.scope), np.array(self.values))
        for var, value in evidence:
            var_index = self.scope.index(var)

            for i in range(self.values.size):
                assg = self.itoa(i)[var_index]
                if assg != value:
                    j.values[i] = 0

        return j

    def reduce(self, evidence):
        j = self.observe(evidence)

        for var, _ in evidence:
            j = j.marginalize(var)

        return j

    def sample(self, evidence=[]):
        j = self.reduce(evidence)

        if len(j.scope) != 1:
            raise Exception('Invalid scope for sampling')

        return np.random.choice(range(j.scope[0].k), p=j.values)

    def normalize(self):
        return Factor(list(self.scope), self.values / self.values.sum())

    def to_cpd(self):
        if len(self.scope) == 0:
            return CPD([], np.array([1.0]))
        elif len(self.scope) == 1:
            value = self.values.sum()
            if np.allclose(value, 0):
                k = self.scope[0].k
                return CPD(list(self.scope), np.ones(k, dtype=np.float) / k)
            else:
                return CPD(list(self.scope), self.values / value)

        values = np.array(self.values)

        marginal = self.marginalize(self.scope[0])
        for i in range(len(values)):
            value = marginal.values[marginal.atoi(self.itoa(i)[1:])]

            if np.allclose(value, 0):
                values[i] = 1. / self.scope[0].k
            else:
                values[i] = values[i] / value

        return CPD(list(self.scope), values)

    def add_scalar(self, scalar):
        return Factor(list(self.scope), self.values + scalar)

    def __repr__(self):
        if len(self.scope) == 0:
            return '{0}\n'.format(self.values[0])

        string = ''.join([v.name + ' ' for v in self.scope]) + '\n'
        for i in range(self.values.size):
            string += '{0} -> {1}\n'.format(self.itoa(i), self.values[i])
        return string

    def __eq__(self, other):
        return self.scope == other.scope and \
            np.allclose(self.values, other.values)


class CPD(Factor):

    def __init__(self, scope, values):
        Factor.__init__(self, scope, values)

        if not self.valid():
            raise Exception('Invalid CPD')

    def valid(self):
        if not np.allclose((self.values >= 0), 1):
            return False

        if len(self.scope) == 1:
            return np.allclose(self.values.sum(), 1.0)

        rscope = [v.k for v in self.scope[1:]]
        acc = np.zeros(np.prod(rscope))

        for i in range(self.values.size):
            j = np.ravel_multi_index(self.itoa(i)[1:], rscope)
            acc[j] += self.values[i]

        return np.allclose(acc, 1.0)
