import numpy as np

from model.gd import GibbsDistribution


class Sampler:

    def __init__(self):
        self.samples = []

    def posterior(self, var_assg_pairs):
        var_assg_pairs = list(var_assg_pairs)

        n = len(self.samples)
        if n == 0:
            raise Exception('No samples for estimate')

        m = 0

        for s in self.samples:
            valid = 1

            for var, assg in var_assg_pairs:
                if s[var] != assg:
                    valid = 0
                    break

            m += valid

        return float(m) / n

    def reset(self):
        self.samples = []

    def samples_to_matrix(self):
        if len(self.samples) == 0:
            return None, np.array([], dtype=np.float)

        scope = sorted(self.samples[0].keys())
        var_pos = {v: i for (i, v) in enumerate(scope)}

        X = np.zeros((len(self.samples), len(scope)), dtype=np.int)
        for i, sample in enumerate(self.samples):
            for var, assg in sample.items():
                X[i, var_pos[var]] = assg

        return scope, X


class ForwardSampler(Sampler):

    def __init__(self, bn):
        super(ForwardSampler, self).__init__()
        self.bn = bn

    def sample(self, n):
        variables = self.bn.topological_sorting()
        cpds = {f.scope[0]: f for f in self.bn.factors}
        for _ in range(n):
            s = dict()

            for v in variables:
                assg = [(var, s[var]) for var in cpds[v].scope[1:]]
                s[v] = cpds[v].sample(assg)

            self.samples.append(s)


class GibbsSampler(Sampler):

    def __init__(self, gd):
        super(GibbsSampler, self).__init__()
        self.gd = gd

    def sample(self, evidence=[], var_assg_pairs0=None, burn_in=1000,
               n=1000):
        gd = self.gd.reduce(evidence)
        variables = gd.variables()

        if var_assg_pairs0:
            s = {v: a for (v, a) in var_assg_pairs0}
        else:
            s = {v: np.random.choice(v.k) for v in variables}

        for t in range(burn_in + n):
            for v in s.keys():
                gd_v = GibbsDistribution(
                    [f for f in gd.factors if v in f.scope])

                del s[v]
                gd_v = gd_v.reduce(s.items())

                f = gd_v.joint()
                s[v] = f.sample()

            if t >= burn_in:
                self.samples.append(dict(s))
