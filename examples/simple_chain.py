
from model.factor import RandomVar
from model.factor import CPD
from model.gd import BayesianNetwork
from inference.exact import VariableElimination
from inference.exact import JointMarginalization
from inference.approximate import ForwardSampler
from inference.approximate import GibbsSampler
import itertools


def main():
    x1 = RandomVar('X1', 2)
    x2 = RandomVar('X2', 2)
    x3 = RandomVar('X3', 2)

    fx1 = CPD([x1], [0.11, 0.89])
    fx2_x1 = CPD([x2, x1], [0.59, 0.22, 0.41, 0.78])
    fx3_x2 = CPD([x3, x2], [0.39, 0.06, 0.61, 0.94])

    bn = BayesianNetwork([fx1, fx2_x1, fx3_x2])
#    mn = MarkovNetwork([fx1, fx2_x1, fx3_x2])

    ve = VariableElimination(bn)
    jm = JointMarginalization(bn)

    print(ve.posterior([x1, x2], [(x3, 0)]))
    print(jm.posterior([x1, x2], [(x3, 0)]))

    print(ve.posterior([x1, x2, x3]))
    print(jm.posterior([x1, x2, x3]))

    print(ve.maximum_a_posteriori(evidence=[(x3, 0)]))
    print(jm.maximum_a_posteriori([x1, x2], [(x3, 0)]))

    fs = ForwardSampler(bn)
    fs.sample(10000)

    for c in itertools.product(range(2), repeat=3):
        print('{0}: {1}'.format(c, fs.posterior(zip([x1, x2, x3], c))))

    px3_0 = fs.posterior([(x3, 0)])
    for c in itertools.product(range(2), repeat=2):
        assg = list(zip([x1, x2], c)) + [(x3, 0)]

        print('{0}: {1}'.format(c, fs.posterior(assg) / px3_0))

    gs = GibbsSampler(bn)
    gs.sample(burn_in=1000, n=2000)

    for c in itertools.product(range(2), repeat=3):
        print('{0}: {1}'.format(c, gs.posterior(zip([x1, x2, x3], c))))

    gs.reset()
    gs.sample(burn_in=1000, n=1000, evidence=[(x3, 0)])

    for c in itertools.product(range(2), repeat=2):
        print('{0}: {1}'.format(c, gs.posterior(zip([x1, x2], c))))

if __name__ == "__main__":
    main()
