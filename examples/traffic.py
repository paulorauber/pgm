from model.factor import RandomVar
from model.factor import CPD
from model.gd import BayesianNetwork

from inference.exact import VariableElimination
from inference.exact import JointMarginalization


def main():
    A = RandomVar('A', 2)
    T = RandomVar('T', 2)
    P = RandomVar('P', 2)

    fP = CPD([P], [0.99, 0.01])
    fA = CPD([A], [0.9, 0.1])

    fT_AP = CPD([T, P, A], [0.9, 0.5, 0.4, 0.1, 0.1, 0.5, 0.6, 0.9])

    bn = BayesianNetwork([fP, fA, fT_AP])

    ve = VariableElimination(bn)
    jm = JointMarginalization(bn)

    print(jm.maximum_a_posteriori([A], [(T, 1)]))

    print(ve.posterior([A], [(T, 1)]))
    print(jm.posterior([A], [(T, 1)]))

    print(ve.posterior([A, T, P]))
    print(jm.posterior([A, T, P]))

    # for c in itertools.product(range(2), repeat = 3):
    #     print('{0}: {1}'.format(c, fs.posterior(zip([A,T,P],c))))

if __name__ == "__main__":
    main()
