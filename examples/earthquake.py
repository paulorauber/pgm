from model.factor import RandomVar
from model.factor import CPD
from model.gd import BayesianNetwork
from inference.exact import VariableElimination
from inference.exact import JointMarginalization
from inference.approximate import ForwardSampler


def main():
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

    ve = VariableElimination(bn)
    jm = JointMarginalization(bn)

    print(ve.posterior([B, E, A, R]) == jm.posterior([B, E, A, R]))

    fs = ForwardSampler(bn)
    fs.sample(1000)

    # for c in itertools.product(range(2), repeat = 4):
    #     print('{0}: {1}'.format(c, fs.posterior(zip([A,R,B,E],c))))

if __name__ == "__main__":
    main()
