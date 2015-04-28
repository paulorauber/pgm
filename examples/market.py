from model.factor import RandomVar
from model.factor import CPD
from model.factor import Factor
from model.influence import InfluenceDiagram
from inference.exact import ExpectedUtility


def three_variables():
    M = RandomVar('Market', 3)
    F = RandomVar('Found', 2)

    uMF = Factor([M, F], [0, -7, 0, 5, 0, 20])

    cM = CPD([M], [0.5, 0.3, 0.2])

    # Alternative decision rules for F
    dF_1 = CPD([F], [1.0, 0])
    dF_2 = CPD([F], [0, 1.0])  # Optimal

    id = InfluenceDiagram([cM], [uMF])
    eu = ExpectedUtility(id)

    print(eu.expected_utility([dF_1]))
    print(eu.expected_utility([dF_2]))

    print(eu.optimal_decision_rule([F]))


def six_variables():
    M = RandomVar('Market', 3)
    S = RandomVar('Survey', 4)  # S = 3 means no survey

    T = RandomVar('Test', 2)
    F = RandomVar('Found', 2)

    uMF = Factor([M, F], [0, -7, 0, 5, 0, 20])
    uT = Factor([T], [0, -1])

    cM = CPD([M], [0.5, 0.3, 0.2])

    cST = CPD([S, M, T], [0.0, 0.6, 0.0, 0.3, 0.0, 0.1,
                          0.0, 0.3, 0.0, 0.4, 0.0, 0.4,
                          0.0, 0.1, 0.0, 0.3, 0.0, 0.5,
                          1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    # Alternative decision rules for F given S
    dFS_1 = CPD([F, S], [0, 0, 0, 1, 1, 1, 1, 0])
    dFS_2 = CPD([F, S], [1, 0, 0, 0, 0, 1, 1, 1])  # Optimal

    # Alternative decision rules for T
    dT_1 = CPD([T], [1.0, 0.0])
    dT_2 = CPD([T], [0.0, 1.0])  # Optimal

    id = InfluenceDiagram([cM, cST], [uMF, uT])
    eu = ExpectedUtility(id)

    print(eu.expected_utility([dFS_1, dT_1]))
    print(eu.expected_utility([dFS_1, dT_2]))
    print(eu.expected_utility([dFS_2, dT_1]))
    print(eu.expected_utility([dFS_2, dT_2]))

    # New influence diagram with a single decision rule
    dT = dT_2

    id2 = InfluenceDiagram([cM, cST, dT], [uMF, uT])
    eu2 = ExpectedUtility(id2)

    dFS_optimal = eu2.optimal_decision_rule([F, S])
    print(eu.expected_utility([dFS_optimal, dT]))


def main():
    three_variables()
    six_variables()

if __name__ == '__main__':
    main()
