import numpy as np

from model.factor import RandomVar
from model.factor import CPD
from model.gd import BayesianNetwork

from inference.exact import JointMarginalization

from inference.approximate import ForwardSampler


def phenotype_given_genotype(variables, prob_trait_genotype):
    v1, v2, v3 = variables

    dims = [v1.k, v2.k, v3.k]
    values = np.zeros(np.prod(dims))

    for i in range(len(values)):
        assg = np.unravel_index(i, dims)

        values[i] = prob_trait_genotype[assg[1], assg[2]]
        if not assg[0]:
            values[i] = 1 - values[i]

    return CPD([v3, v1, v2], values)


def allele_given_parent_alleles(allele, p_alleles):
    dims = [allele.k, p_alleles[0].k, p_alleles[1].k]

    values = np.zeros(np.prod(dims))

    for i in range(len(values)):
        assg = np.unravel_index(i, dims)

        if assg[0] == assg[1]:
            values[i] += 0.5
        if assg[0] == assg[2]:
            values[i] += 0.5

    return CPD([allele, p_alleles[0], p_alleles[1]], values)


def build_genetic_network(parents, allele_freqs, prob_trait_genotype):
    prob_trait_genotype = np.array(prob_trait_genotype)

    variables = {}
    for person in parents.keys():
        v1 = RandomVar(person + '_allele_1', len(allele_freqs))
        v2 = RandomVar(person + '_allele_2', len(allele_freqs))
        v3 = RandomVar(person + '_trait', 2)

        variables[person] = [v1, v2, v3]

    factors = []
    for person in parents.keys():
        v1, v2, v3 = variables[person]

        if parents[person]:
            p1_vars = variables[parents[person][0]]
            p2_vars = variables[parents[person][1]]

            f_allele1 = allele_given_parent_alleles(v1, p1_vars)
            f_allele2 = allele_given_parent_alleles(v2, p2_vars)
        else:
            f_allele1 = CPD([v1], allele_freqs)
            f_allele2 = CPD([v2], allele_freqs)

        f_phenotype = phenotype_given_genotype(
            variables[person], prob_trait_genotype)

        factors += [f_allele1, f_allele2, f_phenotype]

    return BayesianNetwork(factors)


def main():
    parents = {'alice': [], 'bob': [], 'eve': ['bob', 'alice']}
    allele_freqs = [0.9, 0.1]
    prob_trait_genotype = [[0.0, 0.0], [0.0, 1.0]]

    bn = build_genetic_network(parents, allele_freqs, prob_trait_genotype)

    # Examples
    alice_trait = RandomVar('alice_trait', 2)
    bob_trait = RandomVar('bob_trait', 2)
    eve_trait = RandomVar('eve_trait', 2)

#    alice_allele_1 = RandomVar('alice_allele_1', 2)
#    alice_allele_2 = RandomVar('alice_allele_2', 2)
#
#    bob_allele_1 = RandomVar('bob_allele_1', 2)
#    bob_allele_2 = RandomVar('bob_allele_2', 2)
#
#    eve_allele_1 = RandomVar('eve_allele_1', 2)
#    eve_allele_2 = RandomVar('eve_allele_2', 2)

    ve = JointMarginalization(bn)

    print(ve.posterior([eve_trait]))
    print(ve.posterior([eve_trait], [(bob_trait, 1)]))
    print(ve.posterior([alice_trait, bob_trait], [(eve_trait, 1)]))

    fs = ForwardSampler(bn)
    fs.sample(1000)
    print(fs.posterior([(alice_trait, 0), (bob_trait, 0),
                        (eve_trait, 1)]) / fs.posterior([(eve_trait, 1)]))
                        
    
    print(bn.graph())


if __name__ == "__main__":
    main()
