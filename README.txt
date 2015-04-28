About
-----
Probabilistic graphical models in python.

This code is intended mainly as proof of concept of the algorithms presented in 
[1]. The implementations are not particularly clear, efficient, well tested or 
numerically stable. We advise against using this software for nondidactic 
purposes.

This software is licensed under the MIT License. 

Features
--------

Models:
    Bayesian network (table conditional probability distributions)
    Markov network (table potentials)
    Influence diagram

Inference:
    Variable elimination
    Forward sampling
    Gibbs sampling

Learning:
    Parameter learning (maximum likelihood, uniform BDe, expectation
    maximization for missing data)
    Structure learning (local search, likelihood score, BIC score, Bayesian 
    score)

Examples
--------
See the examples directory.

References
----------
[1] Koller, D. and Friedman, N. Probabilistic Graphical Models: Principles and
 Techniques. The MIT Press. 2009.