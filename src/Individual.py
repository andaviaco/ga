import random as rand
import numpy as np


class Individual(object):
    fitness = None

    """docstring for Individual"""
    def __init__(self, genotype, initial_fitness):
        super(Individual, self).__init__()
        self.genotype = genotype
        self.fitness = initial_fitness

    def mutate(self, mutation_p, lb, ub):
        mutated_genotype = []

        for i, gen in enumerate(self.genotype):
            r = rand.random()

            if r < mutation_p:
                mutated_value = lb[i] + (ub[i] - lb[i]) * r
                mutated_genotype.append(mutated_value)
            else:
                mutated_genotype.append(gen)

        self.genotype = np.around(mutated_genotype, decimals=4)

    def __repr__(self):
        return f'<Individual {self.genotype} : {self.fitness}>'
