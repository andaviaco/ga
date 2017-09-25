import random as rd


class Individual(object):
    fitness = None

    """docstring for Individual"""
    def __init__(self, genotype):
        super(Individual, self).__init__()
        self.genotype = genotype

    def mutate(self, mutation_p):
        r = rd.random()

        if r < mutation_p:
            allele_index = rd.randrange(0, len(self.genotype))

            if self.genotype[allele_index] == 0:
                self.genotype[allele_index] = 1
            else:
                self.genotype[allele_index] = 0

        return self

    def __repr__(self):
        return f'<Individual {self.genotype} : {self.fitness}>'
