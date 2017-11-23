import numpy as np
import random as rd
import pprint as pp

from operator import attrgetter

from Individual import Individual


class GAManager(object):
    """docstring for GAManager"""
    def __init__(self, npopulation, ngen, evaluate, *, mutation_p=0.01, evalsegments=16, llow=0, lhigh=1):
        super(GAManager, self).__init__()
        self.npopulation = npopulation
        self.ngen = ngen
        self.evaluate = evaluate

        self.mutation_p = mutation_p
        self.evalsegments = evalsegments
        self.llow = llow
        self.lhigh = lhigh

    def optimize(self):
        self.population = self.random_init(self.npopulation, self.evalsegments)

        for i in range(self.ngen):
            self.fitness_assign(self.population)

            population_iterations = int(len(self.population) / 2)
            new_population = []

            for i in range(population_iterations):
                parents = self.selection()
                offsprings = self.crossover(*parents)

                new_population += offsprings

            self.population = self.mutation(new_population)

        self.fitness_assign(self.population)
        fittest_individual = max(self.population, key=attrgetter('fitness'))

        # pp.pprint(self.population)
        # print('FITTEST', fittest_individual)

        return self.get_phenotype(self.descode_genotype(fittest_individual.genotype, pad=(4, 0))[0])

    def random_init(self, population, segments):
        rands = np.array([[rd.randint(0, segments-1)] for i in range(population)], dtype=np.uint8)
        genotypes = np.unpackbits(rands, axis=1)
        genotypes = genotypes[:, -4:] #limit bits to the most-right 4 alleles

        return [Individual(genotype) for genotype in genotypes]

    def fitness_assign(self, population):
        genotypes = np.array([individual.genotype for individual in population])
        rep_genotypes = self.descode_genotype(genotypes, axis=1)
        tuples = zip(population, rep_genotypes)

        for individual, genotype_rep in tuples:
            individual.fitness = self.calc_fitness(self.evaluate, genotype_rep[0])

    def selection(self):
        fs = [individual.fitness for individual in self.population]

        parent_1_index = self.select_parent(fs)
        parent_2_index = self.select_parent(fs)

        parent_1 = self.population[parent_1_index]
        parent_2 = self.population[parent_2_index]

        return (parent_1, parent_2)

    def crossover(self, parent_1, parent_2):
        crosspoint = rd.randint(1, len(parent_1.genotype) - 1) # 0 < crosspoint < len()

        offsprings = self.create_offsprings(parent_1, parent_2, crosspoint)

        return offsprings

    def mutation(self, population):
        return list(map(lambda x: x.mutate(self.mutation_p), population))

    def select_parent(self, fs):
        p = rd.uniform(0, sum(fs))

        for i, f in enumerate(fs):
            if p <= 0:
                break
            p -= f
        return i

    def create_offsprings(self, parent_1, parent_2, crosspoint):
        parent_1_part_1 = parent_1.genotype[:crosspoint]
        parent_1_part_2 = parent_1.genotype[crosspoint:]

        parent_2_part_1 = parent_2.genotype[:crosspoint]
        parent_2_part_2 = parent_2.genotype[crosspoint:]

        offspring_1 = Individual(np.concatenate((parent_1_part_1, parent_2_part_2)))
        offspring_2 = Individual(np.concatenate((parent_2_part_1, parent_1_part_2)))

        return (offspring_1, offspring_2)

    def descode_genotype(self, genotypes, *, axis=0, pad=((0,0), (4,0))):
        paded_genotypes = np.pad(genotypes, pad, 'constant', constant_values=0)
        rep_genotypes = np.packbits(paded_genotypes, axis=axis)

        return rep_genotypes

    def calc_fitness(self, evaluate, value):
        return -evaluate(self.get_phenotype(value)) + 10 # max -fn() + offset(10)

    def get_phenotype(self, value):
        step = abs(self.lhigh - self.llow) / (self.evalsegments - 1)

        return (value * step) + self.llow
