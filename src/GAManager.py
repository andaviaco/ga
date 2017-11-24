import numpy as np
import random as rand
import pprint as pp

from operator import attrgetter

from Individual import Individual


class GAManager(object):
    """docstring for GAManager"""
    def __init__(
        self,
        npopulation,
        ngen,
        fn_eval,
        *,
        mutation_p=0.01,
        fn_lb=[-10, -10],
        fn_ub=[10, 10]
    ):
        super(GAManager, self).__init__()
        self.npopulation = npopulation
        self.ngen = ngen
        self.fn_eval = fn_eval

        self.mutation_p = mutation_p
        self.fn_lb = np.array(fn_lb)
        self.fn_ub = np.array(fn_ub)

    def optimize(self):
        self.initialize()

        for i in range(self.ngen):
            self.update_fitness()

            population_iterations = int(len(self.population) / 2)
            new_population = []

            for i in range(population_iterations):
                parents = self.selection()
                offsprings = self.crossover(*parents)

                new_population += offsprings

            self.population = new_population
            self.mutation()

        self.update_fitness()
        fittest_individual = max(self.population, key=attrgetter('fitness'))

        return fittest_individual.genotype

    def initialize(self):
        self.population = [self.create_agent() for i in range(self.npopulation)]

    def update_fitness(self):
        for individual in self.population:
            individual.fitness = self.fitness(individual.genotype)

    def selection(self):
        fs = [individual.fitness for individual in self.population]
        parent_1 = self.select_parent(self.population, fs).pop()

        partial_fs = [individual.fitness for individual in self.population if individual != parent_1]
        partial_population = [individual for individual in self.population if individual != parent_1]

        parent_2 = self.select_parent(partial_population, partial_fs).pop()

        return (parent_1, parent_2)

    def crossover(self, parent_1, parent_2):
        crosspoint = rand.randint(1, len(parent_1.genotype) - 1) # 0 < crosspoint < len()

        offsprings = self.create_offsprings(parent_1, parent_2, crosspoint)

        return offsprings

    def mutation(self):
        for agent in self.population:
            agent.mutate(self.mutation_p, self.fn_lb, self.fn_ub)

    def select_parent(self, population, fs):
        fs_sum = sum(fs)
        normalized_fs = [fs_sum / (i + 1) for i in fs]

        return rand.choices(population, fs)

    def create_offsprings(self, parent_1, parent_2, crosspoint):
        parent_1_part_1 = parent_1.genotype[:crosspoint]
        parent_1_part_2 = parent_1.genotype[crosspoint:]

        parent_2_part_1 = parent_2.genotype[:crosspoint]
        parent_2_part_2 = parent_2.genotype[crosspoint:]

        new_solution_1 = np.concatenate((parent_1_part_1, parent_2_part_2))
        new_solution_2 = np.concatenate((parent_2_part_1, parent_1_part_2))

        rounded_solution_1 = np.around(new_solution_1, decimals=4)
        rounded_solution_2 = np.around(new_solution_2, decimals=4)

        offspring_1 = Individual(rounded_solution_1, 0.0)
        offspring_2 = Individual(rounded_solution_2, 0.0)

        return (offspring_1, offspring_2)

    def fitness(self, solution):
        return 1 / ( 1 + self.fn_eval(solution))

    def create_agent(self):
        solution = self.random_vector(self.fn_lb, self.fn_ub)

        return Individual(solution, 0.0)

    def random_vector(self, lb, ub):
        r = np.array([rand.random() for i in range(len(lb))])
        solution = lb + (ub - lb) * r

        return np.around(solution, decimals=4)
