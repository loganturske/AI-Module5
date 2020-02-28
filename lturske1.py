from random import gauss
import random
import math


def sphere(shift, xs):
    return sum([(x - shift) ** 2 for x in xs])


parameters = {
    "f": lambda xs: sphere(0.5, xs),
    "minimization": True,
    'population_size': 10,
    'individual_length': 10,
    'limit': 500,
    'crossover_rate': .2,
    'mutation_rate': .1,
    'divide': 3,
    'mu': .3,
    'sigma': 0.0

    # put other parameters in here.
}


def generate_binary_population(params):
    return [random.randint(0, 1) for i in range(params.get('population_size') * params.get('individual_length'))]


def generate_real_population(params):
    return [random.uniform(-5.12, 5.12) for i in range(params.get('population_size'))]


def get_individuals(params, pop):
    individuals = [pop[i:i + 10] for i in range(0, len(pop), 10)]
    for index, individual in enumerate(individuals):
        temp_str = ''
        for bit in individual:
            temp_str += str(bit)
        individuals[index] = temp_str
    return individuals


def get_pheno(params, pop):
    individuals = get_individuals(params, pop)
    for index, individual in enumerate(individuals):
        individuals[index] = individual_value_func(individual)
    return individuals


def individual_value_func(individual):
    temp_str = ''
    for bit in individual:
        temp_str += str(bit)
    return ((int(temp_str, 2) - 512) / 100)


def pick_parents(params, pop):
    temp_pop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    weights = []
    max = 0
    for index in temp_pop:
        max += 1 / (1 + params.get('f')([(individual_value_func(
            pop[(index * params.get('individual_length')):(index * params.get('individual_length')) + 10]))]))
    for index in temp_pop:
        val = 1 / (1 + params.get('f')([(individual_value_func(
            pop[(index * params.get('individual_length')):(index * params.get('individual_length')) + 10]))]))
        weights.append(val / max)

    father = random.choices(temp_pop, weights)[0]

    mother = random.choices(temp_pop, weights)[0]
    while father == mother:
        mother = random.choices(temp_pop, weights)[0]

    return father * 10, mother * 10


def pick_parents_real(params, pop):
    temp_pop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    weights = []
    max = 0
    for index in temp_pop:
        max += 1 / (1 + params.get('f')([pop[index]]))
    for index in temp_pop:
        val = 1 / (1 + params.get('f')([pop[index]]))
        weights.append(val / max)

    father = random.choices(temp_pop, weights)[0]

    mother = random.choices(temp_pop, weights)[0]
    while father == mother:
        mother = random.choices(temp_pop, weights)[0]
    return father, mother


def reproduce(divide, father, mother, pop):
    son = pop[father:father + divide] + pop[mother + divide:mother + 10]
    daughter = pop[mother:mother + divide] + pop[father + divide:father + 10]
    return son, daughter


def reproduce_real(father, mother):
    father_vals = math.modf(father)
    mother_vals = math.modf(mother)
    # print("Father: " + str(father) +" Mother  : " + str(mother))

    son = (abs(father_vals[1]) + abs(mother_vals[0])) /2
    daughter = father_vals[0] + mother_vals[1]
    # print("Son   : " + str(son) +   " Daughter: " + str(daughter) )

    return son, daughter


def mutate_real(child, mu, sigma):
    return gauss(mu, sigma)


def mutate(child):
    index = random.randint(0, 9)
    child[index] = random.randint(0, 1)
    return child


def binary_ga(parameters, debug=False):
    pop = generate_binary_population(parameters)
    generation = 0
    limit = parameters.get('limit')
    best_so_far_f_val = 0
    best_so_far_geno = None
    best_so_far_func_val = None
    best_so_far_pheno = None
    best_so_far_generation = 0
    while generation < limit:

        generation += 1
        func_val = parameters.get('f')(get_pheno(parameters, pop))
        f_val = 1 / (1 + func_val)
        if f_val > best_so_far_f_val:
            best_so_far_generation = generation
            best_so_far_f_val = f_val
            best_so_far_geno = pop
            best_so_far_func_val = func_val
            best_so_far_pheno = get_pheno(parameters, pop)
        if debug and generation % 25 == 0:
            print("Generation: " + str(generation))
            print("    Best so far   : " + str(best_so_far_generation))
            print("    The Genotype  : " + str(best_so_far_geno))
            print("    The Phenotype : " + str(best_so_far_pheno))
            print("    The F Value   : " + str(best_so_far_f_val))
            print("    The Func Value: " + str(best_so_far_func_val))

        next_pop = []
        for index in range(int((len(pop) / parameters.get('individual_length')) / 2)):

            parents = pick_parents(parameters, pop)
            children = reproduce(parameters.get('divide'), parents[0], parents[1], pop)

            if random.random() < parameters.get('crossover_rate'):
                next_pop = next_pop + pop[parents[0]:parents[0] + 10]
                next_pop = next_pop + pop[parents[1]:parents[1] + 10]
            else:
                child_one = children[0]
                child_two = children[1]
                if random.random() < parameters.get('mutation_rate'):
                    child_one = mutate(child_one)
                if random.random() < parameters.get('mutation_rate'):
                    child_two = mutate(child_two)
                next_pop = next_pop + child_one
                next_pop = next_pop + child_two
        pop = next_pop
    print("    Best so far   : " + str(best_so_far_generation))
    print("    The Genotype  : " + str(best_so_far_geno))
    print("    The Phenotype : " + str(best_so_far_pheno))
    print("    The F Value   : " + str(best_so_far_f_val))
    print("    The Func Value: " + str(best_so_far_func_val))


# binary_ga(parameters, False)


def real_ga(parameters, debug=False):
    pop = generate_real_population(parameters)
    generation = 0
    limit = parameters.get('limit')
    best_so_far_f_val = 0
    best_so_far_geno = None
    best_so_far_func_val = None
    best_so_far_pheno = None
    best_so_far_generation = 0
    while generation < limit:
        generation += 1
        func_val = parameters.get('f')(pop)
        f_val = 1 / (1 + func_val)
        if f_val > best_so_far_f_val:
            best_so_far_generation = generation
            best_so_far_f_val = f_val
            best_so_far_geno = pop
            best_so_far_func_val = func_val
            best_so_far_pheno = pop
        if debug and generation % 25 == 0:
            print("Generation: " + str(generation))
            print("    Best so far   : " + str(best_so_far_generation))
            print("    The Genotype  : " + str(best_so_far_geno))
            print("    The Phenotype : " + str(best_so_far_pheno))
            print("    The F Value   : " + str(best_so_far_f_val))
            print("    The Func Value: " + str(best_so_far_func_val))

        next_pop = []
        for index in range(int(len(pop))):

            parents = pick_parents_real(parameters, pop)
            children = reproduce_real(pop[parents[0]], pop[parents[1]])

            if random.random() < parameters.get('crossover_rate'):
                next_pop.append(pop[parents[0]])
                # next_pop.append(pop[parents[1]])
            else:
                child_one = children[0]
                # child_two = children[1]
                if random.random() < parameters.get('mutation_rate'):
                    child_one = mutate_real(child_one, parameters.get('mu'), parameters.get('sigma'))
                # if random.random() < parameters.get('mutation_rate'):
                    # child_two = mutate_real(child_two, parameters.get('mu'), parameters.get('sigma'))
                next_pop = next_pop + [child_one]
                # next_pop = next_pop + [child_two]
        pop = next_pop
    print("    Best so far   : " + str(best_so_far_generation))
    print("    The Genotype  : " + str(best_so_far_geno))
    print("    The Phenotype : " + str(best_so_far_pheno))
    print("    The F Value   : " + str(best_so_far_f_val))
    print("    The Func Value: " + str(best_so_far_func_val))


real_ga(parameters, False)