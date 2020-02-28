from random import gauss
import random
import math


def sphere(shift, xs):
    return sum([(x - shift) ** 2 for x in xs])


parameters = {
    "f": lambda xs: sphere(0.5, xs),
    "minimization": True,
    'population_size': 10,
    'individual_vars': 10,
    'individual_var_length': 10,
    'limit': 10000,
    'crossover_rate': .2,
    'mutation_rate': .1,
    'divide': 3,
    'mu': .5,
    'sigma': 0.0

    # put other parameters in here.
}


def generate_binary_individual(params):
    return [random.randint(0, 1) for i in range(params.get('individual_vars') * params.get('individual_var_length'))]


def generate_real_individual(params):
    return [random.uniform(-5.12, 5.12) for i in range(params.get('individual_vars'))]


def get_vars(params, individual):
    vars = [individual[i:i + parameters.get('individual_var_length')] for i in range(0, params.get('individual_vars')* params.get('individual_var_length'), 10)]
    for index, individual in enumerate(vars):
        temp_str = ''
        for bit in individual:
            temp_str += str(bit)
        vars[index] = temp_str
    return vars


def get_pheno(params, individual):
    vars = get_vars(params, individual)
    for index, individual in enumerate(vars):
        vars[index] = individual_value_func(individual)
    return vars


def individual_value_func(individual):
    temp_str = ''
    for bit in individual:
        temp_str += str(bit)
    return ((int(temp_str, 2) - 512) / 100)


def pick_parents(params, pop, real=False):
    temp_pop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    weights = []
    max = 0
    for individual in pop:
        if real:
            max += 1 / (1 + params.get('f')(individual))
        else:
            max += 1 / (1 + params.get('f')(get_pheno(params, individual)))

    for individual in pop:
        if real:
            val = 1 / (1 + params.get('f')(individual))
            weights.append(val / max)
        else:
            val = 1 / (1 + params.get('f')(get_pheno(params, individual)))
            weights.append(val / max)

    father = random.choices(temp_pop, weights)[0]

    mother = random.choices(temp_pop, weights)[0]
    while father == mother:
        mother = random.choices(temp_pop, weights)[0]

    return father, mother



def reproduce(divide, father, mother, pop):
    son = pop[father][:divide] + pop[mother][divide:]
    daughter = pop[mother][:divide] + pop[father][divide:]
    return son, daughter


def mutate(parameters, child, real=False):
    index = random.randint(0, parameters.get('individual_vars')-1)
    if real:
        child[index] = gauss(parameters.get('mu'), parameters.get('sigma'))
    else:
        for i in range(parameters.get('individual_var_length')-1):
            child[index*parameters.get('individual_vars') + i] = random.randint(0, 1)
    return child


def binary_ga(parameters, debug=False):
    pop = [generate_binary_individual(parameters) for x in range(parameters.get('population_size'))]
    generation = 0
    limit = parameters.get('limit')
    best_so_far_f_val = 0
    best_so_far_geno = None
    best_so_far_func_val = None
    best_so_far_pheno = None
    best_so_far_generation = 0
    while generation < limit:

        generation += 1

        for individual in pop:

            func_val = parameters.get('f')(get_pheno(parameters, individual))
            f_val = 1 / (1 + func_val)
            if f_val > best_so_far_f_val:
                best_so_far_generation = generation
                best_so_far_f_val = f_val
                best_so_far_geno = individual
                best_so_far_func_val = func_val
                best_so_far_pheno = get_pheno(parameters, individual)
            if debug and generation % 25 == 0:
                print("Generation: " + str(generation))
                print("    Best so far   : " + str(best_so_far_generation))
                print("    The Genotype  : " + str(best_so_far_geno))
                print("    The Phenotype : " + str(best_so_far_pheno))
                print("    The F Value   : " + str(best_so_far_f_val))
                print("    The Func Value: " + str(best_so_far_func_val))

        next_pop = []
        for index in range(int(parameters.get('population_size') / 2)):

            parents = pick_parents(parameters, pop)
            children = reproduce(parameters.get('divide'), parents[0], parents[1], pop)
            if random.random() < parameters.get('crossover_rate'):
                next_pop.append(pop[parents[0]])
                next_pop.append(pop[parents[1]])
            else:
                child_one = children[0]
                child_two = children[1]
                if random.random() < parameters.get('mutation_rate'):
                    child_one = mutate(parameters, child_one)
                if random.random() < parameters.get('mutation_rate'):
                    child_two = mutate(parameters, child_two)
                next_pop.append(child_one)
                next_pop.append(child_two)
        pop = next_pop
    print("    Best so far   : " + str(best_so_far_generation))
    print("    The Genotype  : " + str(best_so_far_geno))
    print("    The Phenotype : " + str(best_so_far_pheno))
    print("    The F Value   : " + str(best_so_far_f_val))
    print("    The Func Value: " + str(best_so_far_func_val))


# binary_ga(parameters, False)


def real_ga(parameters, debug=False):
    pop = [generate_real_individual(parameters) for x in range(parameters.get('population_size'))]
    generation = 0
    limit = parameters.get('limit')
    best_so_far_f_val = 0
    best_so_far_geno = None
    best_so_far_func_val = None
    best_so_far_pheno = None
    best_so_far_generation = 0
    while generation < limit:

        generation += 1

        for individual in pop:

            func_val = parameters.get('f')(individual)
            f_val = 1 / (1 + func_val)
            if f_val > best_so_far_f_val:
                best_so_far_generation = generation
                best_so_far_f_val = f_val
                best_so_far_geno = individual
                best_so_far_func_val = func_val
                best_so_far_pheno = individual
            if debug and generation % 25 == 0:
                print("Generation: " + str(generation))
                print("    Best so far   : " + str(best_so_far_generation))
                print("    The Genotype  : " + str(best_so_far_geno))
                print("    The Phenotype : " + str(best_so_far_pheno))
                print("    The F Value   : " + str(best_so_far_f_val))
                print("    The Func Value: " + str(best_so_far_func_val))

        next_pop = []
        for index in range(int(parameters.get('population_size') / 2)):

            parents = pick_parents(parameters, pop, True)
            children = reproduce(parameters.get('divide'), parents[0], parents[1], pop)
            if random.random() < parameters.get('crossover_rate'):
                next_pop.append(pop[parents[0]])
                next_pop.append(pop[parents[1]])
            else:
                child_one = children[0]
                child_two = children[1]
                if random.random() < parameters.get('mutation_rate'):
                    child_one = mutate(parameters, child_one, True)
                if random.random() < parameters.get('mutation_rate'):
                    child_two = mutate(parameters, child_two, True)
                next_pop.append(child_one)
                next_pop.append(child_two)
        pop = next_pop
    print("    Best so far   : " + str(best_so_far_generation))
    print("    The Genotype  : " + str(best_so_far_geno))
    print("    The Phenotype : " + str(best_so_far_pheno))
    print("    The F Value   : " + str(best_so_far_f_val))
    print("    The Func Value: " + str(best_so_far_func_val))


real_ga(parameters, True)