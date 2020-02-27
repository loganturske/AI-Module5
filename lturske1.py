from random import gauss
import random


def sphere(shift, xs):
    return sum([(x - shift) ** 2 for x in xs])


parameters = {
    "f": lambda xs: sphere(0.5, xs),
    "minimization": True,
    'population_size': 10,
    'individual_length': 10,
    'limit': 2,
    'crossover_rate': .3,
    'mutation_rate': .05,
    'divide': 5

    # put other parameters in here.
}


def generate_binary_population(params):
    return [random.randint(0, 1) for i in range(params.get('population_size') * params.get('individual_length'))]


def get_individuals(params, pop):
    individuals = [pop[i:i + 10] for i in range(0, len(pop), 10)]
    for index, individual in enumerate(individuals):
        temp_str = ''
        for bit in individual:
            temp_str += str(bit)
        individuals[index] = temp_str
    return individuals


def evaluate_binary_population(params, pop):
    individuals = get_individuals(params, pop)
    for index, individual in enumerate(individuals):
        individuals[index] = individual_value_func(individual)
    return params.get('f')(individuals)


def individual_value_func(individual):
    temp_str = ''
    for bit in individual:
        temp_str += str(bit)
    return 1 / (1 + ((int(temp_str, 2) - 512) / 100))


def pick_parents(params, pop):
    temp_pop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    weights = []
    max = 0
    for index in temp_pop:
        max += abs(individual_value_func(pop[(index*params.get('individual_length')):(index*params.get('individual_length')) + 9]))
    for index in temp_pop:
        val = abs(individual_value_func(pop[(index*params.get('individual_length')):(index*params.get('individual_length')) + 9]))
        weights.append(val/max)
    father = random.choices(temp_pop, weights)[0]
    mother = random.choices(temp_pop, weights)[0]
    return father*10, mother*10


def remove_individual(params, individual, pop):
    temp_pop = get_individuals(params, pop)
    ret_pop = []
    for test_individual in temp_pop:
        if test_individual == individual:
            continue
        else:
            ret_pop = ret_pop + list(test_individual)
    return ret_pop


def reproduce(divide, father, mother, pop):
    son = pop[father:father+divide] + pop[mother+divide:mother+9]
    daughter = pop[mother:mother+divide] + pop[father+divide:father+9]
    return son, daughter


def mutate(child):
    index = random.randint(0, 9)
    child[index] = random.randint(0, 1)
    return child


def binary_ga(parameters, debug=True):
    pop = generate_binary_population(parameters)
    if debug:
        print("Starting pop: " + str(pop))
    generation = 0
    limit = parameters.get('limit')
    while generation < limit:
        if debug:
            print("### New Generation ###")
        generation += 1
        eval = evaluate_binary_population(parameters, pop)
        if debug:
            print("Evaluated:" + str(eval))
        next_pop = []
        for index in range(int((len(pop) / parameters.get('individual_length')) / 2)):

            parents = pick_parents(parameters, pop)

            if debug:
                print("Parents : " + str(parents))

            children = reproduce(parameters.get('divide'), parents[0], parents[1], pop)
            if debug:
                print("Children: " + str(children))
            if random.random() < parameters.get('crossover_rate'):
                next_pop = next_pop + pop[parents[0]:parents[0]+9]
                next_pop = next_pop + pop[parents[1]:parents[1]+9]
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
        print("NEXT: " + str(pop))


binary_ga(parameters)


def real_ga(parameters):
    pass
