import random


#the population will be :
# a set of sets -> each set containing : [N_value_Added,P_value_Added,K_value_Added,Water_Added,irrigation_frequency] -> the added value are the final one's. 



def roulette_wheel_selection(population, problem):
    # Compute fitness for each individual (higher fitness = better)
    fitness_values = []
    for individual in population:
        #evaluate here will be the heuristic that we are going to use, change its name with heuristic function name !!!!!!!
        fitness = problem.evaluate(individual)
        fitness_values.append(fitness)
    total_fitness = sum(fitness_values)
    # Generate a random threshold
    threshold = random.uniform(0, total_fitness)
    # Iterate through the population and select the individual
    cumulative_fitness = 0
    for individual, fitness in zip(population, fitness_values):
        cumulative_fitness += fitness
        if cumulative_fitness >= threshold:
            return individual

    # In case of rounding errors, return the last individual
    return population[-1]
