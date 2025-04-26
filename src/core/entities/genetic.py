import random
import pandas as pd
from environment import optimization_problem, Node

class GeneticAlgorithm:

    def __init__(self, problem, population_size=50, generations=1000, mutation_rate=0.1, 
                 tournament_size=3, selection_method='tournament'):
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.selection_method = selection_method

    def tournament_selection(self, population):
        """
        Tournament selection method to select individuals for reproduction.
        
        Parameters:
            population (list): List of individuals
            
        Returns:
            dict: Selected individual
        """
        # Randomly select tournament_size individuals
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        
        # Find the best individual in the tournament
        best = None
        best_fitness = float('inf')
        
        for individual in tournament:
            # Apply action to initial state
            result_state = self.problem.apply_action(self.problem.initial_state.copy(), individual)
            # Calculate fitness
            fitness = self.problem.cost_function(individual) + self.problem.heuristic(result_state)
            
            if fitness < best_fitness:
                best = individual
                best_fitness = fitness
        
        return best

    def roulette_wheel_selection(self, population, problem):
        # Compute fitness for each individual (higher fitness = better)
        fitness_values = []
        for individual in population:
            # We want to minimize cost, so invert the fitness
            result_state = problem.apply_action(problem.initial_state.copy(), individual)
            cost = problem.cost_function(individual) + problem.heuristic(result_state)
            fitness = 1 / (cost + 0.1)  # Add small value to avoid division by zero
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

    def mutate(self, individual):
        """Mutate one parameter of the action"""
        mutated = individual.copy()
        param = random.choice(["water_added", "N_added", "P_added", "K_added"])
        
        # Get possible values for this parameter (could use the get_valid_action() too)
        options = {
            "water_added": [-10, -5, 0, 10, 20],
            "N_added": [-10, -5, 0, 5, 10, 20],
            "P_added": [-10, -5, 0, 5, 10, 20],
            "K_added": [-10, -5, 0, 5, 10, 20]
        }[param]
        
        # Select new value different from current
        new_val = random.choice([x for x in options if x != mutated[param]])
        mutated[param] = new_val
        return mutated

    def uniform_crossover_continuous(self, parent1, parent2):
        size = len(parent1)
        if size != len(parent2):
            raise ValueError("Parents must have same number of resources")
        
        offspring = []
        
        for i in range(size):
            # Randomly choose gene from parent1 or parent2
            if random.random() < 0.5:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
        
        return offspring

    def two_point_crossover_continuous(self, parent1, parent2):
        """
        Two-point crossover for continuous values.
        Simply swaps a middle segment between two parents.
        
        Parameters:
            parent1 (list): Resource values from first parent
            parent2 (list): Resource values from second parent
            
        Returns:
            list: Offspring resource values
        """
        size = len(parent1)
        if size != len(parent2):
            raise ValueError("Parents must have same number of resources")
        
        # Initialize offspring by copying parent1
        offspring = parent1.copy()
        
        # 1. Select two random crossover points
        pt1 = random.randint(0, size - 2)
        pt2 = random.randint(pt1 + 1, size - 1)
        
        # 2. Swap the segment between pt1 and pt2 from parent2
        for i in range(pt1, pt2 + 1):
            offspring[i] = parent2[i]
        
        return offspring

    def resource_crossover(self, parent1, parent2, method='two_point'):
        resource_keys = ['water_added', 'N_added', 'P_added', 'K_added']
        
        # Verify parents have the required structure
        if not all(key in parent1 and key in parent2 for key in resource_keys):
            raise ValueError("Parents must contain all required resource keys")
        
        # Create offspring with non-resource attributes from parent1
        offspring = {k: parent1[k] for k in parent1 if k not in resource_keys}
        
        # Convert resource values to lists for crossover processing
        parent1_res = [parent1[k] for k in resource_keys]
        parent2_res = [parent2[k] for k in resource_keys]
        
        # Select crossover method
        if method == 'two_point':
            offspring_res = self.two_point_crossover_continuous(parent1_res, parent2_res)
        elif method == 'uniform':
            offspring_res = self.uniform_crossover_continuous(parent1_res, parent2_res)
        else:
            raise ValueError(f"Unknown crossover method: {method}")
        
        # Add crossed-over resources to offspring
        for i, key in enumerate(resource_keys):
            offspring[key] = offspring_res[i]
        
        return offspring

    def solve(self):
        """
        Run the genetic algorithm to find an optimal solution for the resource allocation problem.
        
        Returns:
            tuple: (best_action, result_state, cost) - The optimal action, resulting state, and its cost
        """
        # Initialize population with random valid actions
        population = []
        for _ in range(self.population_size):
            # Generate random valid actions
            valid_actions = self.problem.get_valid_actions()
            individual = random.choice(valid_actions)
            population.append(individual)
        
        best_individual = None
        best_fitness = float('inf')  # We're minimizing cost
        best_state = None
        
        print(f"Starting GA optimization with {self.population_size} individuals for {self.generations} generations")
        
        # Main GA loop
        for generation in range(self.generations):
            # Evaluate population
            fitness_values = []
            for individual in population:
                # Apply action to initial state
                result_state = self.problem.apply_action(self.problem.initial_state.copy(), individual)
                # Calculate fitness (lower is better - we want minimum cost)
                fitness = self.problem.cost_function(individual) + self.problem.heuristic(result_state)
                fitness_values.append((individual, result_state, fitness))
            
            # Sort by fitness (lower is better)
            fitness_values.sort(key=lambda x: x[2])
            
            # Update best individual if found
            if fitness_values[0][2] < best_fitness:
                best_individual = fitness_values[0][0]
                best_state = fitness_values[0][1]
                best_fitness = fitness_values[0][2]
                
            # Print progress every 50 generations
            if generation % 50 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.2f}")
                print(f"  Best action: Water={best_individual['water_added']}L, " +
                      f"N={best_individual['N_added']}kg, " +
                      f"P={best_individual['P_added']}kg, " +
                      f"K={best_individual['K_added']}kg")
            
            # Check if we've reached the goal state
            if self.problem.goalstate(Node(best_state)):
                print(f"Goal state reached at generation {generation}!")
                break
                
            # Create new population
            new_population = []
            
            # Elitism - keep the best 2 individuals
            new_population.extend([fitness_values[0][0], fitness_values[1][0]])
            
            # Fill the rest of the population with offspring
            while len(new_population) < self.population_size:
                # Select parents
                if self.selection_method == 'tournament':
                    parent1 = self.tournament_selection(population)
                    parent2 = self.tournament_selection(population)
                elif self.selection_method == 'roulette':
                    parent1 = self.roulette_wheel_selection(population, self.problem)
                    parent2 = self.roulette_wheel_selection(population, self.problem)
                else:
                    raise ValueError(f"Unknown selection method: {self.selection_method}")
                
                # Crossover
                offspring = self.resource_crossover(parent1, parent2)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring = self.mutate(offspring)
                
                new_population.append(offspring)
            
            # Replace old population with new one
            population = new_population
        
        # Final result
        result_state = self.problem.apply_action(self.problem.initial_state.copy(), best_individual)
        cost = self.problem.cost_function(best_individual)
        
        return best_individual, result_state, cost


# Initial state definition
initial_state = {
    'soil_moisture': 5.28,
    'N': 24.33,
    'P': 24.83,
    'K': 20.33,
    'ph': 6.5,
    'label': "rice",
    'soil_type': 3,
    'temperature': 28,
    'crop_density': 14,
    'humidity': 45,
    'rainfall_forecast': 6,
    'growth_stage': 1,
    'growth_type': "monocot",
    'water_availability': "medium",
    'fertilizer_availability': "high",
    'irrigation_system': 'drip',
    'water_used': 0.0,
    'fertilizer_used': 0.0
}

def test_optimization_ga():
    print("\n===== Smart Farming Resource Optimization GA =====")
    
    df = pd.read_csv("src\core\entities\FS25.csv")
    problem = optimization_problem(initial_state, df)

    # Only testing selection methods now
    for selection_method in ['tournament', 'roulette']:
        print(f"\n=== Testing with {selection_method.title()} Selection ===")
        ga = GeneticAlgorithm(
            problem,
            population_size=30,
            generations=10,
            mutation_rate=0.5,
            selection_method=selection_method
        )
        
        best_action, result_state, cost = ga.solve()
        
        print("\n=== OPTIMAL RESOURCE ALLOCATION ===")
        print(f"{'Resource':<15} | {'Amount Added':<15}")
        print("-" * 30)
        print(f"{'Water':<15} | {best_action['water_added']:<15} L")
        print(f"{'Nitrogen (N)':<15} | {best_action['N_added']:<15} kg")
        print(f"{'Phosphorus (P)':<15} | {best_action['P_added']:<15} kg")
        print(f"{'Potassium (K)':<15} | {best_action['K_added']:<15} kg")
        
        print(f"\nTotal Optimization Cost: {cost:.2f}")
        print(f"Goal Achieved: {problem.goalstate(Node(result_state))}")
        
        print("\nFinal State:")
        for key in ['soil_moisture', 'N', 'P', 'K', 'water_used', 'fertilizer_used']:
            target = ""
            if key in ['soil_moisture', 'N', 'P', 'K']:
                opt_range = problem.optimal_ranges[key]
                target = f" (Target range: {opt_range[0]:.2f} - {opt_range[1]:.2f})"
            print(f"  {key}: {result_state[key]:.2f}{target}")

if __name__ == "__main__":
    test_optimization_ga()