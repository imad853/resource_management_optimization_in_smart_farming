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

    #Mutate func uses the same values in get_valid_action()
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

    def pmx_crossover_continuous(parent1, parent2):
        """
        Modified PMX for continuous values that preserves segment relationships.
        
        Parameters:
            parent1 (list): Resource values from first parent
            parent2 (list): Resource values from second parent
            
        Returns:
            list: Offspring resource values
        """
        size = len(parent1)
        if size != len(parent2):
            raise ValueError("Parents must have same number of resources")
        
        # Initialize offspring with None values
        offspring = [None] * size
        
        # 1. Select random crossover segment
        pt1 = random.randint(0, size - 2)
        pt2 = random.randint(pt1 + 1, size - 1)
        
        # 2. Copy segment from parent1 to offspring
        for i in range(pt1, pt2 + 1):
            offspring[i] = parent1[i]
        
        # 3. Create value mapping between parents
        value_map = {}
        for i in range(pt1, pt2 + 1):
            p1_val = parent1[i]
            p2_val = parent2[i]
            
            # For continuous values, create proportional mapping
            if p1_val != p2_val:
                ratio = p2_val / p1_val if p1_val != 0 else 1
                value_map[p1_val] = p2_val
                value_map[p2_val] = p1_val
        
        # 4. Fill remaining positions from parent2 using mapping
        for i in range(size):
            if offspring[i] is None:  # Only fill empty positions
                val = parent2[i]
                
                # If value exists in parent1's segment, apply mapping
                while val in parent1[pt1:pt2+1] and val in value_map:
                    val = value_map[val]
                
                offspring[i] = val
        
        return offspring
    
    

    def resource_crossover(parent1, parent2):
        """
        Performs Partially Mapped Crossover (PMX) between two parent resource allocation plans.
        Combines segments from both parents while maintaining valid resource allocations.
        
        Parameters:
            parent1 (dict): First parent's resource allocation plan
            parent2 (dict): Second parent's resource allocation plan
            
        Returns:
            dict: New offspring combining characteristics from both parents
        """
        # Resources we want to optimize (must be in both parents)
        resource_keys = ['water_used', 'N_added', 'P_added', 'K_added']
        
        # Verify parents have the required structure
        if not all(key in parent1 and key in parent2 for key in resource_keys):
            raise ValueError("Parents must contain all required resource keys")
        
        # Create offspring with non-resource attributes from parent1
        offspring = {k: parent1[k] for k in parent1 if k not in resource_keys}
        
        # Convert resource values to lists for PMX processing
        parent1_res = [parent1[k] for k in resource_keys]
        parent2_res = [parent2[k] for k in resource_keys]
        
        # Perform PMX crossover on resources
        offspring_res = pmx_crossover_continuous(parent1_res, parent2_res)
        
        # Add crossed-over resources to offspring
        for i, key in enumerate(resource_keys):
            offspring[key] = offspring_res[i]
        
        return offspring
    

    

def test_optimization_ga():
    print("\n===== Smart Farming Resource Optimization GA =====")
    
    df = pd.read_csv("FS25.csv")
    problem = optimization_problem(initial_state, df)

    # Only testing selection methods now
    for selection_method in ['tournament', 'roulette']:
        print(f"\n=== Testing with {selection_method.title()} Selection ===")
        ga = GeneticAlgorithm(
            problem,
            population_size=20,
            generations=200,
            mutation_rate=0.2,
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

if __name__ == "__main__":
    test_optimization_ga()
