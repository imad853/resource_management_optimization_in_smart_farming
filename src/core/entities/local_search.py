
import random
import pandas as pd
import random
class GoalState:
    def __init__(self):
        self.df = None
        self.optimal_soil_moisture = None
        self.optimal_ph = None
        self.optimal_n = None
        self.optimal_p = None
        self.optimal_k = None

    def estimate_optimal_params(self, label, growth_stage, soil_type, crop_density_input, df,density_tolerance=2):
        self.df = df
        growth_stage = int(growth_stage)
        soil_type = int(soil_type)

        filtered = df[
            (df['label'].str.lower() == label.lower()) &
            (df['growth_stage'] == growth_stage) &
            (df['soil_type'] == soil_type) &
            (np.abs(df['crop_density'] - crop_density_input) <= density_tolerance)
        ]

        if filtered.empty:
            print(" No matching data found for the given conditions.")
            return None

        filtered2 = df[
            (df['label'].str.lower() == label.lower()) &
            (df['growth_stage'] == growth_stage) &
            (df['soil_type'] == soil_type)
        ]

        env_features = ['rainfall', 'humidity', 'temperature', 'sunlight_exposure']
        env_avgs = filtered2[env_features].mean()
        print("\n Environmental Averages from Dataset:")
        for feature in env_features:
            print(f"  • {feature.title()}: {env_avgs[feature]:.2f}")

        original_soil_moisture_avg = filtered['soil_moisture'].mean()
        original_soil_moisture_std = filtered['soil_moisture'].std()
        self.optimal_ph = filtered['ph'].mean()
        self.optimal_n = filtered['N'].mean()
        self.optimal_p = filtered['P'].mean()
        self.optimal_k = filtered['K'].mean()

        print(f"\n Average Original Soil Moisture: {original_soil_moisture_avg:.2f}%")
        print(f"\n Standard Deviation of Original Soil Moisture: {original_soil_moisture_std:.2f}%")
        print(" Relevant Row Data:")
        print(filtered[['soil_moisture', 'rainfall', 'humidity', 'temperature', 'sunlight_exposure', 'crop_density', 'water_usage_efficiency', 'N']])

        def adjust_soil_moisture(row):
            adj = row['soil_moisture']
            if original_soil_moisture_std < 3:
                return adj

            rain_diff = row['rainfall'] - env_avgs['rainfall']
            humidity_diff = row['humidity'] - env_avgs['humidity']
            temp_diff = row['temperature'] - env_avgs['temperature']
            sun_diff = row['sunlight_exposure'] - env_avgs['sunlight_exposure']

            if soil_type == 1:
                if growth_stage == 1:
                    adj -= 0.02 * rain_diff
                    adj -= 0.6 * humidity_diff
                    adj += 0.8 * temp_diff
                    adj += 1.2 * sun_diff
                elif growth_stage == 2:
                    adj -= 0.02 * rain_diff
                    adj -= 0.7 * humidity_diff
                    adj += 0.75 * temp_diff
                    adj += 1.3 * sun_diff
                else:
                    adj -= 0.02 * rain_diff
                    adj -= 0.8 * humidity_diff
                    adj += 1.4 * temp_diff
                    adj += 1.1 * sun_diff

            elif soil_type == 2:
                if growth_stage == 1:
                    adj -= 0.03 * rain_diff
                    adj -= 0.06 * humidity_diff
                    adj += 0.15 * temp_diff
                    adj += 0.25 * sun_diff
                elif growth_stage == 2:
                    adj -= 0.035 * rain_diff
                    adj -= 0.07 * humidity_diff
                    adj += 0.28 * temp_diff
                    adj += 0.28 * sun_diff
                else:
                    adj -= 0.04 * rain_diff
                    adj -= 0.08 * humidity_diff
                    adj += 0.3 * temp_diff
                    adj += 0.3 * sun_diff

            else:
                if growth_stage == 1:
                    adj -= 0.02 * rain_diff
                    adj -= 0.6 * humidity_diff
                    adj += 0.1 * temp_diff
                    adj += 1.2 * sun_diff
                elif growth_stage == 2:
                    adj -= 0.02 * rain_diff
                    adj -= 0.7 * humidity_diff
                    adj += 0.4 * temp_diff
                    adj += 1.3 * sun_diff
                else:
                    adj -= 0.02 * rain_diff
                    adj -= 0.8 * humidity_diff
                    adj += 0.4 * temp_diff
                    adj += 1.4 * sun_diff

            return adj

        filtered = filtered.copy()
        filtered['Adjusted Soil Moisture'] = filtered.apply(adjust_soil_moisture, axis=1)

        print("\n Adjusted Soil Moisture Values:")
        print(filtered['Adjusted Soil Moisture'].values)

        self.optimal_soil_moisture = np.average(
            filtered['Adjusted Soil Moisture'],
            weights=1 / filtered['water_usage_efficiency']
        )

        print(f"\n Weighted Optimal Soil Moisture (favoring low WUE): {self.optimal_soil_moisture:.2f}%")
        print(f" Std Dev (Adjusted): {filtered['Adjusted Soil Moisture'].std():.2f}%")
        print(f" Estimated Optimal Soil Moisture (normal average): {filtered['Adjusted Soil Moisture'].mean():.2f}%")
        print(f" Estimated Optimal ph  : {self.optimal_ph:.2f}")
        print(f" Estimated Optimal N   : {self.optimal_n:.2f}")
        print(f" Estimated Optimal P   : {self.optimal_p:.2f}")
        print(f" Estimated Optimal K   : {self.optimal_k:.2f}")

        return self.optimal_soil_moisture


import pandas as pd
import numpy as np
import itertools 
import heapq

class Node:
    def __init__(self, state, g=0, f=0):
        self.state = state
        self.g = g
        self.f = f
        self.parent = None  # Important for path reconstruction

    def __hash__(self):
        # Convert state dict to a hashable representation
        # Only consider the key state variables that affect goal checking
        key_vars = ['soil_moisture', 'ph', 'N', 'P', 'K']
        return hash(tuple((k, round(self.state[k], 2)) for k in key_vars if k in self.state))

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
            
        # Compare only key state variables
        key_vars = ['soil_moisture', 'ph', 'N', 'P', 'K']
        for k in key_vars:
            if k in self.state and k in other.state:
                # Round to handle floating point comparisons
                if round(self.state[k], 2) != round(other.state[k], 2):
                    return False
        return True

    # Add these comparison methods
    def __lt__(self, other):
        # Compare nodes based on f-score (total cost)
        return self.f < other.f

    def __gt__(self, other):
        return self.f > other.f

    def copy(self):
        new_node = Node(self.state.copy(), self.g, self.f)
        new_node.parent = self.parent
        return new_node

class optimization_problem:
    def __init__(self, initial_state, df):
        self.df = df
        self.initial_state = initial_state
        self.transition_model = self._transition_model()
        
        goal = GoalState()
        
        goal.estimate_optimal_params(
            self.initial_state['label'],
            self.initial_state['growth_stage'],
            self.initial_state['soil_type'],
            self.initial_state['crop_density'], 
            self.df, 
        )
        self.goal = goal
        self.optimal_ranges = {
            'soil_moisture': (self.goal.optimal_soil_moisture - 2.5, self.goal.optimal_soil_moisture + 2.5),
            'ph': (self.goal.optimal_ph - 1, self.goal.optimal_ph + 1),
            'N': (self.goal.optimal_n - 2.5, self.goal.optimal_n + 2.5),
            'P': (self.goal.optimal_p - 2.5, self.goal.optimal_p + 2.5),
            'K': (self.goal.optimal_k - 2.5, self.goal.optimal_k + 2.5),
        }
    
    def _generate_random_state(self):
        """
        Generates a random state dictionary with the following parameters:
        - Water_applied: ranges from 0 to 100 (represents water volume/amount)
        - N_applied: ranges from 0 to 100 (represents nitrogen application)
        - p_applied: ranges from 0 to 100 (represents phosphorus application)
        - irrigation_frequency: ranges from 0 to 4 (represents irrigation frequency)
        
        Returns:
            dict: A dictionary with random values for the specified parameters
        """
        state = {
            'water_added': random.uniform(-10, 100),
            'N_added': random.uniform(0, 100),
            'K_added': random.uniform(0,100),
            'P_added': random.uniform(0, 100)
        }
    
        return state

    def in_range(self, actual, target, tolerance):
        return abs(actual - target) <= tolerance

    def goalstate(self, node):
        # Check if the state matches our goal conditions
        moisture_in_range = self.in_range(node.state['soil_moisture'], self.goal.optimal_soil_moisture, 2.5)
        ph_in_range = self.in_range(node.state['ph'], self.goal.optimal_ph, 1)
        n_in_range = self.in_range(node.state['N'], self.goal.optimal_n, 2.5)
        p_in_range = self.in_range(node.state['P'], self.goal.optimal_p, 2.5)
        k_in_range = self.in_range(node.state['K'], self.goal.optimal_k, 2.5)
        
        # Debug output to see what's happening
        print("\nChecking if goal state:")
        print(f"  Soil Moisture: {node.state['soil_moisture']:.2f} -> Target: {self.goal.optimal_soil_moisture:.2f} ± 2.5 -> In range: {moisture_in_range}")
        print(f"  pH: {node.state['ph']:.2f} -> Target: {self.goal.optimal_ph:.2f} ± 1.0 -> In range: {ph_in_range}")
        print(f"  N: {node.state['N']:.2f} -> Target: {self.goal.optimal_n:.2f} ± 2.5 -> In range: {n_in_range}")
        print(f"  P: {node.state['P']:.2f} -> Target: {self.goal.optimal_p:.2f} ± 2.5 -> In range: {p_in_range}")
        print(f"  K: {node.state['K']:.2f} -> Target: {self.goal.optimal_k:.2f} ± 2.5 -> In range: {k_in_range}")
        
        is_goal = moisture_in_range and ph_in_range and n_in_range and p_in_range and k_in_range
        print(f"  Is goal state: {is_goal}")
        
        return is_goal

    def priorities(self):
        s = self.initial_state  # shorthand

        water_priority = 0.33
        fertilizer_priority = 0.33
        irrigation_frequency_priority = 0.33

        # Growth stage
        if s['growth_stage'] == 1:
            water_priority += 0.1
            fertilizer_priority += 0.05
            irrigation_frequency_priority += 0.1
        elif s['growth_stage'] == 2:
            fertilizer_priority += 0.15
            water_priority += 0.05
        elif s['growth_stage'] == 3:
            water_priority += 0.15
            fertilizer_priority += 0.1
            irrigation_frequency_priority += 0.05

        # Soil type
        soil_type = str(s['soil_type'])
        if soil_type == "1":
            irrigation_frequency_priority += 0.15
            water_priority += 0.1
        elif soil_type == "3":
            irrigation_frequency_priority -= 0.1
            water_priority -= 0.05
            fertilizer_priority += 0.05

        # Temperature effects
        heat_stress = max(0, min(1, (s['temperature'] - 25) / 15))
        water_priority += heat_stress * 0.2
        irrigation_frequency_priority += heat_stress * 0.15

        # Humidity effects
        water_priority -= (s['humidity'] / 100) * 0.1

        # Soil moisture effects
        drought_factor = max(0, 1 - (s['soil_moisture'] / 0.2))
        water_priority += drought_factor * 0.25
        irrigation_frequency_priority += drought_factor * 0.2

        # Ensure priorities don't go below minimum values
        return {
            'water_priority': max(0.33, min(1.0, water_priority)),
            'fertilizer_priority': max(0.33, min(1.0, fertilizer_priority)),
            'irrigation_frequency_priority': max(0.33, min(1.0, irrigation_frequency_priority))
        }

    def optimal_distance_calc(self, var, value):
        min_val, max_val = self.optimal_ranges[var]
        if value < min_val:
            return min_val - value
        elif value > max_val:
            return value - max_val
        return 0

    def heuristic(self, state):
        p = self.priorities()
        deviation_score = 0
        
        # Only consider parameters that are outside their optimal ranges
        if not (self.optimal_ranges['soil_moisture'][0] <= state['soil_moisture'] <= self.optimal_ranges['soil_moisture'][1]):
            deviation_score += p['water_priority'] * self.optimal_distance_calc('soil_moisture', state['soil_moisture'])
        
        nutrient_priority = p['fertilizer_priority']
        for nutrient in ['N', 'P', 'K']:
            if not (self.optimal_ranges[nutrient][0] <= state[nutrient] <= self.optimal_ranges[nutrient][1]):
                deviation_score += nutrient_priority * self.optimal_distance_calc(nutrient, state[nutrient])
        
        # Only consider irrigation if soil moisture needs adjustment
        if not (self.optimal_ranges['soil_moisture'][0] <= state['soil_moisture'] <= self.optimal_ranges['soil_moisture'][1]):
            deviation_score += p['irrigation_frequency_priority']
        
        return deviation_score

    def get_valid_actions(self, state):
        valid_actions = []
        
        # Determine which parameters need adjustment
        needs_adjustment = {
            'soil_moisture': not (self.optimal_ranges['soil_moisture'][0] <= state['soil_moisture'] <= self.optimal_ranges['soil_moisture'][1]),
            'N': not (self.optimal_ranges['N'][0] <= state['N'] <= self.optimal_ranges['N'][1]),
            'P': not (self.optimal_ranges['P'][0] <= state['P'] <= self.optimal_ranges['P'][1]),
            'K': not (self.optimal_ranges['K'][0] <= state['K'] <= self.optimal_ranges['K'][1]),
            'irrigation': not (self.optimal_ranges['soil_moisture'][0] <= state['soil_moisture'] <= self.optimal_ranges['soil_moisture'][1])
        }
        
        # Generate base action with no changes
        base_action = {
            "water_added": 0,
            "N_added": 0,
            "P_added": 0,
            "K_added": 0,
            "irrigation_update": 0
        }
        
        # If no parameters need adjustment, return only the base action
        if not any(needs_adjustment.values()):
            return [base_action]
        
        # Generate possible actions only for parameters that need adjustment
        water_options = [-3, -2, 0, 2, 3] if needs_adjustment['soil_moisture'] else [0]
        N_options = [-3, 0, 3, 6] if needs_adjustment['N'] else [0]
        P_options = [-3, 0, 3, 6] if needs_adjustment['P'] else [0]
        K_options = [-3, 0, 3, 6] if needs_adjustment['K'] else [0]
        irrigation_options = [-1, 0, 1, 2] if needs_adjustment['irrigation'] else [0]
        
        for water, n, p, k, irrigation in itertools.product(water_options, N_options, P_options, K_options, irrigation_options):
            action = {
                "water_added": water,
                "N_added": n,
                "P_added": p,
                "K_added": k,
                "irrigation_update": irrigation
            }
            
            # Additional validation to ensure actions make sense
            valid = True
            
            # Soil moisture
            if needs_adjustment['soil_moisture']:
                current = state['soil_moisture']
                target = self.goal.optimal_soil_moisture
                if current < target and water <= 0:
                    valid = False
                if current > target and water >= 0:
                    valid = False
            
            # Nutrients
            for nutrient, val in zip(['N', 'P', 'K'], [n, p, k]):
                if needs_adjustment[nutrient]:
                    current = state[nutrient]
                    target = getattr(self.goal, f'optimal_{nutrient.lower()}')
                    if current < target and val <= 0:
                        valid = False
                    if current > target and val >= 0:
                        valid = False
            
            # Irrigation
            if needs_adjustment['irrigation']:
                current = state['soil_moisture']
                target = self.goal.optimal_soil_moisture
                if current < target and irrigation <= 0:
                    valid = False
                if current > target and irrigation >= 0:
                    valid = False
            
            if valid:
                valid_actions.append(action)
        
        return valid_actions

    def apply_action(self, node, action):
        soil_type = str(node.state['soil_type'])
        water_source = str(node.state['water_source'])
        new_node = node.copy()
        
        # Only apply water if soil moisture needs adjustment
        if not (self.optimal_ranges['soil_moisture'][0] <= node.state['soil_moisture'] <= self.optimal_ranges['soil_moisture'][1]):
            moisture_per_L = self.transition_model["add_water"]["soil_moisture_increase_per_L"][soil_type]
            delta_moisture = action["water_added"] * moisture_per_L
            new_node.state['soil_moisture'] += delta_moisture
            ##### impact on ph level#####
            new_node.state["ph"] += self.transition_model["add_water"]["ph_change_per_L_by_water_source"][water_source][soil_type]
            uptake_per_1pct = self.transition_model["add_water"]["npk_uptake_increase_per_1_percent_moisture"][soil_type]
            for nutrient in ['N', 'P', 'K']:
                new_node.state[nutrient] += delta_moisture * uptake_per_1pct[nutrient]
            new_node.state['water_used'] += action["water_added"]
        
        # Only apply fertilizer if nutrient needs adjustment
        for nutrient in ['N', 'P', 'K']:
            if not (self.optimal_ranges[nutrient][0] <= node.state[nutrient] <= self.optimal_ranges[nutrient][1]):
                new_node.state[nutrient] += action[f"{nutrient}_added"]
                ####impact on ph level of the soil#####
                new_node.state['ph']+= action[f"{nutrient}_added"] * self.transition_model["add_fertilizer"]["ph_change_per_application"][nutrient][soil_type]
                new_node.state['fertilizer_used'] += action[f"{nutrient}_added"] * new_node.state[f"{nutrient}_percentage"]
        
        return new_node

    def expand_node(self, node):
        children = []
        for action in self.get_valid_actions(node.state):
            new_node = self.apply_action(node, action)
            g_cost = self.cost_function(node.state, action)
            h = self.heuristic(new_node.state)
            new_node.g = node.g + g_cost
            new_node.f = new_node.g + h
            new_node.parent = node
            children.append(new_node)
        return children
    
    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node)
            node = node.parent
        return path[::-1]  # Reverse to get start-to-goal order
    

    def cost_function(self, state, action):
        wa = state['water_availability']
        fa = state['fertilizer_availability']
        water = action["water_added"]
        fert = action["N_added"] + action["P_added"] + action["K_added"]

        if (wa == "high" and fa == "high") or (wa == "medium" and fa == "medium") or (wa == "low" and fa == "low"):
            return water + fert * 2
        elif (wa == "medium" and fa == "high") or (wa == "low" and fa == "medium"):
            return water + fert
        elif (wa == "high" and fa == "medium") or (wa == "medium" and fa == "low"):
            return water + fert * 3
        elif wa == "low" and fa == "high":
            return water * 2 + fert
        elif wa == "high" and fa == "low":
            return water + fert * 4
        
        if state.get("irrigation_frequency", 0) >= 4:
            return water + fert + 10
        else:
            return water + fert
        
    
    
    def calculate_moisture_increase(self,soil_type, depth_cm=30):

        soil_types = {
        "Sandy": {"bulk_density": 1.43},  # g/cm³
        "Loamy": {"bulk_density": 1.43},  # g/cm³
        "Clay": {"bulk_density": 1.33}    # g/cm³
        }

        """
        Calculate how much 1L of water will increase soil moisture in 1m² area
        
        Parameters:
        - soil_type: "Sandy", "Loamy", or "Clay"
        - depth_cm: Soil depth in centimeters (default 30cm)
        
        Returns:
        - Moisture increase percentage
        """
        water_volume_cm3 = 1000  # 1L water = 1000 cm³
        area_cm2 = 10000         # 1m² = 10000 cm²
        soil_volume_cm3 = area_cm2 * depth_cm
        bulk_density = soil_types[soil_type]["bulk_density"]
        soil_mass_g = soil_volume_cm3 * bulk_density
        moisture_increase = (water_volume_cm3 / soil_mass_g) * 100
        return round(moisture_increase, 2)
    
    def _transition_model(self):
    # Calculate moisture increases for each soil type (1: Sandy, 2: Loamy, 3: Clay)
        return {
            "add_water": {
                "units": "1 L/m²",
                "soil_moisture_increase_per_L": {
                    "1": self.calculate_moisture_increase("Sandy"),
                    "2": self.calculate_moisture_increase("Loamy"),
                    "3": self.calculate_moisture_increase("Clay")
                },
                "npk_uptake_increase_per_1_percent_moisture": {
                    "1": {"N": 0.2, "P": 0.15, "K": 0.18},
                    "2": {"N": 0.25, "P": 0.2, "K": 0.22},
                    "3": {"N": 0.15, "P": 0.25, "K": 0.2}
                },
                "ph_change_per_L_by_water_source": {
                    
                    "1": {"1": -0.01, "2": -0.015, "3": -0.01},   
                    "2": {"1": 0.01, "2": 0.015, "3": 0.01}, 
                    "3": {"1": -0.02, "2": -0.025, "3": -0.02}   
                }
            },
            "add_fertilizer": {
                "units": "per application",
                "ph_change_per_application": {
                    # Based on general acidifying potential
                    "N": {"1": 0.01, "2": 0.015, "3": 0.01},
                    "P": {"1": -0.01, "2": -0.015, "3": -0.01},
                    "K": {"1": 0.00, "2": 0.00, "3": 0.00},
                }
            }
        }

class GeneticAlgorithm:
    def __init__(self, problem, population_size=50, generations=1000, mutation_rate=0.1, tournament_size=3,
                 selection_method='tournament', crossover_method='pmx', mutation_method='swap'):
        
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method

    def initialize_population(self):
        # Creates an initial population using the problem's _generate_random_state() method.
        population = []
        for _ in range(self.population_size):
            population.append(self.problem._generate_random_state())
        return population

    def evolve_population(self, population):
        """
        Evolves the current population to the next generation.

        Input:
          - population: A list of individuals representing the current population.

        Output:
          - A new list of individuals representing the next generation.

        Hints:
          1. Use elitism: Identify the best individual and always include it in the new population.
          2. While the new population size is less than the desired size:
             a. Select two parents using the selection method.
             b. Generate a child using the crossover method.
             c. With probability equal to mutation_rate, apply mutation to the child.
             d. Add the resulting child to the new population.
        Example:
          population = [[A, B, C, D], [A, C, B, D], ...]
          A valid new_population might preserve the best tour and include several mutated/crossed-over offspring.
        """
        # Find the best individual (elitism)
        best_individual = min(population, key=lambda x: self.evaluate(x))
        
        # Create new population with the best individual
        new_population = [best_individual.copy()]
        
        # Fill the rest of the population with offspring
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.select_parent(population)
            parent2 = self.select_parent(population)
            
            # Create offspring through crossover
            child = self.perform_crossover(parent1, parent2)
            
            # Apply mutation with probability mutation_rate
            if random.random() < self.mutation_rate:
                child = self.perform_mutation(child)
                
            # Add child to new population
            new_population.append(child)
            
        return new_population

    def select_parent(self, population):
        # Selects a parent using the specified selection method.
        if self.selection_method == 'tournament':
            return self.tournament_selection(population, self.tournament_size)
        elif self.selection_method == 'roulette':
            return self.roulette_wheel_selection(population)
        else:
            # Default to tournament selection if unknown.
            return self.tournament_selection(population, self.problem, self.tournament_size)

    def perform_crossover(self, parent1, parent2):
        # Performs crossover between two parents using the specified method.
        if self.crossover_method == 'order':
            return self.order_crossover(parent1, parent2)
    
    def evaluate(self,individual):
        new_node = Node(self.problem.initial_state)
        potential_goal = self.problem.apply_action(new_node,individual)
        return self.problem.heuristic(potential_goal.state) 
    
    def perform_mutation(self, individual):
        # Applies mutation to an individual using the specified mutation method.
        return self.mutate(individual)
    def tournament_selection(self,population, tournament_size):
        """Selects an individual using tournament selection."""
        # Randomly select tournament_size individuals
        tournament = random.sample(population, tournament_size)
        
        # Return the best individual from the tournament
        return min(tournament, key=lambda x: self.evaluate(x)).copy()
    def roulette_wheel_selection(self,population):
        """Selects an individual using roulette wheel selection."""
        # Calculate fitness values (lower cost = higher fitness)
        fitness_values = [1.0 / (self.evaluate(individual) + 1e-10) for individual in population]
        total_fitness = sum(fitness_values)
        
        # Normalize fitness values to create probability distribution
        probabilities = [fitness / total_fitness for fitness in fitness_values]
        
        # Select an individual based on probabilities
        selected_index = random.choices(range(len(population)), weights=probabilities, k=1)[0]
        return population[selected_index].copy()
    
    def order_crossover(self, parent1, parent2):
        """
        Order Crossover (OX) adapted for dictionary-based representations.
        Each parent is a dictionary with keys representing parameters.
        """
        # Create a new offspring dictionary
        offspring = {}
        
        # Get all keys from the parent dictionary
        keys = list(parent1.keys())
        
        # Choose a random subset of keys to inherit from parent1
        num_keys = len(keys)
        num_to_inherit = random.randint(1, num_keys - 1)
        keys_from_parent1 = random.sample(keys, num_to_inherit)
        
        # Copy values for selected keys from parent1
        for key in keys_from_parent1:
            offspring[key] = parent1[key]
        
        # Copy remaining values from parent2
        for key in keys:
            if key not in offspring:
                offspring[key] = parent2[key]
        
        return offspring
    def mutate( self,individual):
        """Mutate one parameter of the action (values now range from 0 to 100)"""
        mutated = individual.copy()
        param = random.choice(["water_added", "N_added", "P_added", "K_added"])
        
        # Possible mutation steps (can be adjusted)
        mutation_steps = [-20, -10, -5, 5, 10, 20]
        
        current_val = mutated[param]
        
        # Apply mutation while clamping between 0 and 100
        step = random.choice(mutation_steps)
        new_val = max(0, min(100, current_val + step))  # Clamp to [0, 100]
        
        # Ensure the value actually changes (avoid no-op mutations)
        while new_val == current_val:
            step = random.choice(mutation_steps)
            new_val = max(0, min(100, current_val + step))
        
        mutated[param] = new_val
        return mutated
    def solve(self):
        """
        Executes the Genetic Algorithm to solve the problem.

        Output:
          - A tuple (best_solution, best_cost) where:
              best_solution: The best individual found.
              best_cost: The evaluation score (cost/conflicts) associated with the best solution.

        Hints:
          1. Initialize the population.
          2. For each generation:
             a. Evolve the population using the evolve_population() function.
             b. Evaluate the current population to track the best solution.
             c. Optionally, print progress every N generations.
          3. Return the best solution and its evaluation once all generations are complete.

        Example:
          For TSP, best_solution might be a permutation of cities, and best_cost the distance of the tour.
          For Eight Queens, best_solution might be a board configuration and best_cost the number of conflicts.
        """
        # Initialize population
        population = self.initialize_population()
        print("\ninitial_population")
        for i in range(len(population)):
            print(population[i])
        
        # Track the best solution found so far
        best_solution = None
        best_cost = float('inf')
        
        # Evolution loop
        for generation in range(self.generations):
            # Evolve population
            population = self.evolve_population(population)
            
            # Find current best solution
            current_best = min(population, key=lambda x: self.evaluate(x))
            current_cost = self.evaluate(current_best)
            
            # Update overall best if necessary
            if current_cost < best_cost:
                best_solution = current_best.copy()
                best_cost = current_cost
            
            # Print progress every 100 generations
            if generation % 100 == 0:
                print(f"Generation {generation}: Best cost = {best_cost}")
        
        # Return the best solution found
        return best_solution, best_cost

# Helper functions for selection, crossover, and mutation

import random


def test_opti_ga():

    # Runs GA on a sample TSP instance with various configurations.
    print("\n===== TSP Genetic Algorithm Extensive Testing =====")
    initial_state = {
    'soil_moisture': 5.28,
    'N': 24.33,
    'P': 24.83,
    'K': 20.33,
    'N_percentage':0.3,
    'K_percentage':0.3,
    'P_percentage':0.3,
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
    'fertilizer_used': 0.0,
    'water_source':1
}
    df = pd.read_csv("src\core\entities\FS25.csv")
    problem = optimization_problem(initial_state, df)

    selection_methods = ['tournament', 'roulette']
    crossover_methods = [ 'order']
    mutation_methods = ['swap']

    for sel in selection_methods:
        for cr in crossover_methods:
            for mut in mutation_methods:
                print(f"\n--- OPTIMIZATION Test: Selection={sel}, Crossover={cr}, Mutation={mut} ---")
                ga = GeneticAlgorithm(problem, population_size=100, generations=100, mutation_rate=0.1,
                                        tournament_size=3, selection_method=sel,
                                        crossover_method=cr, mutation_method=mut)
                best_tour, best_cost = ga.solve()
                print(f"Result:  combination = {best_tour}")
if __name__ == "__main__":
    random.seed()
    test_opti_ga()