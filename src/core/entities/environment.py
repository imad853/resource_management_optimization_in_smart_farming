import pandas as pd
import itertools 
from itertools import count
from goal_state import GoalState
import heapq
import random

class Node:
    def __init__(self, state, g=0, f=0):
        self.state = state
        self.g = g
        self.f = f
        self.parent = None  # Important for path reconstruction

    def __hash__(self):
        return hash(frozenset(self.state.items()))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

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
            self.initial_state['crop_density'] , 
            self.df , 
        )
        self.goal = goal
        self.optimal_ranges = {
            'soil_moisture': (self.goal.optimal_soil_moisture -2.5, self.goal.optimal_soil_moisture + 2.5),
            'ph': (self.goal.optimal_ph - 1, self.goal.optimal_ph + 1),
            'N': (self.goal.optimal_n - 2.5, self.goal.optimal_n + 2.5),
            'P': (self.goal.optimal_p - 2.5, self.goal.optimal_p + 2.5),
            'K': (self.goal.optimal_k - 2.5, self.goal.optimal_k + 2.5),
        }

    def in_range(self, actual, target, tolerance):
        return abs(actual - target) <= tolerance

    def goalstate(self, node):
        return (
            self.in_range(node.state['soil_moisture'], self.goal.optimal_soil_moisture, 2.5) and
            self.in_range(node.state['ph'], self.goal.optimal_ph, 1) and
            self.in_range(node.state['N'], self.goal.optimal_n, 15) and
            self.in_range(node.state['P'], self.goal.optimal_p, 10) and
            self.in_range(node.state['K'], self.goal.optimal_k, 10)
        )

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
        else:
            water_priority -= 0.05
            fertilizer_priority -= 0.1

        # Environmental
        heat_stress = max(0, min(1, (s['temperature'] - 25) / 15))
        water_priority += heat_stress * 0.2
        irrigation_frequency_priority += heat_stress * 0.15

        water_priority -= (s['humidity'] / 100) * 0.1

        rainfall_factor = min(1, s['rainfall_forecast'] / 25)
        water_priority -= rainfall_factor * 0.2
        irrigation_frequency_priority -= rainfall_factor * 0.3

        drought_factor = max(0, 1 - (s['soil_moisture'] / 0.7))
        water_priority += drought_factor * 0.25
        irrigation_frequency_priority += drought_factor * 0.2

        # Soil type
        soil_type = str(s['soil_type'])
        if soil_type == "1":
            irrigation_frequency_priority += 0.15
            water_priority += 0.1
        elif soil_type == "3":
            irrigation_frequency_priority -= 0.1
            water_priority -= 0.05
            fertilizer_priority += 0.05

        # Irrigation system
        if s['irrigation_system'] == "drip":
            irrigation_frequency_priority += 0.1
            water_priority -= 0.15
        elif s['irrigation_system'] == "flood":
            irrigation_frequency_priority -= 0.2
            water_priority += 0.15

        return {
            'water_priority': max(0.1, water_priority),
            'fertilizer_priority': max(0.1, fertilizer_priority),
            'irrigation_frequency_priority': max(0.1, irrigation_frequency_priority)
        }

    def penalty(self, var, value):
        min_val, max_val = self.optimal_ranges[var]
        if value < min_val:
            return (min_val - value) ** 2
        elif value > max_val:
            return (value - max_val) ** 2
        return 0

    def heuristic(self, state):
        deviation_score = (
            self.penalty('soil_moisture', state['soil_moisture']) +
            self.penalty('N', state['N']) +
            self.penalty('P', state['P']) +
            self.penalty('K', state['K'])
        )
        p = self.priorities()
        cost_penalty = (
            p['water_priority'] * state.get('water_used', 0) +
            p['fertilizer_priority'] * state.get('fertilizer_used', 0)
        )
        return deviation_score + cost_penalty

    def get_valid_actions(self, action=None):
        valid_actions = []
        water_options = [0, 10, 20]
        N_options = [0, 5, 10, 20]
        P_options = [0, 5, 10, 20]
        K_options = [0, 5, 10, 20]
        for water, n, p, k in itertools.product(water_options, N_options, P_options, K_options):
            if (water <= self._calculate_max_water()) and (n + p + k <= self._calculate_max_fertilizer()):
                valid_actions.append({
                    "water_added": water,
                    "N_added": n,
                    "P_added": p,
                    "K_added": k
                })
        return valid_actions

    def _calculate_max_water(self):
        moist = self.initial_state['soil_moisture']
        soil_type = str(self.initial_state['soil_type'])
        cap = {"1": 45, "2": 60, "3": 75}[soil_type]
        per_L = self.transition_model["add_water"]["soil_moisture_increase_per_L"][soil_type]
        return max(0, (cap - moist) / per_L)

    def _calculate_max_fertilizer(self):
        soil_type = str(self.initial_state['soil_type'])
        ranges = self.optimal_ranges
        limits = []
        for nutrient in ['N', 'P', 'K']:
            current = self.initial_state[nutrient]
            max_val = ranges[nutrient][1]
            gain = self.transition_model["apply_fertilizer"]["npk_availability_increase"][soil_type][nutrient]
            limits.append((max_val - current) / gain if gain > 0 else float('inf'))
        return max(0, min(limits))

    def apply_action(self, node, action):
        soil_type = str(node['soil_type'])
        new_node = node.copy()
        delta_moisture = 0
    
        # Apply water action
        if action["water_added"] > 0:
            moisture_per_L = self.transition_model["add_water"]["soil_moisture_increase_per_L"][soil_type]
            delta_moisture = action["water_added"] * moisture_per_L
            new_node['soil_moisture'] += delta_moisture
            uptake_per_1pct = self.transition_model["add_water"]["npk_uptake_increase_per_1_percent_moisture"][soil_type]
            for nutrient in ['N', 'P', 'K']:
                new_node[nutrient] += delta_moisture * uptake_per_1pct[nutrient]
                new_node[nutrient] = max(new_node[nutrient], 1.0)
    
        new_node['water_used'] += action["water_added"]
        total_fertilizer = action["N_added"] + action["P_added"] + action["K_added"]
    
        # Apply fertilizer action
        if total_fertilizer > 0:
            npk_gain = self.transition_model["apply_fertilizer"]["npk_availability_increase"][soil_type]
            for nutrient in ['N', 'P', 'K']:
                new_node[nutrient] += action[nutrient + "_added"] * npk_gain[nutrient]
                new_node[nutrient] = max(new_node[nutrient], 1.0)
    
        new_node['fertilizer_used'] += total_fertilizer
    
         # Print current NPK values after applying action
        print("\nAfter applying action:")
        print(f"  Water added: {action['water_added']}L")
        print(f"  N added: {action['N_added']}kg, P added: {action['P_added']}kg, K added: {action['K_added']}kg")
        print("Current nutrient levels:")
        print(f"  N: {new_node['N']:.3f} (Target: {self.goal.optimal_n:.1f} ± {self.optimal_ranges['N'][1]-self.goal.optimal_n:.1f})")
        print(f"  P: {new_node['P']:.3f} (Target: {self.goal.optimal_p:.1f} ± {self.optimal_ranges['P'][1]-self.goal.optimal_p:.1f})")
        print(f"  K: {new_node['K']:.3f} (Target: {self.goal.optimal_k:.1f} ± {self.optimal_ranges['K'][1]-self.goal.optimal_k:.1f})")
    
        return new_node

    def expand_node(self, node):
        children = []
        for action in self.get_valid_actions():
            new_state = self.apply_action(node.state.copy(), action)
            g_cost = self.cost_function(action)
            h = self.heuristic(new_state)
            child_node = Node(new_state, node.g + g_cost, node.g + g_cost + h)
            child_node.parent = node
            children.append(child_node)
        return children
    
    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node)
            node = getattr(node, 'parent', None)  # Safe access to parent
        return path[::-1]  # Reverse to get start-to-goal order

    def solve(self):
        print("\n=== Starting A* Search ===")
        print(f"Initial state: {self.initial_state}")
        print(f"Goal ranges: {self.optimal_ranges}")
        start_node = Node(self.initial_state, g=0, f=self.heuristic(self.initial_state))
        open_list = []
        heapq.heappush(open_list, (start_node.f, start_node))  # Store (priority, node)
        closed_set = set()
        steps = 0

        while open_list:
            current_f, current_node = heapq.heappop(open_list)  # Get both priority and node
            steps += 1

            if self.goalstate(current_node):
                print("Solution found after:")
                return self.reconstruct_path(current_node)

            if current_node in closed_set:
                continue
            closed_set.add(current_node)

            for child in self.expand_node(current_node):
                if child not in closed_set:
                    heapq.heappush(open_list, (child.f, child))  # Push (priority, node)

        print(f"⚠️ Stopped (no solution found).")
        return None

    def _transition_model(self):
        return {
            "add_water": {
                "units": "1 L/m²",
                "soil_moisture_increase_per_L": {"1": 1.0, "2": 0.6, "3": 0.3},
                "npk_uptake_increase_per_1_percent_moisture": {
                    "1": {"N": 0.02, "P": 0.015, "K": 0.018},
                    "2": {"N": 0.025, "P": 0.02, "K": 0.022},
                    "3": {"N": 0.015, "P": 0.025, "K": 0.02}
                }
            },
            "apply_fertilizer": {
                "units": "1 kg/m²",
                "npk_availability_increase": {
                    "1": {"N": 0.15, "P": 0.10, "K": 0.12},
                    "2": {"N": 0.20, "P": 0.18, "K": 0.20},
                    "3": {"N": 0.12, "P": 0.25, "K": 0.18}
                }
            }
        }

    def cost_function(self, action):
        wa = self.initial_state['water_availability']
        fa = self.initial_state['fertilizer_availability']
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
        return water + fert

def uniform_crossover_continuous(parent1, parent2):

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

def two_point_crossover_continuous(parent1, parent2):
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


import random

def resource_crossover(parent1, parent2, method='two_point'):


    resource_keys = ['water_used', 'N_added', 'P_added', 'K_added']
    
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
        offspring_res = two_point_crossover_continuous(parent1_res, parent2_res)
    elif method == 'uniform':
        offspring_res = uniform_crossover_continuous(parent1_res, parent2_res)
    else:
        raise ValueError(f"Unknown crossover method: {method}")
    
    # Add crossed-over resources to offspring
    for i, key in enumerate(resource_keys):
        offspring[key] = offspring_res[i]
    
    return offspring



# === Initial state ===
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

def main():
    import os
    
    # Set working directory to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Debug: Print current directory and files
    print("Current directory:", os.getcwd())
    print("Files here:", os.listdir())
    
    # Now read the CSV
    try:
        df = pd.read_csv("FS25.csv")
        farm_problem = optimization_problem(initial_state, df)        
        initial_node = Node(initial_state)
        solution_path = farm_problem.solve()
        if solution_path:
            print("\n✅ Solution found in", len(solution_path) - 1, "steps:")
            for i, node in enumerate(solution_path):
                print(f"\nStep {i}:")
                for k, v in node.state.items():
                    print(f"  {k}: {round(v, 3) if isinstance(v, float) else v}")
        else:
            print("\n❌ No solution found.")
    
    except FileNotFoundError:
        print("\n❌ Error: 'FS25.csv' not found in:", os.getcwd())
        print("Available files:", os.listdir())
    
    except Exception as e:
        print("\n❌ An error occurred:", str(e))

if __name__ == "__main__":
    main()