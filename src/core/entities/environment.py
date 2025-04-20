import pandas as pd 
from  goal_state import GoalState
import itertools

class Node:
    def __init__(self,state,g=0,f=0):
        self.state = state
        self.g = g
        self.f = f

    def __hash__(self):
        # Convert the dict to a frozenset of items for hashing
        return hash(frozenset(self.state.items()))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state
    
    def copy(self):
            return Node(self.state.copy(), self.g)

class optimization_problem:
    def __init__(self, initial_state,df):
        self.df = df
        self.initial_state = initial_state
        self.transition_model = self._transition_model()
    


    def in_range(self,actual, target, tolerance):
        return abs(actual - target) <= tolerance

    def goalstate(self, node):
        goal = GoalState(
            self.initial_state['label'],
            self.initial_state['growth_stage'],
            self.initial_state['growth_type'],
            self.initial_state['crop_density']
        )
        
        return (
        self.in_range(node.state['soil_moisture'], goal.optimal_soil_moisture, 5) and
        self.ssin_range(node.state['ph'], goal.optimal_ph, 0.3) and
        self.in_range(node.state['N'], goal.optimal_n, 10) and
        self.in_range(node.state['P'], goal.optimal_p, 5) and
        self.in_range(node.state['K'], goal.optimal_k, 5)
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

        # Resource constraints
        water_priority -= s['water_availability'] * 0.1
        if s['water_availability'] < 0.3:
            water_priority += 0.3
            irrigation_frequency_priority += 0.1

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

        if 'WUE' in state and 'WUE' in self.optimal_ranges:
            deviation_score += self.penalty('WUE', state['WUE'])

        return deviation_score + cost_penalty

    def get_valid_actions(self):
        valid_actions = []
        water_options = [0, 10, 20]  # liters/m^2
        N_options = [0, 5, 10, 20]
        P_options = [0, 5, 10, 20]
        K_options = [0, 5, 10, 20]  # kg/m^2

        # Generate all possible combinations
        for water, n, p, k in itertools.product(water_options, N_options, P_options, K_options):
            # Optional constraint check
            if (water <= self._calculate_max_water()) and (n + p + k <= self._calculate_max_fertilizer()):
                action = {
                    'water': water,
                    'N': n,
                    'P': p,
                    'K': k
                }
                valid_actions.append(action)

        return valid_actions

    def _calculate_max_water(self):
        moist = self.initial_state['soil_moisture']
        soil_type = str(self.initial_state['soil_type'])
        cap = {"1": 45, "2": 60, "3": 75}[soil_type]
        per_L = self.transition_model["add_water"]["soil_moisture_increase_per_L"][soil_type]
        return max(0, (cap - moist) / per_L)

    def _calculate_max_fertilizer(self):
        soil_type = str(self.initial_state['soil_type'])
        ranges = self.goalstate()
        limits = []

        for nutrient in ['N', 'P', 'K']:
            current = self.initial_state[nutrient]
            max_val = ranges[nutrient][1]
            gain = self.transition_model["apply_fertilizer"]["npk_availability_increase"][soil_type][nutrient]
            limits.append((max_val - current) / gain if gain > 0 else float('inf'))

        return max(0, min(limits))

    def apply_action(self, node, action):
        water_added, fertilizer_added = action
        soil_type = str(node['soil_type'])

        new_node = node.copy()

        delta_moisture = 0
        if water_added > 0:
            moisture_per_L = self.transition_model["add_water"]["soil_moisture_increase_per_L"][soil_type]
            delta_moisture = water_added * moisture_per_L
            new_node['soil_moisture'] += delta_moisture

            uptake_per_1pct = self.transition_model["add_water"]["npk_uptake_increase_per_1_percent_moisture"][soil_type]
            for nutrient in ['N', 'P', 'K']:
                new_node[nutrient] += delta_moisture * uptake_per_1pct[nutrient]
                new_node[nutrient] = min(new_node[nutrient], 1.0)

        new_node['water_used'] += water_added

        if fertilizer_added > 0:
            npk_gain = self.transition_model["apply_fertilizer"]["npk_availability_increase"][soil_type]
            for nutrient in ['N', 'P', 'K']:
                new_node[nutrient] += fertilizer_added * npk_gain[nutrient]
                new_node[nutrient] = min(new_node[nutrient], 1.0)

        new_node['fertilizer_used'] += fertilizer_added

        return new_node

    def expand_node(self, node):
        children = []
        for action in self.get_actions():
            child_node = self.apply_action(node, action)
            g_cost = self.cost_function(action)
            h = self.heuristic(child_node)
            child_node.g = g_cost + node.g
            child_node.f = g_cost + h
            children.append(child_node)
        return children

    def _transition_model(self):
        return {
            "add_water": {
                "units": "1 L/m²",
                "soil_moisture_increase_per_L": {
                    "1": 1.0,
                    "2": 0.6,
                    "3": 0.3
                },
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

   
    def cost_function(self,action) :
        if (self.initial_state['water_availability'] == "high" and self.initial_state['fertilizer_availability'] == "high") or (self.initial_state['water_availability'] == "medium" and self.initial_state['fertilizer_availability'] == "medium") or ((self.initial_state['water_availability'] == "low" and self.initial_state['fertilizer_availability']== "low"))   :
            """
            if both have same availability levels -> we will take into consideration only that water is less expensive then fertilizer 
            """
            water_cost = action["water_added"]
            fertilizer_cost = (action["N_added"] + action["P_added"] + action["K_added"])*2
        elif self.initial_state['water_availability'] == "medium" and self.initial_state['fertilizer_availability'] == "high" or (self.initial_state['water_availability'] == "low" and self.initial_state['fertilizer_availability'] == "medium") :
            """
            Here since water is available in medium levels, we will consider that it costs the same as using fertilizer
            """
            water_cost = action["water_added"]
            fertilizer_cost = action["N_added"] + action["P_added"] + action["K_added"]
        elif (self.initial_state['water_availability'] == "high" and self.initial_state['fertilizer_availability'] == "medium") or (self.initial_state['water_availability'] == "medium" and self.initial_state['fertilizer_availability'] == "low")  :
            water_cost = action["water_added"]
            fertilizer_cost = (action["N_added"] + action["P_added"] + action["K_added"])*3
        elif (self.initial_state['water_availability'] == "low" and self.initial_state['fertilizer_availability'] == "high") :
            water_cost = action["water_added"]*2
            fertilizer_cost = (action["N_added"] + action["P_added"] + action["K_added"])
        elif self.initial_state['water_availability'] == "high" and self.initial_state['fertilizer_availability'] == "low" :
            water_cost = action["water_added"]
            fertilizer_cost = (action["N_added"] + action["P_added"] + action["K_added"])*4

        return water_cost + fertilizer_cost


def main():
    # === Initial State ===
    print("Initializing the initial state with soil moisture, nutrients, and other factors...\n")
    
    # Instantiate the optimization problem with the initial state
    df = pd.read_csv("FS25.csv")
    farm_problem = optimization_problem(initial_state,df)
    
    # Creating a minimal Node class to represent a state and its associated costs
    print("Defining a simple Node class to represent a state and its associated costs...\n")


    # === 1. Create initial node ===
    print("1. Testing initial state:")
    root = Node(initial_state)  # Creating the root node with the initial state
    print(f"Initial state: {root.state}\n")

    # === 2. Test heuristic ===
    print("2. Heuristic value of the state:")
    h = farm_problem.heuristic(root.state)  # Calculate heuristic for the initial state
    print(f"Heuristic h(s): {h}\n")

    # === .. Test cost ===
    print("3. Cost value of the state:")
    g = farm_problem.cost(root.state)  # Calculate cost for the initial state
    print(f"Cost g(s): {g}\n")

    # === 4. Testing if the action is valid ===
    print("4. Testing if the action is valid:")
    
    # Assuming you have a method to get valid actions.
    valid_actions = farm_problem.get_valid_actions(root.state)

    # If no valid actions exist, print an appropriate message.
    if valid_actions:
        print("Valid actions are:")
        for action in valid_actions:
            print(f"Action: {action}")
    else:
        print("No valid actions available.\n")

    # === 5. Test apply_action ===
    print("5. Applying one action:")
    test_action = (0.5, 0.2)  # water amount = 0.5, fertilizer amount = 0.2
    new_state = farm_problem.apply_action(root.state.copy(), test_action)  # Apply the action to the state
    print(f"New state after action {test_action}: {new_state}\n")

    # === 6. Test expand_node ===
    print("6. Expanding node:")
    children = farm_problem.expand_node(root)  # Expand the node to generate child nodes
    for idx, child in enumerate(children):
        print(f"Child {idx+1}:")
        print(f"State: {child.state}")
        print(f"g: {child.g}, f: {child.f}")
        print()

    # === 7. Test goal ===
    print("7. Testing if it is a goal state:")

    # Perform goal test
    goal = farm_problem.goal_test(root.state)

    # Print whether the current state is a goal state
    if goal:
        print(f"The current state {root.state} is a goal state.")
    else:
        print(f"The current state {root.state} is NOT a goal state.\n")


# Check if this is the main program being executed
if __name__ == "main":
    main()


initial_state = {
        'soil_moisture': 0.25,  # Initial soil moisture value (arbitrary value between 0 and 1)
        'N': 0.15,  # Initial nitrogen content
        'P': 0.10,  # Initial phosphorus content
        'K': 0.12,  # Initial potassium content
        'label': "rice",
        'soil_type': 1,  # Soil type, representing specific characteristics of soil
        'temperature': 28,  # Current temperature in degrees Celsius
        'crop_density': 14,
        'humidity': 45,  # Current humidity percentage
        'rainfall': 6,  # Expected rainfall forecast (in mm)
        'growth_stage': 1,  # Growth stage of the crop (vegetative stage)
        'water_availability': 0.5,  # Water availability for irrigation (on a scale from 0 to 1)
        'irrigation_system': 'drip',  # Type of irrigation system being used  
        'fertilizer_used': 0.0  # Amount of fertilizer used (initialized to zero)
    }