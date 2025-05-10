import pandas as pd
import numpy as np

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
        
        for water, n, p, k in itertools.product(water_options, N_options, P_options, K_options):
            action = {
                "water_added": water,
                "N_added": n,
                "P_added": p,
                "K_added": k,
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

            if valid:
                valid_actions.append(action)
        
        return valid_actions
    
    def evaporation_factor(self):
        daily_moisture_loss = 0.3 * self.initial_state['temperature'] + 0.1 * self.initial_state['sunlight_exposure'] + 0.04 * self.initial_state['wind_speed'] + 0.01 *(100-self.initial_state['humidity']) * self.initial_state['soil_type'] * self.initial_state['growth_stage']
        return max(daily_moisture_loss, 0.01)  # prevent zero or negative values
    
    def calculate_drought_time(self, node):
        goal_node =  node.copy()
        moisture_difference = goal_node.state['soil_moisture'] - self.initial_state['soil_moisture']
        soil_reset_duration = moisture_difference / self.evaporation_factor()
        return soil_reset_duration
    
    def get_irrigation_frequency(self, node):
        return   int  (7 / self.calculate_drought_time(node))


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
    
    def solve(self):
        print("\n=== Starting A* Search ===")
        print(f"Initial state: {self.initial_state}")
        print(f"Goal ranges: {self.optimal_ranges}")
        start_node = Node(self.initial_state.copy(), g=0, f=self.heuristic(self.initial_state))
        open_list = []
        heapq.heappush(open_list, (start_node.f, id(start_node), start_node))  # Store (priority, id, node) to avoid comparison errors
        closed_set = set()
        steps = 0
        max_steps = 100000  # Safety limit to prevent infinite loops

        while open_list and steps < max_steps:
            current_f, _, current_node = heapq.heappop(open_list)
            steps += 1

            # Check if this is a goal state BEFORE adding to closed set
            if self.goalstate(current_node):
            # Calculate irrigation frequency
                current_node.state['irrigation_frequency'] = self.get_irrigation_frequency(current_node)

                print(f"✅ Solution found after {steps} steps!")
                return self.reconstruct_path(current_node)

            # Add node hash to closed set
            node_hash = hash(current_node)
            if node_hash in closed_set:
                continue
            closed_set.add(node_hash)

            for child in self.expand_node(current_node):
                child_hash = hash(child)
                if child_hash not in closed_set:
                    heapq.heappush(open_list, (child.f, id(child), child))

        print(f"⚠️ Stopped after {steps} steps (no solution found or max steps reached).")
        return None
    
    def greedy_best_first_search(self):
        """
        Implements the Greedy Best-First Search algorithm.
        This search uses only the heuristic function to guide the search,
        without considering the path cost.
        """
        print("\n=== Starting Greedy Best-First Search ===")
        print(f"Initial state: {self.initial_state}")
        print(f"Goal ranges: {self.optimal_ranges}")
        
        start_node = Node(self.initial_state.copy())
        # Calculate heuristic for start node
        start_node.f = self.heuristic(start_node.state)
        
        # Initialize open list (priority queue) and closed set
        open_list = []
        heapq.heappush(open_list, (start_node.f, id(start_node), start_node))
        closed_set = set()
        
        # Keep track of steps for reporting
        steps = 0
        max_steps = 1000  # Safety limit to prevent infinite loops
        
        while open_list and steps < max_steps:
            # Get node with lowest heuristic value
            _, _, current_node = heapq.heappop(open_list)
            steps += 1
            
            # Check if this is a goal state
            print(f"\nStep {steps}, checking node with heuristic: {current_node.f:.2f}")
            if self.goalstate(current_node):
                current_node.state['irrigation_frequency'] = self.get_irrigation_frequency(current_node)
                print(f"✅ Solution found after {steps} steps!")
                return self.reconstruct_path(current_node)
            
            # Add current node to closed set if not already visited
            node_hash = hash(current_node)
            if node_hash in closed_set:
                continue
            closed_set.add(node_hash)
            
            # Generate child nodes
            for child in self.expand_node(current_node):
                child_hash = hash(child)
                if child_hash not in closed_set:
                    # In Greedy Best-First Search, we only use the heuristic
                    child.f = self.heuristic(child.state)
                    # Add to open list
                    heapq.heappush(open_list, (child.f, id(child), child))
        
        print(f"⚠️ Stopped after {steps} steps (no solution found or max steps reached).")
        return None
    

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



def main(soil_moisture,N,P,K,N_percent,P_percent,K_percent,ph,label,soil_type,temperature,crop_density,humidity,sunlight_exposure,water_source,wind_speed,growth_stage):
    import os
    # === Initial state ===
    initial_state = {
        'soil_moisture': soil_moisture,                
        'N': N,                            
        'P': P,                           
        'K': K,                            
        'N_percentage': N_percent,                
        'P_percentage': P_percent,
        'K_percentage': K_percent,
        'ph': ph,                            
        'label': label,
        'soil_type': soil_type,                       
        'temperature': temperature,                    
        'crop_density': crop_density,                   
        'humidity': humidity,                       
        'sunlight_exposure':sunlight_exposure,             
        'water_source': water_source,                    
        'wind_speed': wind_speed,                      
        'growth_stage': growth_stage,                    
        'water_availability': "medium",
        'fertilizer_availability': "medium",
        'water_used': 0.0,
        'fertilizer_used': 0.0
    }

    # Set working directory to script locati
    
    # Now read the CSV
    try:
        df = pd.read_csv("src\core\entities\FS25.csv")
        farm_problem = optimization_problem(initial_state, df)        
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