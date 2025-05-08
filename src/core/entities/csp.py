
import pandas as pd
from goal_state import GoalState
import os

class CSPProblem: 
    def __init__(self, initial_state , df, max_N, max_P, max_K, max_water):
          ### the constructor 
        
        self.max_N = max_N
        self.max_P = max_P
        self.max_K = max_K
        self.max_water = max_water

        self.df = df  
        ## we will se by which unit we will add 
        
        # domains = {
        #     'water_amount': list(range(0, 100)), ## this should be changed 
        #     'N': list(range(0, 101)),
        #     'P': list(range(0, 101)),
        #     'K': list(range(0, 101)),
              #'ph': list(range(0, 101)),
        # }
        domains = {   ## the domains are adding by 0.5 and not 1 for more accuracy ex. [0.5,1,1.5 ---- 100]
            'water_amount': [x *  0.1 for x in range(0, max_water*10)],
            'N': [x * 0.1 for x in range(0, int(max_N*10))],            
            'P': [x * 0.5 for x in range(0, int(max_P*10))],
            'K': [x * 0.5 for x in range(0, int(max_K*10))],
        }

        variables = {'Water_amount', 'N', 'P', 'K' }
        
        self.initial_state = initial_state 
        self.solution = None
        self.variables = variables
        self.domains = domains
        goal =  GoalState()
        goal.estimate_optimal_params( self.initial_state['label'],
            self.initial_state['growth_stage'],
            self.initial_state['soil_type'],
            self.initial_state['crop_density'], 
            self.df, )
        self.goal = goal 
        self.optimal_ranges = {
            'soil_moisture': (self.goal.optimal_soil_moisture - 1, self.goal.optimal_soil_moisture + 1),
            'ph': (self.goal.optimal_ph - 30, self.goal.optimal_ph + 30),
            'N': (self.goal.optimal_n - 2.5, self.goal.optimal_n + 2.5),
            'P': (self.goal.optimal_p - 2.5, self.goal.optimal_p + 2.5),
            'K': (self.goal.optimal_k - 2.5, self.goal.optimal_k + 2.5),
        }
       
   
    def in_range(self, actual, optimal_min, optimal_max):
     return optimal_min <= actual <= optimal_max
       
    def transition_model(self):
        """ Transition model: how soil reacts to added water """
        return {
            "add_water": {
                "units": "1 L/m²",
                    "soil_moisture_increase_per_L": {
                    1: self.calculate_moisture_increase("Sandy"),
                    2: self.calculate_moisture_increase("Loamy"),
                    3: self.calculate_moisture_increase("Clay")
                },
                "npk_uptake_increase_per_1_percent_moisture": {
                    1: {"N": 0.2, "P": 0.15, "K": 0.18},
                    2: {"N": 0.25, "P": 0.2, "K": 0.22},
                    3: {"N": 0.15, "P": 0.25, "K": 0.2}
                }, 
                 "ph_change_per_L_by_water_source": {
                    1: {1: -0.01, 2: -0.015, 3: -0.01},   
                    2: {1: 0.01, 2: 0.015, 3: 0.01}, 
                    3: {1: -0.02, 2: -0.025, 3: -0.02}   
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
        }
    

    ##### CONSTRAINTS ########

    def water_moisture_constraint( self , water_amount):   ### for the water added if it  satisfies the soil moisture constraint 
        ###
        soil_type = self.initial_state['soil_type']  
        initial_moisture = self.initial_state['soil_moisture']
        ####
        transition = self.transition_model()   
        # Get  how much moisture increases per liter for the soil type
        moisture_increase_per_L = transition['add_water']['soil_moisture_increase_per_L'][soil_type]
        # Predict new soil moisture after adding water
        new_moisture = initial_moisture + water_amount * moisture_increase_per_L
        optimal_min, optimal_max = self.optimal_ranges['soil_moisture']   ## the accepted range 
        return self.in_range(new_moisture, optimal_min, optimal_max) ### returns true if the water 
    

    ###### water and n constraint 
    def water_and_n_constraint(self, water_amount, n_amount):
        soil_type = self.initial_state['soil_type']
        initial_moisture = self.initial_state['soil_moisture']
        initial_n = self.initial_state['N']
        transition = self.transition_model()
        moisture_increase_per_L = transition['add_water']['soil_moisture_increase_per_L'][soil_type]
          ### how much the n will be added for one soil moisture increase 
        n_increase_per_1_percent_moisture = transition['add_water']['npk_uptake_increase_per_1_percent_moisture'][soil_type]['N']
        new_moisture = initial_moisture + water_amount * moisture_increase_per_L
        ## how much soil mositure increased 
        moisture_fark = new_moisture - initial_moisture
        ## calculating the new n 
        n_uptake_increase = moisture_fark * n_increase_per_1_percent_moisture
        new_n = initial_n + n_uptake_increase + n_amount
        ## for  the optimal ranges 
        optimal_moisture_min, optimal_moisture_max = self.optimal_ranges['soil_moisture']
        optimal_n_min, optimal_n_max = self.optimal_ranges['N']
         ### if both in the optimal range 
        return (self.in_range(new_moisture, optimal_moisture_min, optimal_moisture_max) and
                self.in_range(new_n, optimal_n_min, optimal_n_max))
    
    def water_and_p_constraint(self, water_amount, p_amount):
        soil_type = self.initial_state['soil_type']
        initial_moisture = self.initial_state['soil_moisture']
        initial_p = self.initial_state['P']
        transition = self.transition_model()
        moisture_increase_per_L = transition['add_water']['soil_moisture_increase_per_L'][soil_type]
          ### how much the p will be added for one soil moisture increase 
        p_increase_per_1_percent_moisture = transition['add_water']['npk_uptake_increase_per_1_percent_moisture'][soil_type]['P']
        new_moisture = initial_moisture + water_amount * moisture_increase_per_L
        ## how much soil mositure increased 
        moisture_fark = new_moisture - initial_moisture
        ## calculating the new p
        p_uptake_increase = moisture_fark * p_increase_per_1_percent_moisture
        new_p = initial_p + p_uptake_increase + p_amount
        
        optimal_moisture_min, optimal_moisture_max = self.optimal_ranges['soil_moisture']
        optimal_p_min, optimal_p_max = self.optimal_ranges['P']

         ### if both in the optimal range 
        return (self.in_range(new_moisture, optimal_moisture_min, optimal_moisture_max) and
                self.in_range(new_p, optimal_p_min, optimal_p_max))
    
    def water_and_k_constraint(self, water_amount, k_amount):
        soil_type = self.initial_state['soil_type']
        initial_moisture = self.initial_state['soil_moisture']
        initial_k = self.initial_state['K']
        transition = self.transition_model()
        moisture_increase_per_L = transition['add_water']['soil_moisture_increase_per_L'][soil_type]
          ### how much the k will be added for one soil moisture increase 
        k_increase_per_1_percent_moisture = transition['add_water']['npk_uptake_increase_per_1_percent_moisture'][soil_type]['K']
        new_moisture = initial_moisture + water_amount * moisture_increase_per_L
        ## how much soil mositure increased 
        moisture_fark = new_moisture - initial_moisture
        ## calculating the new k
        k_uptake_increase = moisture_fark * k_increase_per_1_percent_moisture
        new_k = initial_k + k_uptake_increase + k_amount
        
        optimal_moisture_min, optimal_moisture_max = self.optimal_ranges['soil_moisture']
        optimal_k_min, optimal_k_max = self.optimal_ranges['K']

         ### if both in the optimal range 
        return (self.in_range(new_moisture, optimal_moisture_min, optimal_moisture_max) and
                self.in_range(new_k, optimal_k_min, optimal_k_max))
    

    def water_and_ph_constraint(self, water_amount):
            soil_type = self.initial_state['soil_type']
            initial_moisture = self.initial_state['soil_moisture']
            initial_ph = self.initial_state['ph']
            water_source = self.initial_state['water_source']
            transition = self.transition_model()
            moisture_increase_per_L = transition['add_water']['soil_moisture_increase_per_L'][soil_type]
            ph_increase_per_L = transition["add_water"]['ph_change_per_L_by_water_source'][water_source][soil_type]
            ### how much the n will be added for one soil moisture increase 
            new_moisture = initial_moisture + water_amount * moisture_increase_per_L
            new_ph = water_amount * ph_increase_per_L + initial_ph
            ## how much soil mositure increased 

            ## for  the optimal ranges 
            optimal_moisture_min, optimal_moisture_max = self.optimal_ranges['soil_moisture']
            optimal_ph_min, optimal_ph_max = self.optimal_ranges['ph']
            ### if both in the optimal range 
            return (self.in_range(new_moisture, optimal_moisture_min, optimal_moisture_max) and
                    self.in_range(new_ph,  optimal_ph_min, optimal_ph_max))
        

    ###### CONSTRIANTS ######

     ## you could  use the queue (normal ac3) with revisit or without a queue in optimized Ac3 with our problem 
    def ac3(self, use_queue=True):

        print("Starting AC-3 algorithm...")
        self.domains['water_amount'] = [
            water for water in self.domains['water_amount'] 
            if self.water_moisture_constraint(water) and self.water_and_ph_constraint(water)
        ]

        print("After pruning by soil moisture and pH constraints:")
        print("Water domain:", self.domains['water_amount'])
        print('\n')

        if not use_queue:
           
           ## without using a queue OPTIMIZED  
            final_water_domain = []
            for water in self.domains['water_amount']:
                n_ok = any(self.water_and_n_constraint(water, n) for n in self.domains['N'])
                p_ok = any(self.water_and_p_constraint(water, p) for p in self.domains['P'])
                k_ok = any(self.water_and_k_constraint(water, k) for k in self.domains['K'])
                if n_ok and p_ok and k_ok:
                    final_water_domain.append(water)
            self.domains['water_amount'] = final_water_domain

            
            self.domains['N'] = [n for n in self.domains['N']
                                if any(self.water_and_n_constraint(water, n) for water in self.domains['water_amount'])]
            self.domains['P'] = [p for p in self.domains['P']
                                if any(self.water_and_p_constraint(water, p) for water in self.domains['water_amount'])]
            self.domains['K'] = [k for k in self.domains['K']
                                if any(self.water_and_k_constraint(water, k) for water in self.domains['water_amount'])]

        else:

            #### using a queue 
      
            queue = [
                ('water_amount', 'N'),
                ('N', 'water_amount'),
                ('water_amount', 'P'),
                ('P', 'water_amount'),
                ('water_amount', 'K'),
                ('K', 'water_amount') , 
            ]

            def revise(X, Y):
              
                revised = False
                new_domain = []

                for x in self.domains[X]:
                    if X == 'water_amount' and Y == 'N':
                        satisfies = any(self.water_and_n_constraint(x, n) for n in self.domains['N'])
                    elif X == 'N' and Y == 'water_amount':
                        satisfies = any(self.water_and_n_constraint(water, x) for water in self.domains['water_amount'])
                    elif X == 'water_amount' and Y == 'P':
                        satisfies = any(self.water_and_p_constraint(x, p) for p in self.domains['P'])
                    elif X == 'P' and Y == 'water_amount':
                        satisfies = any(self.water_and_p_constraint(water, x) for water in self.domains['water_amount'])
                    elif X == 'water_amount' and Y == 'K':
                        satisfies = any(self.water_and_k_constraint(x, k) for k in self.domains['K'])
                    elif X == 'K' and Y == 'water_amount':
                        satisfies = any(self.water_and_k_constraint(water, x) for water in self.domains['water_amount'])
                    else:
                        satisfies = True  

                    if satisfies:  
                        new_domain.append(x)
                    else:
                        revised = True

                if revised:
                    self.domains[X] = new_domain

                return revised

            while queue:
                (X, Y) = queue.pop(0)
                if revise(X, Y):
                    if X == 'N' or X == 'P' or X == 'K' :
                        queue.append(('water_amount', X))
                    elif X == 'water_amount':
                        queue.append(('N', 'water_amount'))
                        queue.append(('P', 'water_amount'))
                        queue.append(('K', 'water_amount'))
                        

        print("Final Domains after AC-3 pruning:")
        print("Water:", self.domains['water_amount'])
        print("N:", self.domains['N'])
        print("P:", self.domains['P'])
        print("K:", self.domains['K'])
        print('\n')
        
    def backtrack(self, pruned_domains):
        best_solution = None
        min_total_error = float('inf')

        for water in pruned_domains['Water']:
            if not self.water_moisture_constraint(water):
                continue

            for n in pruned_domains['N']:
                if not self.water_and_n_constraint(water, n):
                    continue

                for p in pruned_domains['P']:
                    if not self.water_and_p_constraint(water, p):
                        continue

                    for k in pruned_domains['K']:
                        if not self.water_and_k_constraint(water, k):
                            continue

                        # Valid solution found
                        # Optionally: rank based on closeness to goal
                        total_error = (
                            abs(n - self.goal.optimal_n) +
                            abs(p - self.goal.optimal_p) +
                            abs(k - self.goal.optimal_k)
                        )

                        if total_error < min_total_error:
                            min_total_error = total_error
                            best_solution = {
                                'Water': water,
                                'N': n,
                                'P': p,
                                'K': k
                            }

        return best_solution
    
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
            



initial_state = {
    'soil_moisture': 10.8,
    'N': 20.33,
    'P': 40.83,
    'K': 10.33,
    'ph': 7.5,
    'label': "rice",
    'soil_type': 1,
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
    'fertilizer_used': 0.0 ,
     "water_source":1,
}

### just testing for now 
def main():
    try:
        
        
        # Get the directory where your script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to the correct directory (adjust as needed)
        csv_path = os.path.join(script_dir, "FS25.csv")  # If in same folder as script
        # OR if CSV is in a parent/subdirectory:
        # csv_path = os.path.join(script_dir, "../FS25.csv")  # Parent dir
        # csv_path = os.path.join(script_dir, "data/FS25.csv")  # Subdir

        df = pd.read_csv(csv_path)
        farm_problem = CSPProblem(initial_state, df,100,50, 50,50)
        farm_problem.ac3()

        #getting the values from domains after pruning
        pruned_domains = {
            "Water": farm_problem.domains["water_amount"],
            "N": farm_problem.domains["N"],
            "P": farm_problem.domains["P"],
            "K": farm_problem.domains["K"]
        }

        #drtlha print parcq ay t returni
        print(f"The optimal Values are : \n{farm_problem.backtrack(pruned_domains)}")

    except FileNotFoundError:
        print("\n❌ Error: 'FS25.csv' not found.")
    except Exception as e:
        print("\n❌ An error occurred:", str(e))

if __name__ == "__main__":
    main()



        
        
            