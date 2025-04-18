import pandas as pd
import numpy as np


class GoalState:
    def __init__(self, csv_path='src/data/datasets/SF24.csv'):
        self.df = pd.read_csv(csv_path)

    def estimate_optimal_soil_moisture(
        self,
        label,
        growth_stage,
        soil_type,
        crop_density_input,
        density_tolerance=2
    ):
        df = self.df
        growth_stage = int(growth_stage)
        soil_type = int(soil_type)

        #  Filter based on input conditions
        filtered = df[
            (df['label'].str.lower() == label.lower()) &
            (df['growth_stage'] == growth_stage) &
            (df['soil_type'] == soil_type) &
            (np.abs(df['crop_density'] - crop_density_input) <= density_tolerance) # the deffrence between the user input and the tuple of the crop density less than
        ]

        if filtered.empty:
            print(" No matching data found for the given conditions.")
            return None

        filtered2 = df[
            (df['label'].str.lower() == label.lower()) &
            (df['growth_stage'] == growth_stage) &
            (df['soil_type'] == soil_type) # based only on the soil type growth stage and crop density i will get the optimal env condition with it 
        ]

        #  environmental averages
        env_features = ['rainfall', 'humidity', 'temperature', 'sunlight_exposure']
        env_avgs = filtered2[env_features].mean()
        print("\n Environmental Averages from Dataset:")
        for feature in env_features:
            print(f"  • {feature.title()}: {env_avgs[feature]:.2f}")

        #  original soil moisture 
        original_soil_moisture_avg = filtered['soil_moisture'].mean()
        original_soil_moisture_std = filtered['soil_moisture'].std()
        print(f"\n Average Original Soil Moisture: {original_soil_moisture_avg:.2f}%")
        print(f"\n standard deviation  Original Soil Moisture: {original_soil_moisture_std:.2f}%")
        print(" Relevant Row Data (Soil Moisture + Environmental Factors + Crop Density):")
        print(filtered[['soil_moisture', 'rainfall', 'humidity', 'temperature', 'sunlight_exposure', 'crop_density' , 'water_usage_efficiency']])
        
        # Step 4: Adjust soil moisture based on environmental conditions
        def adjust_soil_moisture(row):
            adj = row['soil_moisture']

            #### if the standard deviation is already low dont do anything 
            if original_soil_moisture_std < 3:
                return adj

            # Calculate differences
            rain_diff = (row['rainfall'] - env_avgs['rainfall'])
            humidity_diff = (row['humidity'] - env_avgs['humidity']) 
            temp_diff = (row['temperature'] - env_avgs['temperature']) 
            sun_diff = (row['sunlight_exposure'] - env_avgs['sunlight_exposure'])

            # PARAMS 

            #### for each  soil type  i could later on add the growth stage for more detaild params 

            ## the more rain the more water it needs O.02 is small factor due to the deffrence is bigger than other factors so it will be big 
            # please before changing the parameter test as much cases as possible ; it is with minus due to high humidity leads to less soil moisture optimal ( think of it )
            # Adjust for temperature: higher temp -> more moisture needed
            #  Adjust for sunlight exposure: more sun -> more moisture    
            
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
                else:  # growth_stage == 3
                    adj -= 0.04 * rain_diff
                    adj -= 0.08 * humidity_diff
                    adj += 0.3 * temp_diff
                    adj += 0.3 * sun_diff

            else:  # soil_type == 3
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
                else:  # growth_stage == 3
                    adj -= 0.02 * rain_diff
                    adj -= 0.8 * humidity_diff
                    adj += 0.4 * temp_diff
                    adj += 1.4 * sun_diff

            return adj
        
        ## PARAMS 

        filtered = filtered.copy()
        filtered['Adjusted Soil Moisture'] = filtered.apply(adjust_soil_moisture, axis=1)

        print("\n Adjusted Soil Moisture Values:")
        print(filtered['Adjusted Soil Moisture'].values)

        # Step 5: Compute average adjusted moisture
        optimal_moisture = filtered['Adjusted Soil Moisture'].mean()
        # Inverse WUE weights
        filtered['inverse_wue'] = 1 / filtered['water_usage_efficiency']
        filtered['inv_wue_weights'] = filtered['inverse_wue'] / filtered['inverse_wue'].sum()

        # print("\n Inverse WUE Weights:")
        # print(filtered[['water_usage_efficiency', 'inv_wue_weights']])

        # Weighted average     #### each adjusted soil moisture have weight deppending on the wue the less the bigger the weight 
        weighted_optimal = np.average(
            filtered['Adjusted Soil Moisture'],
            weights=filtered['inv_wue_weights']
        )

        print(f"\n Weighted Optimal Soil Moisture (favoring low WUE): {weighted_optimal:.2f}%") ## the result 
        print(f" Std Dev (Adjusted): {filtered['Adjusted Soil Moisture'].std():.2f}%")
        print(f" Estimated Optimal Soil Moisture (normal average) : {optimal_moisture:.2f}%")

        ## if this is lees than the orginal means all the tyuples converges into one value 
        ## which is the mean soil moisture which is the OPTIMAL ONE   

        return weighted_optimal


goal = GoalState()
goal.estimate_optimal_soil_moisture(
    label="rice",
    growth_stage=1,
    soil_type=1,
    crop_density_input=14,
)