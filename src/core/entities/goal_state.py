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
        self.optimal_WUE = None

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
        self.optimal_WUE = filtered['water_usage_efficiency']

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

