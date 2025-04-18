

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.entities.goal_state.goal_state import GoalState


if __name__ == "__main__":
    # Create an instance of the class
    goal = GoalState()

    # Call the method to test it
    optimal_soil_moisture = goal.estimate_optimal_soil_moisture(
        label="rice",           # Replace with your test crop
        growth_stage=1,         # Example stage
        soil_type=1,            # Example soil type
        crop_density_input=14   # Example crop density
    )

    print("\nFinal Estimated Optimal Soil Moisture:", optimal_soil_moisture)