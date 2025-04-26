"""
This function calculates the cost of going from parent state to child state by calculating the sum of the sum of the added resources multiplied by their availability and their cost irl
"""

"""
water_availability : high _ medium _ low based on remaining water 
fertilizer _ availability : high _ medium _ low based on remaining fertilizer


"""
def cost_function(self,action,water_availability,fertilizer_availability) :
    if (water_availability == "high" and fertilizer_availability == "high") or (water_availability == "medium" and fertilizer_availability == "medium") or ((water_availability == "low" and fertilizer_availability == "low"))   :
        """
        if both have same availability levels -> we will take into consideration only that water is less expensive then fertilizer 
        """
        water_cost = action["water_added"]
        fertilizer_cost = (action["N_added"] + action["P_added"] + action["K_added"])*2
    elif water_availability == "medium" and fertilizer_availability == "high" or (water_availability == "low" and fertilizer_availability == "medium") :
        """
        Here since water is available in medium levels, we will consider that it costs the same as using fertilizer
        """
        water_cost = action["water_added"]
        fertilizer_cost = action["N_added"] + action["P_added"] + action["K_added"]
    elif (water_availability == "high" and fertilizer_availability == "medium") or (water_availability == "medium" and fertilizer_availability == "low")  :
        water_cost = action["water_added"]
        fertilizer_cost = (action["N_added"] + action["P_added"] + action["K_added"])*3
    elif (water_availability == "low" and fertilizer_availability == "high") :
        water_cost = action["water_added"]*2
        fertilizer_cost = (action["N_added"] + action["P_added"] + action["K_added"])
    elif water_availability == "high" and fertilizer_availability == "low" :
        water_cost = action["water_added"]
        fertilizer_cost = (action["N_added"] + action["P_added"] + action["K_added"])*4

    return water_cost + fertilizer_cost