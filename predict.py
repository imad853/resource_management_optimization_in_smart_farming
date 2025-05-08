def estimate_crop_yield(water_used, crop_type):
    """
    Estimate crop yield based on water used and crop type
    
    Parameters:
    water_used (float): Total water used in inches during the growing season
    crop_type (str): Type of crop ('corn', 'grain_sorghum', 'soybean', or 'wheat')
    
    Returns:
    float: Estimated yield in bushels per acre
    """
    # Crop water use efficiency parameters from K-State research
    crop_params = {
        'corn': {
            'intercept': -84.6,
            'slope': 8.33,
            'max_yield': 240,  # bushels/acre
            'max_water_use': 25  # inches
        },
        'grain_sorghum': {
            'intercept': -41.2,
            'slope': 5.32,
            'max_yield': 120,  # bushels/acre
            'max_water_use': 20  # inches
        },
        'soybean': {
            'intercept': -29.2,
            'slope': 4.44,
            'max_yield': 70,  # bushels/acre
            'max_water_use': 20  # inches
        },
        'wheat': {
            'intercept': -13.5,
            'slope': 4.85,
            'max_yield': 80,  # bushels/acre
            'max_water_use': 18  # inches
        }
    }
    
    # Check if crop type is valid
    if crop_type not in crop_params:
        valid_crops = ", ".join(crop_params.keys())
        raise ValueError(f"Invalid crop type. Valid options are: {valid_crops}")
    
    # Get parameters for the specified crop
    params = crop_params[crop_type]
    
    # Calculate yield based on linear relationship (y = intercept + slope * water_used)
    calculated_yield = params['intercept'] + params['slope'] * water_used
    
    # Apply maximum yield constraint (yield doesn't increase beyond max_water_use)
    if water_used > params['max_water_use']:
        calculated_yield = params['intercept'] + params['slope'] * params['max_water_use']
    
    # Yield can't be negative
    estimated_yield = max(0, calculated_yield)
    
    # Apply absolute maximum yield for the crop type
    estimated_yield = min(estimated_yield, params['max_yield'])
    
    return round(estimated_yield, 1)