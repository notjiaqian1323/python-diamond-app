import pandas as pd
import numpy as np

# --- CONSTANTS & MAPPINGS ---
# These match the training notebook exactly
CUT_MAP = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
COLOR_MAP = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
CLARITY_MAP = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

def calculate_physics(carat, depth_pct):
    """
    Calculates estimated physical stats needed for the model and UI display.
    """
    # Physics: Estimate Diameter from Carat (Density ~3.51)
    # 6.5mm is standard for 1ct. Scale by cube root.
    diameter = 6.5 * (carat ** (1/3)) 
    
    # Calculate depth in mm
    total_depth_mm = diameter * (depth_pct / 100)
    
    # Volume (Bounding Box approximation used in training)
    # Note: Training likely used x*y*z. Here we assume x=y=diameter.
    est_volume = diameter * diameter * total_depth_mm
    
    return {
        'diameter': round(diameter, 2),
        'depth_mm': round(total_depth_mm, 2),
        'vol': round(est_volume, 2)
    }

def prepare_input_df(u_carat, u_cut, u_color, u_clarity, u_depth, u_table, u_vol):
    """
    Feature Engineering Pipeline:
    Transforms raw inputs -> Log Transforms -> Interactions -> DataFrame
    """
    # 1. Feature Engineering
    log_carat = np.log1p(u_carat)
    log_volume = np.log1p(u_vol)
    
    c_score = CUT_MAP[u_cut]
    col_score = COLOR_MAP[u_color]
    cla_score = CLARITY_MAP[u_clarity]
    
    # 2. Interactions (Multipliers)
    int_cut = log_carat * c_score
    int_col = log_carat * col_score
    int_cla = log_carat * cla_score

    # 3. Build Dictionary
    # Keys MUST match model feature names
    data_dict = {
        'log_carat': [log_carat],
        'log_volume': [log_volume],
        'table': [u_table],
        'depth': [u_depth],
        'int_carat_color': [int_col],
        'int_carat_clarity': [int_cla],
        'int_carat_cut': [int_cut],
    }
    
    return pd.DataFrame(data_dict)

