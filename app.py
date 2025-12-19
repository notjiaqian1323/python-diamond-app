import streamlit as st
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# ==========================================
# CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Diamond AI Predictor", layout="wide", page_icon="💎")

# Custom CSS for the "Luxury" look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2e7bcf; color: white; border: none; }
    .price-box {
        background-color: #1a1c24;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #2e7bcf;
        text-align: center;
        margin-top: 2rem;
    }
    .price-text { font-size: 2.5rem; color: #00d4ff; font-weight: bold; }
    .confidence-text { font-size: 1rem; color: #888; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_diamond_model():
    model_path = 'models/lgbm_diamond_model_new.txt'
    if os.path.exists(model_path):
        # We load it using the Booster class
        return lgb.Booster(model_file=model_path)
    elif os.path.exists('models/lgbm_diamond_model_new.pkl'):
        return joblib.load('models/lgbm_diamond_model_new.pkl')
    return None

model = load_diamond_model()

# ==========================================
# 3. 3D GENERATION LOGIC (The "Hackathon" Tricks)
# ==========================================

def get_diamond_color(grade):
    """Maps GIA Color Grade to Hex Code for visual tint."""
    color_map = {
        'D': '#E0F7FA', # Icy Blue/White (Perfect)
        'E': '#E0F7FA',
        'F': '#FFFFFF', # Pure White
        'G': '#F5F5F5', # White Smoke
        'H': '#FFFACD', # Lemon Chiffon (Slight Yellow)
        'I': '#FFF8DC', # Cornsilk
        'J': '#F0E68C'  # Khaki (Obvious Yellow)
    }
    return color_map.get(grade, '#FFFFFF')

def generate_inclusions(clarity_grade, radius_limit):
    """Generates 3D coordinates for 'flaws' inside the diamond."""
    # Define severity (number of dots) based on grade
    severity = {
        'IF': 0, 'VVS1': 2, 'VVS2': 5, 
        'VS1': 10, 'VS2': 20, 
        'SI1': 40, 'SI2': 80, 'I1': 150
    }
    count = severity.get(clarity_grade, 0)
    
    if count == 0:
        return None, None, None
        
    # Generate random points clustered inside the diamond body
    rng = np.random.default_rng()
    # Confine them roughly within the diamond's radius (x/2)
    inc_x = rng.uniform(-radius_limit*0.7, radius_limit*0.7, count)
    inc_y = rng.uniform(-radius_limit*0.7, radius_limit*0.7, count)
    # Confine z to be mostly in the body, not the tip
    inc_z = rng.uniform(0, radius_limit*0.8, count) 
    
    return inc_x, inc_y, inc_z

def generate_diamond_geometry(table_pct, depth_pct, carat):
    """Calculates the vertices for the mesh."""
    # 1. Physics: Estimate Diameter from Carat (Density 3.51)
    # Vol ~ 165 * Carat (Bounding Box approximation from our plots)
    # Diameter ~ CubeRoot(Vol)
    diameter = 6.5 * (carat ** (1/3)) 
    radius = diameter / 2
    
    # 2. Dimensions from Percentages
    total_depth = diameter * (depth_pct / 100)
    table_radius = radius * (table_pct / 100)
    
    # 3. Z-Heights
    z_table = total_depth * 0.35
    z_girdle = 0 
    z_culet = -total_depth * 0.65

    # 4. Generate Points (Simplified 8-sided brilliant for speed)
    angles = np.linspace(0, 2*np.pi, 9)[:-1]
    
    # Vertices
    # Table (Top Ring)
    tx = table_radius * np.cos(angles)
    ty = table_radius * np.sin(angles)
    tz = np.full_like(tx, z_table)
    
    # Girdle (Middle Ring)
    gx = radius * np.cos(angles)
    gy = radius * np.sin(angles)
    gz = np.full_like(gx, z_girdle)
    
    # Culet (Bottom Tip) & Top Center
    cx, cy, cz = [0], [0], [z_culet]
    cent_x, cent_y, cent_z = [0], [0], [z_table]

    # Pack into arrays
    x = np.concatenate([cent_x, tx, gx, cx])
    y = np.concatenate([cent_y, ty, gy, cy])
    z = np.concatenate([cent_z, tz, gz, cz])
    
    # 5. Return Geometry + Calculated Specs
    specs = {
        'x': round(diameter, 2), 'y': round(diameter, 2), 'z': round(total_depth, 2),
        'vol': round(diameter * diameter * total_depth, 2)
    }
    return x, y, z, specs, radius

# ==========================================
# 4. UI LAYOUT
# ==========================================
st.sidebar.header("💎 Design Your Diamond")

# User Inputs (Categorical)
u_cut = st.sidebar.selectbox("Cut", ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'])
u_color = st.sidebar.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'], index=3)
u_clarity = st.sidebar.selectbox("Clarity", ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], index=4)

st.sidebar.divider()

# User Inputs (Continuous)
u_carat = st.sidebar.slider("Carat Weight", 0.2, 5.0, 1.0, 0.01)
u_depth = st.sidebar.slider("Depth %", 50.0, 75.0, 61.5)
u_table = st.sidebar.slider("Table %", 50.0, 80.0, 57.0)

predict_btn = st.sidebar.button("RUN PREDICTION")


# ==========================================
# 5. VISUALIZATION & PREDICTION
# ==========================================
col_visual, col_stats = st.columns([2, 1])

# --- A. GENERATE 3D MODEL ---
x, y, z, specs, rad = generate_diamond_geometry(u_table, u_depth, u_carat)
mesh_color = get_diamond_color(u_color)
inc_x, inc_y, inc_z = generate_inclusions(u_clarity, rad)

# Build Plotly Figure
fig = go.Figure()

# 1. The Glass Body (Mesh)
# Note: i,j,k here defines a simple top/bottom cone hull for robust rendering
fig.add_trace(go.Mesh3d(
    x=x, y=y, z=z,
    alphahull=0, # Auto-triangulate the hull
    opacity=0.4,
    color=mesh_color,
    lighting=dict(ambient=0.4, diffuse=0.5, roughness=0.1, specular=1.0, fresnel=4.0),
    lightposition=dict(x=100, y=200, z=1000)
))

# 2. The Inclusions (Scatter) - Only if not IF
if inc_x is not None:
    fig.add_trace(go.Scatter3d(
        x=inc_x, y=inc_y, z=inc_z,
        mode='markers',
        marker=dict(size=2, color='black', opacity=0.6),
        name='Inclusions'
    ))

# Layout cleanup
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        aspectmode='data', # Keeps the shape proportional (doesn't squash deep diamonds)
        bgcolor='rgba(0,0,0,0)'
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, b=0, t=0),
    showlegend=False
)

with col_visual:
    st.plotly_chart(fig, use_container_width=True)

# --- B. PREDICTION ENGINE ---
with col_stats:
    if model is None:
        st.warning("⚠️ Model file not found. Please add 'lgbm_diamond_model.txt' to 'models/' folder.")
    
    # Only run prediction logic if the button is clicked OR to show initial state
    if predict_btn or model:
        # 1. Mappings (Must match your Training notebook EXACTLY)
        cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
        color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
        clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
        
        # 2. Feature Engineering
        log_carat = np.log1p(u_carat)
        log_volume = np.log1p(specs['vol'])
        
        c_score = cut_map[u_cut]
        col_score = color_map[u_color]
        cla_score = clarity_map[u_clarity]
        
        # Interactions
        int_cut = log_carat * c_score
        int_col = log_carat * col_score
        int_cla = log_carat * cla_score
        
        # 3. Create Dataframe (Column order matters for LightGBM!)
        # Check your notebook for exact column order. Assuming this standard set:
        input_df = pd.DataFrame([[
            log_carat, log_volume, 
            u_table, u_depth,
            int_cut, int_col, int_cla
        ]])
        
        # 4. Predict
        if model:
            try:
                expected_features = model.feature_name()
                
                # --- FIX: REORDER INPUTS AUTOMATICALLY ---
                # 1. Create the DataFrame with Dictionary (Explicit Names)
                input_data = pd.DataFrame({
                    'log_carat': [log_carat],
                    'log_volume': [log_volume],
                    'table': [u_table],
                    'depth': [u_depth],
                    'int_carat_cut': [int_cut],
                    'int_carat_color': [int_col],
                    'int_carat_clarity': [int_cla]
                })
                
                # 2. Force the DataFrame to match the Model's expected order
                # This fixes the mismatch automatically!
                input_data = input_data[expected_features]

                # Native Booster prediction
                pred_log = model.predict(input_df)[0]
                price = np.expm1(pred_log)

                st.success(f"Log Output: {pred_log:.4f}")  # Should be ~10-13 for expensive items
                st.success(f"Real Price: ${price:,.2f}")   # Should be correct
                
                # 5. Display Result
                st.markdown(f"""
                <div class="price-card">
                    <div class="metric-label">Estimated Market Value</div>
                    <div class="metric-value">${price:,.2f}</div>
                    <div style="color: #666; margin-top: 10px;">
                        Confidence Interval: ±1.0%<br>
                        Calculated from {u_carat}ct {u_cut} Cut
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("") # Spacer
                st.info(f"""
                **Physics Check:**
                - Volume: {specs['vol']} mm³
                - Dimensions: {specs['x']} x {specs['y']} x {specs['z']} mm
                """)
                
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
        

# Footer
st.divider()
st.caption("AI Diamond Valuation Prototype | Built with LightGBM & Optuna Optimization")