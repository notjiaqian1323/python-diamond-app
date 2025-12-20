import streamlit as st
import numpy as np
import lightgbm as lgb
import xgboost as xgb # Ensure xgboost is installed: pip install xgboost
import joblib
import os

# Import our custom modules
from utils.preprocessing import calculate_physics, prepare_input_df
from utils.visualizer import create_diamond_fig

# 1. SETUP
st.set_page_config(page_title="Diamond AI Architect", layout="wide", page_icon="💎")
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { background-color: #00B4D8; color: white; width: 100%; border-radius: 8px; font-weight: bold;}
    .price-card { background: linear-gradient(145deg, #1e2130, #13151c); padding: 20px; border-radius: 15px; border: 1px solid #333; text-align: center; }
    .metric-value { color: #00B4D8; font-size: 2.5em; font-weight: 700; }
    .warning-box { background-color: #332b00; color: #ffcc00; padding: 15px; border-radius: 8px; border-left: 5px solid #ffcc00; margin-top: 15px; font-size: 0.9em; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL CONFIGURATION & LOADING
# ---------------------------------------------------------
MODEL_REGISTRY = {
    "LightGBM (Recommended)": {"file": "lgbm_diamond_model_new.txt", "type": "lgbm"},
    "XGBoost (High Accuracy)": {"file": "xgboost_diamond_model.json", "type": "xgb"},
    "Decision Tree": {"file": "decision_tree_diamond_model.pkl", "type": "sklearn"}
}

# 2. LOAD MODEL
@st.cache_resource
def load_selected_model(model_key):
    """Loads the model based on the dropdown selection."""
    config = MODEL_REGISTRY[model_key]
    path = os.path.join('models', config["file"])
    model_type = config["type"]
    
    if not os.path.exists(path):
        return None, model_type, False # File missing

    try:
        if model_type == "lgbm":
            # Native LightGBM
            return lgb.Booster(model_file=path), model_type, True
        
        elif model_type == "xgb":
            # Native XGBoost
            model = xgb.Booster()
            model.load_model(path)
            return model, model_type, True
            
        elif model_type == "sklearn":
            # Random Forest / Decision Tree (Pickle)
            return joblib.load(path), model_type, True
            
    except Exception as e:
        st.error(f"Error loading {model_key}: {e}")
        return None, model_type, False
    
    return None, model_type, False

# 3. SIDEBAR (Inputs)
st.sidebar.header("💎 Design Your Diamond")

# --- MODEL SELECTOR ---
st.sidebar.subheader("⚙️ Prediction Engine")
selected_model_name = st.sidebar.selectbox("Choose Model", list(MODEL_REGISTRY.keys()), index=0)

# Load the model dynamically
model, model_type, is_loaded = load_selected_model(selected_model_name)

st.sidebar.divider()

u_cut = st.sidebar.selectbox("Cut", ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'])
u_color = st.sidebar.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'], index=3)
u_clarity = st.sidebar.selectbox("Clarity", ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], index=4)
st.sidebar.divider()
u_carat = st.sidebar.slider("Carat Weight", 0.2, 5.0, 1.0, 0.01)
u_depth = st.sidebar.slider("Depth %", 50.0, 75.0, 61.5)
u_table = st.sidebar.slider("Table %", 50.0, 80.0, 57.0)

predict_btn = st.sidebar.button("RUN PREDICTION")

# 4. MAIN UI
col_visual, col_stats = st.columns([2, 1])

# --- Generate Visuals (using visualizer.py) ---
fig = create_diamond_fig(u_carat, u_table, u_depth, u_color, u_clarity)
with col_visual:
    st.plotly_chart(fig, use_container_width=True)

# --- Generate Stats & Prediction (using preprocessing.py) ---
with col_stats:
    # Calculate physics
    phys_stats = calculate_physics(u_carat, u_depth)

    # Model Status Badge
    if is_loaded:
        st.success(f"🟢 Model Loaded: {selected_model_name}")
    else:
        st.error(f"🔴 File Not Found: {MODEL_REGISTRY[selected_model_name]['file']}")
        st.info("Please place the model file in the 'models/' folder.")
    
    if predict_btn and model:
        try:
            # Prepare Data using helper function
            input_df = prepare_input_df(u_carat, u_cut, u_color, u_clarity, u_depth, u_table, phys_stats['vol'])
            
            # --- PREDICTION LOGIC SWITCHER ---
            pred_log = 0.0
            
            if model_type == "lgbm":
                # LightGBM Native: Strict column alignment
                if hasattr(model, 'feature_name'):
                    input_df = input_df[model.feature_name()]
                pred_log = model.predict(input_df)[0]
                
            elif model_type == "xgb":
                # XGBoost Native: Needs DMatrix
                # Note: Columns must match training order exactly!
                dtrain = xgb.DMatrix(input_df)
                pred_log = model.predict(dtrain)[0]
                
            elif model_type == "sklearn":
                # Random Forest / DT: Input array or DF
                # Sklearn is lenient with names but strict with order
                pred_log = model.predict(input_df)[0]

            # Back-Transform
            price = np.expm1(pred_log)
            
            # Display
            st.markdown(f"""
            <div class="price-card">
                <div style="color:#888; font-size:0.9em;">ESTIMATED VALUE</div>
                <div class="metric-value">${price:,.2f}</div>
                <div style="color:#666; margin-top:10px;">Model: {selected_model_name}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # High-End Warning
            if price > 18500 or u_carat > 2.5:
                st.markdown("""<div class="warning-box">⚠️ <strong>Extrapolation Warning</strong><br>
                Value exceeds standard market data. Prediction is conservative.</div>""", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Always show specs
    st.write("")
    st.info(f"**Physical Specs:**\n- Vol: {phys_stats['vol']} mm³\n- Dim: {phys_stats['diameter']} mm (avg diameter)")

# Footer
st.divider()
st.caption("AI Diamond Valuation Prototype | Built with LightGBM & Optuna Optimization")
