import streamlit as st
import numpy as np
import lightgbm as lgb
import xgboost as xgb
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
    .price-card { 
        background: linear-gradient(145deg, #1e2130, #13151c); 
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid #00B4D8; 
        text-align: center; 
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.2);
    }
    .metric-value { color: #00B4D8; font-size: 2.8em; font-weight: 700; margin-bottom: 5px; }
    .accuracy-tag { 
        background-color: #00B4D822; 
        color: #00B4D8; 
        padding: 5px 15px; 
        border-radius: 20px; 
        font-size: 0.85em; 
        font-weight: bold;
        display: inline-block;
        margin-bottom: 15px;
    }
    .range-text { color: #888; font-size: 0.9em; margin-top: 10px; font-style: italic; }
    .stat-box { 
        display: flex; 
        justify-content: space-around; 
        margin-top: 20px; 
        border-top: 1px solid #333; 
        padding-top: 15px;
    }
    .stat-item { text-align: center; }
    .stat-label { font-size: 0.75em; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .stat-val { font-size: 1.1em; color: #ccc; font-weight: 600; }
    .warning-box { background-color: #332b00; color: #ffcc00; padding: 15px; border-radius: 8px; border-left: 5px solid #ffcc00; margin-top: 15px; font-size: 0.9em; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL CONFIGURATION (Updated with Performance Metrics)
# ---------------------------------------------------------
# MAE and MAPE values are based on our previous Tuning Performance Results
MODEL_REGISTRY = {
    "LightGBM (Recommended)": {
        "file": "lgbm_diamond_model_new.txt", 
        "type": "lgbm",
        "mae": 282.31, 
        "mape": 7.87,
        "r2": 0.9802
    },
    "XGBoost (High Accuracy)": {
        "file": "xgboost_diamond_model.json", 
        "type": "xgb",
        "mae": 310.45, 
        "mape": 8.20,
        "r2": 0.9785
    },
    "Decision Tree": {
        "file": "decision_tree_diamond_model.pkl", 
        "type": "sklearn",
        "mae": 450.12, 
        "mape": 12.4,
        "r2": 0.9410
    }
}

@st.cache_resource
def load_selected_model(model_key):
    config = MODEL_REGISTRY[model_key]
    path = os.path.join('models', config["file"])
    model_type = config["type"]
    if not os.path.exists(path): return None, model_type, False
    try:
        if model_type == "lgbm": return lgb.Booster(model_file=path), model_type, True
        elif model_type == "xgb":
            model = xgb.Booster(); model.load_model(path); return model, model_type, True
        elif model_type == "sklearn": return joblib.load(path), model_type, True
    except Exception as e:
        st.error(f"Error loading {model_key}: {e}")
        return None, model_type, False
    return None, model_type, False

# 3. SIDEBAR (Inputs)
st.sidebar.header("💎 Design Your Diamond")
selected_model_name = st.sidebar.selectbox("Choose Prediction Engine", list(MODEL_REGISTRY.keys()), index=0)
model, model_type, is_loaded = load_selected_model(selected_model_name)
model_meta = MODEL_REGISTRY[selected_model_name]

st.sidebar.divider()
u_cut = st.sidebar.selectbox("Cut Quality", ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'])
u_color = st.sidebar.selectbox("Color Grade", ['D', 'E', 'F', 'G', 'H', 'I', 'J'], index=3)
u_clarity = st.sidebar.selectbox("Clarity Grade", ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], index=4)
st.sidebar.divider()
u_carat = st.sidebar.slider("Carat Weight", 0.2, 5.0, 1.0, 0.01)
u_depth = st.sidebar.slider("Depth %", 50.0, 75.0, 61.5)
u_table = st.sidebar.slider("Table %", 50.0, 80.0, 57.0)
predict_btn = st.sidebar.button("CALCULATE VALUATION")

# 4. MAIN UI
col_visual, col_stats = st.columns([2, 1])

# --- 3D Visualizer ---
fig = create_diamond_fig(u_carat, u_table, u_depth, u_color, u_clarity, u_cut)
with col_visual:
    st.plotly_chart(fig, use_container_width=True)

# --- Stats & Prediction ---
with col_stats:
    phys_stats = calculate_physics(u_carat, u_depth)

    if predict_btn and model:
        try:
            input_df = prepare_input_df(u_carat, u_cut, u_color, u_clarity, u_depth, u_table, phys_stats['vol'])
            
            # Prediction Logic
            pred_log = 0.0
            if model_type == "lgbm":
                if hasattr(model, 'feature_name'): input_df = input_df[model.feature_name()]
                pred_log = model.predict(input_df)[0]
            elif model_type == "xgb":
                pred_log = model.predict(xgb.DMatrix(input_df))[0]
            elif model_type == "sklearn":
                pred_log = model.predict(input_df)[0]

            price = np.expm1(pred_log)
            
            # CALCULATE ERROR RANGES
            lower_bound = price - model_meta['mae']
            upper_bound = price + model_meta['mae']

            # Display Fancy Price Card
            st.markdown(f"""
            <div class="price-card">
                <div class="accuracy-tag">✓ {model_meta['r2']*100:.1f}% EXPLAINED VARIANCE</div>
                <div style="color:#888; font-size:0.8em; text-transform:uppercase;">Estimated Market Value</div>
                <div class="metric-value">${price:,.2f}</div>
                <div class="range-text">Expected range: ${lower_bound:,.0f} — ${upper_bound:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if price > 18500 or u_carat > 2.5:
                st.markdown("""<div class="warning-box">⚠️ <strong>Market Outlier Warning</strong><br>
                High-carat diamond pricing is volatile. The ± Error margin may be wider than usual.</div>""", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
    else:
        st.info("Adjust the sliders and click **Calculate Valuation** to see the price and accuracy metrics.")

    # Physical Specs Footer
    st.write("")
    st.caption(f"**Physical Audit:** {phys_stats['vol']} mm³ | Est. {phys_stats['diameter']}mm width")

st.divider()
st.caption(f"Engine: {selected_model_name} | Protocol: Log-Transformed Regression | Data: Physics-Filtered Diamond Set")
