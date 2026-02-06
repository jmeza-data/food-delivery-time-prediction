"""
Streamlit App - Food Delivery Time Prediction with AI Analysis
Professional UI with Evidently-style reports and working LLM
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Groq for LLM
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from model_pipeline import Predictor, config
    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🚚 AI Delivery Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

GROQ_API_KEY = "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"


@st.cache_resource
def get_groq_client():
    """Initialize Groq client."""
    if not GROQ_AVAILABLE:
        return None, "Groq library not installed. Run: pip install groq"
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # Test connection with an available model
        models_to_test = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ]
        
        for model in models_to_test:
            try:
                test_response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=10
                )
                return client, None
            except:
                continue
        
        return None, "No available models found"
        
    except Exception as e:
        return None, str(e)

def analyze_with_llm(order_data, predicted_time, groq_client):
    """Use LLM to provide intelligent analysis."""
    
    if groq_client is None:
        return None
    
    distance = order_data['Distance_km']
    weather = order_data['Weather']
    traffic = order_data['Traffic_Level']
    time_day = order_data['Time_of_Day']
    vehicle = order_data['Vehicle_Type']
    prep = order_data['Preparation_Time_min']
    experience = order_data['Courier_Experience_yrs']
    
    prompt = """Eres un experto en logística de entregas. Analiza esta orden:

DATOS: {}km, {}, tráfico {}, {}, {}, prep {}min, courier {}años
TIEMPO PREDICHO: {:.1f} minutos

Responde SOLO con:
1. ANÁLISIS: (2 líneas sobre la situación)
2. RECOMENDACIONES: (3 puntos específicos)
3. CLIENTE: (1 mensaje breve para el cliente)

Usa emojis. Sé conciso.""".format(
        distance, weather, traffic, time_day, vehicle, prep, experience, predicted_time
    )

    # Try multiple models in order of preference
    models_to_try = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "llama3-70b-8192"
    ]
    
    for model_name in models_to_try:
        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres experto en logística. Respondes en español, conciso y profesional."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            if model_name == models_to_try[-1]:
                return "Error LLM: Todos los modelos fallaron. Último error: " + str(e)
            continue
    
    return "Error: No se pudo conectar con ningún modelo LLM"

# ============================================================================
# CUSTOM CSS (Professional dark theme)
# ============================================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.4rem;
        color: #4ECDC4;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #4ECDC4;
        padding-bottom: 0.3rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        margin: 1.5rem 0;
        text-align: center;
    }
    
    .prediction-time {
        font-size: 3.5rem;
        font-weight: 900;
        color: white;
        margin: 0;
    }
    
    .prediction-label {
        font-size: 1rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.3rem;
    }
    
    .llm-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #4ECDC4;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4ECDC4;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #999;
        text-transform: uppercase;
    }
    
    .stButton>button {
        background: linear-gradient(120deg, #FF6B6B, #FF8E53);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.5rem;
        font-size: 1rem;
        border-radius: 8px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_prediction_pipeline():
    """Load the complete prediction pipeline."""
    if not IMPORTS_OK:
        return None, "Import error: " + str(IMPORT_ERROR)
    
    try:
        predictor = Predictor()
        
        model_path = config.MODEL_DIR / config.MODEL_NAME
        preprocessor_path = config.MODEL_DIR / config.PREPROCESSOR_NAME
        feature_engineer_path = config.MODEL_DIR / config.FEATURE_ENGINEER_NAME
        
        if not model_path.exists():
            return None, "Model file not found"
        
        predictor.load_pipeline(
            model_path=model_path,
            preprocessor_path=preprocessor_path if preprocessor_path.exists() else None,
            feature_engineer_path=feature_engineer_path if feature_engineer_path.exists() else None,
            verbose=False
        )
        
        return predictor, None
    
    except Exception as e:
        return None, "Error: " + str(e)

# ============================================================================
# VISUALIZATION FUNCTIONS (Evidently-style)
# ============================================================================

def create_distribution_chart(predicted_time):
    """Create a distribution chart showing where prediction falls."""
    fig, ax = plt.subplots(figsize=(8, 3), facecolor='#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    
    # Sample distribution
    times = np.random.normal(45, 12, 1000)
    times = np.clip(times, 15, 90)
    
    ax.hist(times, bins=30, color='#4ECDC4', alpha=0.6, edgecolor='none')
    ax.axvline(predicted_time, color='#FF6B6B', linewidth=3, label='Your Prediction')
    
    ax.set_xlabel('Delivery Time (min)', color='white', fontsize=10)
    ax.set_ylabel('Frequency', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.legend(facecolor='#1E1E1E', edgecolor='#444', labelcolor='white')
    ax.grid(True, alpha=0.1, color='white')
    
    plt.tight_layout()
    return fig

def create_gauge_chart(value, max_val=90):
    """Create a gauge chart."""
    fig, ax = plt.subplots(figsize=(6, 2), facecolor='#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    
    if value < 30:
        color = '#66BB6A'
    elif value < 50:
        color = '#FFA726'
    else:
        color = '#EF5350'
    
    ax.barh([0], [value], height=0.4, color=color, alpha=0.8)
    ax.barh([0], [max_val-value], left=value, height=0.4, color='#333', alpha=0.3)
    
    ax.set_xlim(0, max_val)
    ax.set_ylim(-0.3, 0.3)
    ax.axis('off')
    
    ax.text(value/2, 0, '{:.1f} min'.format(value), 
            ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    
    plt.tight_layout()
    return fig

def create_factors_chart(factors_data):
    """Create horizontal bar chart of factors."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    
    factors = list(factors_data.keys())
    values = list(factors_data.values())
    colors = ['#EF5350' if v > 5 else '#FFA726' if v > 0 else '#66BB6A' for v in values]
    
    bars = ax.barh(factors, values, color=colors, alpha=0.8)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.5, i, '{:+.1f}'.format(val), 
                va='center', fontsize=10, color='white', fontweight='bold')
    
    ax.set_xlabel('Impact on Delivery Time (min)', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(True, axis='x', alpha=0.1, color='white')
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load systems
    with st.spinner('🔄 Loading AI...'):
        predictor, model_error = load_prediction_pipeline()
        groq_client, llm_error = get_groq_client()
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.markdown("## 🤖 AI Systems Status")
        
        # ML Model Status
        if model_error:
            st.error("❌ ML Model")
            st.caption(model_error[:100])
        else:
            st.success("✅ ML Model: Online")
            
            if predictor and predictor.model_metadata:
                metadata = predictor.model_metadata
                
                st.markdown("### 📊 Model Performance")
                
                if 'test_metrics' in metadata:
                    metrics = metadata['test_metrics']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        <div class="metric-card">
                            <div class="metric-value">{:.3f}</div>
                            <div class="metric-label">R² Score</div>
                        </div>
                        """.format(metrics.get('r2', 0)), unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="metric-card">
                            <div class="metric-value">{:.1f}</div>
                            <div class="metric-label">MAE (min)</div>
                        </div>
                        """.format(metrics.get('mae', 0)), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class="metric-card">
                            <div class="metric-value">{:.1f}</div>
                            <div class="metric-label">RMSE (min)</div>
                        </div>
                        """.format(metrics.get('rmse', 0)), unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="metric-card">
                            <div class="metric-value">{:.1f}%</div>
                            <div class="metric-label">MAPE</div>
                        </div>
                        """.format(metrics.get('mape', 0)), unsafe_allow_html=True)
                
                st.markdown("### 🔧 Model Details")
                st.markdown("**Type:** " + metadata.get('model_type', 'N/A').upper())
                st.markdown("**Version:** v1.0")
                st.markdown("**Trained:** " + metadata.get('timestamp', 'N/A')[:10])
        
        st.markdown("---")
        
        # LLM Status
        st.markdown("### 🧠 LLM Status")
        if llm_error:
            st.error("❌ " + llm_error[:50])
        else:
            st.success("✅ Groq: Online")
            st.info("**Model:** Llama 3.3 70B")
        
        st.markdown("---")
        
        # About
        st.markdown("### 📖 About")
        st.markdown("""
        This system combines:
        - **ML Model** for predictions
        - **LLM** for intelligent analysis
        - **Evidently-style** reporting
        """)
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    st.markdown('<div class="main-header">🚚 Food Delivery Time Predictor</div>', 
                unsafe_allow_html=True)
    st.caption("AI-Powered Delivery Time Prediction & Analysis")
    
    # Input form
    st.markdown('<div class="sub-header">📝 Delivery Details</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        distance = st.slider("🗺️ Distance (km)", 0.1, 50.0, 10.5, 0.1)
        weather = st.selectbox("🌤️ Weather", ["Clear", "Cloudy", "Rainy", "Snowy", "Foggy", "Windy"], index=2)
        traffic = st.selectbox("🚦 Traffic", ["Low", "Medium", "High"], index=2)
        time_of_day = st.selectbox("🕐 Time", ["Morning", "Afternoon", "Evening", "Night"], index=2)
    
    with col2:
        vehicle = st.selectbox("🏍️ Vehicle", ["Bike", "Scooter", "Car"], index=2)
        prep_time = st.slider("⏱️ Prep Time (min)", 5, 60, 20, 1)
        experience = st.slider("👤 Courier Exp (years)", 0.0, 15.0, 3.5, 0.5)
    
    st.markdown("")
    
    if st.button("🔮 PREDICT WITH AI ANALYSIS", type="primary"):
        order = {
            "Distance_km": distance,
            "Weather": weather,
            "Traffic_Level": traffic,
            "Time_of_Day": time_of_day,
            "Vehicle_Type": vehicle,
            "Preparation_Time_min": int(prep_time),
            "Courier_Experience_yrs": experience
        }
        
        # Predict
        with st.spinner('🤖 Analyzing...'):
            if predictor:
                predicted_time = predictor.predict_single(order)
            else:
                predicted_time = (distance * 2.5 + prep_time + 
                                (10 if traffic == "High" else 5 if traffic == "Medium" else 0) +
                                (5 if weather in ["Rainy", "Snowy"] else 0) -
                                (experience * 0.5))
        
        # Results
        st.markdown('<div class="sub-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
        
        time_emoji = "⚡" if predicted_time < 30 else "🚚" if predicted_time < 50 else "🐌"
        
        st.markdown("""
        <div class="prediction-box">
            <div class="prediction-time">{} {:.1f} min</div>
            <div class="prediction-label">Estimated Delivery Time</div>
        </div>
        """.format(time_emoji, predicted_time), unsafe_allow_html=True)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        arrival = pd.Timestamp.now() + pd.Timedelta(minutes=predicted_time)
        with col1:
            st.metric("Arrival", arrival.strftime("%H:%M"))
        with col2:
            conf = "High" if 15 <= predicted_time <= 70 else "Medium"
            st.metric("Confidence", conf)
        with col3:
            speed = "Fast" if predicted_time < 30 else "Normal" if predicted_time < 50 else "Slow"
            st.metric("Speed", speed)
        with col4:
            risk = "Low" if predicted_time < 50 else "High"
            st.metric("Risk", risk)
        
        # Visualizations
        st.markdown('<div class="sub-header">📊 Visual Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribution Analysis**")
            fig1 = create_distribution_chart(predicted_time)
            st.pyplot(fig1)
            plt.close()
        
        with col2:
            st.markdown("**Time Gauge**")
            fig2 = create_gauge_chart(predicted_time)
            st.pyplot(fig2)
            plt.close()
        
        # Factor analysis
        st.markdown("**Impact Factors**")
        factors = {
            'Distance': (distance - 10) * 1.5,
            'Traffic': 10 if traffic == "High" else 3 if traffic == "Medium" else -2,
            'Weather': 5 if weather in ["Rainy", "Snowy"] else 0,
            'Prep Time': (prep_time - 15) * 0.5,
            'Experience': -(experience - 3) * 1.5
        }
        fig3 = create_factors_chart(factors)
        st.pyplot(fig3)
        plt.close()
        
        # LLM Analysis
        if groq_client:
            st.markdown('<div class="sub-header">🤖 AI Expert Analysis</div>', unsafe_allow_html=True)
            
            with st.spinner('🧠 LLM analyzing...'):
                analysis = analyze_with_llm(order, predicted_time, groq_client)
            
            if analysis and "Error" not in analysis:
                st.markdown("""
                <div class="llm-box">
                    <h4 style='margin-top:0; color:white;'>💡 Expert Insights</h4>
                    {}
                </div>
                """.format(analysis.replace('\n', '<br>')), unsafe_allow_html=True)
            else:
                st.error(analysis if analysis else "LLM Error")
        else:
            st.warning("⚠️ AI Analysis unavailable: " + (llm_error if llm_error else "Unknown error"))

if __name__ == "__main__":
    main()
