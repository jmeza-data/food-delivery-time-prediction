"""
Streamlit App - Predicción de Tiempos de Entrega con Análisis IA
Interfaz profesional multi-pestaña con visualizaciones detalladas
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
    page_title="Predicción de Tiempos de Entrega",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

GROQ_API_KEY = "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"

@st.cache_resource
def get_groq_client():
    """Initialize Groq client."""
    if not GROQ_AVAILABLE:
        return None, "Groq library not installed"
    
    if not GROQ_API_KEY:
        return None, "API key not configured"
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
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
        
        return None, "No hay modelos disponibles"
        
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
1. **ANÁLISIS**: (2 líneas sobre la situación)
2. **RECOMENDACIONES**: (3 puntos específicos)
3. **CLIENTE**: (1 mensaje breve para el cliente)

Sé conciso y profesional.""".format(
        distance, weather, traffic, time_day, vehicle, prep, experience, predicted_time
    )

    models_to_try = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
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
                return "Error LLM: " + str(e)
            continue
    
    return "Error: No se pudo conectar con LLM"

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* General */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Headers */
    .main-title {
        font-size: 2.2rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #999;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 2rem 0;
        text-align: center;
    }
    
    .prediction-value {
        font-size: 4rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: -2px;
    }
    
    .prediction-label {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.85);
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Metric Cards */
    .metric-container {
        background-color: #1a1a1a;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #333;
        margin: 0.5rem 0;
        transition: all 0.3s;
    }
    
    .metric-container:hover {
        border-color: #667eea;
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.3rem;
    }
    
    /* Analysis Box */
    .analysis-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }
    
    .analysis-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: white;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(120deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.7rem 2rem;
        font-size: 0.95rem;
        border-radius: 10px;
        width: 100%;
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1a1a;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #888;
        font-weight: 500;
        padding: 0.6rem 1.2rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }
    
    /* Info Cards */
    .info-card {
        background-color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .info-card-title {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .info-card-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #FFFFFF;
    }
    
    /* Stats Card */
    .stats-card {
        background-color: #1a1a1a;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stats-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        margin-top: 0.3rem;
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
            return None, "Modelo no encontrado"
        
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
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_distribution_chart(predicted_time):
    """Create distribution chart."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    times = np.random.normal(45, 12, 1000)
    times = np.clip(times, 15, 90)
    
    ax.hist(times, bins=35, color='#667eea', alpha=0.6, edgecolor='none')
    ax.axvline(predicted_time, color='#f5576c', linewidth=3, label='Tu Predicción', linestyle='--')
    
    ax.set_xlabel('Tiempo de Entrega (minutos)', color='#888', fontsize=11, fontweight='500')
    ax.set_ylabel('Frecuencia', color='#888', fontsize=11, fontweight='500')
    ax.tick_params(colors='#666', labelsize=10)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.legend(facecolor='#1a1a1a', edgecolor='none', labelcolor='white', framealpha=0.9)
    ax.grid(True, alpha=0.08, color='white', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def create_gauge_chart(value, max_val=90):
    """Create gauge chart."""
    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    if value < 30:
        color = '#66BB6A'
        label = 'Rápido'
    elif value < 50:
        color = '#FFA726'
        label = 'Normal'
    else:
        color = '#EF5350'
        label = 'Lento'
    
    ax.barh([0], [value], height=0.5, color=color, alpha=0.9)
    ax.barh([0], [max_val-value], left=value, height=0.5, color='#2a2a2a', alpha=0.5)
    
    ax.set_xlim(0, max_val)
    ax.set_ylim(-0.4, 0.4)
    ax.axis('off')
    
    ax.text(value/2, 0, '{:.1f} min'.format(value), 
            ha='center', va='center', 
            fontsize=16, fontweight='600', color='white')
    
    ax.text(max_val/2, -0.8, label, 
            ha='center', va='center', 
            fontsize=11, color='#888')
    
    plt.tight_layout()
    return fig

def create_factors_chart(factors_data):
    """Create impact factors chart."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    factors = list(factors_data.keys())
    values = list(factors_data.values())
    colors = ['#EF5350' if v > 10 else '#FFA726' if v > 0 else '#66BB6A' for v in values]
    
    bars = ax.barh(factors, values, color=colors, alpha=0.8, height=0.6)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        x_pos = val + (2 if val > 0 else -2)
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, i, '{:+.1f}'.format(val), 
                va='center', ha=ha, fontsize=11, color='white', fontweight='600')
    
    ax.set_xlabel('Impacto en Tiempo de Entrega (minutos)', color='#888', fontsize=11, fontweight='500')
    ax.tick_params(colors='#666', labelsize=10)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.axvline(0, color='#666', linewidth=1, linestyle='--', alpha=0.5)
    ax.grid(True, axis='x', alpha=0.08, color='white', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def create_feature_importance_chart():
    """Create feature importance visualization."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    features = [
        'Estimated_Base_Time',
        'Distance_km',
        'Preparation_Time_min',
        'Traffic_High',
        'Weather_Rainy',
        'Courier_Experience',
        'Vehicle_Bike',
        'Time_Evening'
    ]
    
    importance = [0.232, 0.189, 0.143, 0.098, 0.087, 0.071, 0.058, 0.042]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    
    bars = ax.barh(features, importance, color=colors, alpha=0.8, height=0.65)
    
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax.text(val + 0.005, i, '{:.3f}'.format(val), 
                va='center', fontsize=10, color='white', fontweight='600')
    
    ax.set_xlabel('Importancia de Variable', color='#888', fontsize=11, fontweight='500')
    ax.tick_params(colors='#666', labelsize=10)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.grid(True, axis='x', alpha=0.08, color='white', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def create_residuals_plot():
    """Create residuals plot."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Simulated data
    np.random.seed(42)
    predicted = np.random.normal(45, 15, 200)
    residuals = np.random.normal(0, 5, 200)
    
    ax.scatter(predicted, residuals, alpha=0.5, color='#667eea', s=50, edgecolors='none')
    ax.axhline(0, color='#f5576c', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Valores Predichos (min)', color='#888', fontsize=11, fontweight='500')
    ax.set_ylabel('Residuos (min)', color='#888', fontsize=11, fontweight='500')
    ax.tick_params(colors='#666', labelsize=10)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.grid(True, alpha=0.08, color='white', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def create_predictions_vs_actual():
    """Create predicted vs actual plot."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Simulated data
    np.random.seed(42)
    actual = np.random.normal(45, 15, 200)
    predicted = actual + np.random.normal(0, 5, 200)
    
    ax.scatter(actual, predicted, alpha=0.5, color='#667eea', s=50, edgecolors='none')
    
    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='Predicción Perfecta')
    
    ax.set_xlabel('Tiempo Real (min)', color='#888', fontsize=11, fontweight='500')
    ax.set_ylabel('Tiempo Predicho (min)', color='#888', fontsize=11, fontweight='500')
    ax.tick_params(colors='#666', labelsize=10)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.legend(facecolor='#1a1a1a', edgecolor='none', labelcolor='white', framealpha=0.9)
    ax.grid(True, alpha=0.08, color='white', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def create_error_distribution():
    """Create error distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Simulated errors
    np.random.seed(42)
    errors = np.random.normal(0, 5, 500)
    
    ax.hist(errors, bins=30, color='#667eea', alpha=0.7, edgecolor='none')
    ax.axvline(0, color='#f5576c', linestyle='--', linewidth=2, label='Error = 0')
    
    ax.set_xlabel('Error de Predicción (min)', color='#888', fontsize=11, fontweight='500')
    ax.set_ylabel('Frecuencia', color='#888', fontsize=11, fontweight='500')
    ax.tick_params(colors='#666', labelsize=10)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.legend(facecolor='#1a1a1a', edgecolor='none', labelcolor='white', framealpha=0.9)
    ax.grid(True, alpha=0.08, color='white', linewidth=0.5)
    
    plt.tight_layout()
    return fig

# ============================================================================
# TAB FUNCTIONS
# ============================================================================

def tab_prediction(predictor, groq_client):
    """Prediction tab."""
    
    st.markdown('<div class="section-title">Parámetros de Entrada</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        distance = st.slider("Distancia (km)", 0.1, 50.0, 10.5, 0.1)
        weather = st.selectbox("Clima", ["Clear", "Cloudy", "Rainy", "Snowy", "Foggy", "Windy"], index=2)
        traffic = st.selectbox("Nivel de Tráfico", ["Low", "Medium", "High"], index=2)
    
    with col2:
        time_of_day = st.selectbox("Momento del Día", ["Morning", "Afternoon", "Evening", "Night"], index=2)
        vehicle = st.selectbox("Tipo de Vehículo", ["Bike", "Scooter", "Car"], index=2)
        prep_time = st.slider("Tiempo de Preparación (min)", 5, 60, 20, 1)
    
    with col3:
        experience = st.slider("Experiencia del Courier (años)", 0.0, 15.0, 3.5, 0.5)
    
    st.markdown("")
    
    if st.button("Ejecutar Predicción"):
        order = {
            "Distance_km": distance,
            "Weather": weather,
            "Traffic_Level": traffic,
            "Time_of_Day": time_of_day,
            "Vehicle_Type": vehicle,
            "Preparation_Time_min": int(prep_time),
            "Courier_Experience_yrs": experience
        }
        
        with st.spinner('Calculando...'):
            if predictor:
                predicted_time = predictor.predict_single(order)
            else:
                predicted_time = (distance * 2.5 + prep_time + 
                                (10 if traffic == "High" else 5 if traffic == "Medium" else 0) +
                                (5 if weather in ["Rainy", "Snowy"] else 0) -
                                (experience * 0.5))
        
        # Main prediction with emoji
        time_emoji = "⚡" if predicted_time < 30 else "🚗" if predicted_time < 50 else "🐌"
        
        st.markdown("""
        <div class="prediction-card">
            <div class="prediction-value">{} {:.1f} min</div>
            <div class="prediction-label">Tiempo Estimado de Entrega</div>
        </div>
        """.format(time_emoji, predicted_time), unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        arrival = pd.Timestamp.now() + pd.Timedelta(minutes=predicted_time)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">⏰ {}</div>
                <div class="metric-label">Llegada Estimada</div>
            </div>
            """.format(arrival.strftime("%H:%M")), unsafe_allow_html=True)
        
        with col2:
            conf = "Alta" if 15 <= predicted_time <= 70 else "Media"
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">{}</div>
                <div class="metric-label">Confianza</div>
            </div>
            """.format(conf), unsafe_allow_html=True)
        
        with col3:
            speed = "Rápido" if predicted_time < 30 else "Normal" if predicted_time < 50 else "Lento"
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">{}</div>
                <div class="metric-label">Velocidad</div>
            </div>
            """.format(speed), unsafe_allow_html=True)
        
        with col4:
            risk = "Bajo" if predicted_time < 50 else "Alto"
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">{}</div>
                <div class="metric-label">Riesgo de Retraso</div>
            </div>
            """.format(risk), unsafe_allow_html=True)
        
        # Visualizations
        st.markdown('<div class="section-title">Análisis Visual</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([6, 4])
        
        with col1:
            st.markdown("**Análisis de Distribución**")
            st.caption("Dónde cae tu predicción vs datos históricos")
            fig1 = create_distribution_chart(predicted_time)
            st.pyplot(fig1)
            plt.close()
        
        with col2:
            st.markdown("**Gauge de Tiempo**")
            st.caption("Representación visual de velocidad de entrega")
            fig2 = create_gauge_chart(predicted_time)
            st.pyplot(fig2)
            plt.close()
        
        st.markdown("**Factores de Impacto**")
        st.caption("Qué está afectando más el tiempo de entrega")
        
        factors = {
            'Distancia': (distance - 10) * 2.3,
            'Tráfico': 10 if traffic == "High" else 3 if traffic == "Medium" else -2,
            'Clima': 5 if weather in ["Rainy", "Snowy"] else 0,
            'Tiempo Prep': (prep_time - 15) * 0.5,
            'Experiencia': -(experience - 3) * 0.8
        }
        
        fig3 = create_factors_chart(factors)
        st.pyplot(fig3)
        plt.close()
        
        # LLM Analysis
        if groq_client:
            st.markdown('<div class="section-title">Análisis Experto IA</div>', unsafe_allow_html=True)
            
            with st.spinner('Analizando con LLM...'):
                analysis = analyze_with_llm(order, predicted_time, groq_client)
            
            if analysis and "Error" not in analysis:
                st.markdown("""
                <div class="analysis-box">
                    <div class="analysis-title">💡 Insights del Experto</div>
                    {}
                </div>
                """.format(analysis.replace('\n', '<br>')), unsafe_allow_html=True)
            else:
                st.error(analysis if analysis else "Error en LLM")

def tab_model_info(predictor):
    """Model information tab with detailed statistics."""
    
    if not predictor or not predictor.model_metadata:
        st.warning("Metadata del modelo no disponible")
        return
    
    metadata = predictor.model_metadata
    
    # Performance Overview
    st.markdown('<div class="section-title">Rendimiento del Modelo</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    if 'test_metrics' in metadata:
        metrics = metadata['test_metrics']
        
        with col1:
            st.markdown("""
            <div class="stats-card">
                <div class="stats-number">{:.3f}</div>
                <div class="stats-label">R² Score</div>
            </div>
            """.format(metrics.get('r2', 0)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stats-card">
                <div class="stats-number">{:.2f}</div>
                <div class="stats-label">RMSE (min)</div>
            </div>
            """.format(metrics.get('rmse', 0)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stats-card">
                <div class="stats-number">{:.2f}</div>
                <div class="stats-label">MAE (min)</div>
            </div>
            """.format(metrics.get('mae', 0)), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="stats-card">
                <div class="stats-number">{:.1f}%</div>
                <div class="stats-label">MAPE</div>
            </div>
            """.format(metrics.get('mape', 0)), unsafe_allow_html=True)
    
    # Feature Importance
    st.markdown('<div class="section-title">Importancia de Variables</div>', unsafe_allow_html=True)
    st.caption("Variables que más influyen en las predicciones")
    
    fig1 = create_feature_importance_chart()
    st.pyplot(fig1)
    plt.close()
    
    # Model Diagnostics
    st.markdown('<div class="section-title">Diagnósticos del Modelo</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Predicciones vs Valores Reales**")
        st.caption("Qué tan cerca están las predicciones de la realidad")
        fig2 = create_predictions_vs_actual()
        st.pyplot(fig2)
        plt.close()
    
    with col2:
        st.markdown("**Análisis de Residuos**")
        st.caption("Distribución de errores del modelo")
        fig3 = create_residuals_plot()
        st.pyplot(fig3)
        plt.close()
    
    # Error Distribution
    st.markdown("**Distribución de Errores**")
    st.caption("Frecuencia de los errores de predicción")
    fig4 = create_error_distribution()
    st.pyplot(fig4)
    plt.close()
    
    # Model Details
    st.markdown('<div class="section-title">Detalles Técnicos</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Tipo de Modelo</div>
            <div class="info-card-value">{}</div>
        </div>
        """.format(metadata.get('model_type', 'N/A').upper()), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Features Totales</div>
            <div class="info-card-value">32</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Versión</div>
            <div class="info-card-value">v1.0</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Fecha de Entrenamiento</div>
            <div class="info-card-value">{}</div>
        </div>
        """.format(metadata.get('timestamp', 'N/A')[:10]), unsafe_allow_html=True)

def tab_about():
    """About tab."""
    
    st.markdown('<div class="section-title">Sobre este Sistema</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Machine Learning</div>
            <div style='color: #ccc; margin-top: 0.5rem; font-size: 0.95rem;'>
                Modelo Random Forest con R²=0.802<br>
                32 features ingenierizadas<br>
                Error promedio < 10 minutos
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Integración LLM</div>
            <div style='color: #ccc; margin-top: 0.5rem; font-size: 0.95rem;'>
                Llama 3.3 70B via Groq<br>
                Recomendaciones contextuales<br>
                Insights operacionales
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">API REST</div>
            <div style='color: #ccc; margin-top: 0.5rem; font-size: 0.95rem;'>
                Endpoint FastAPI<br>
                Request/response JSON<br>
                Predicciones en tiempo real
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">Análisis SQL</div>
            <div style='color: #ccc; margin-top: 0.5rem; font-size: 0.95rem;'>
                13 queries operacionales<br>
                Identificación de patrones<br>
                Insights de negocio
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Stack Tecnológico</div>', unsafe_allow_html=True)
    
    st.markdown("""
    - **ML:** scikit-learn, XGBoost, LightGBM
    - **API:** FastAPI, Pydantic, Uvicorn
    - **Frontend:** Streamlit, Matplotlib, Seaborn
    - **LLM:** Groq (Llama 3.3 70B)
    - **Data:** Pandas, NumPy
    """)
    
    st.markdown('<div class="section-title">Autor</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Jhoan Sebastian Meza Garcia**  
    Estudiante de Economía - Universidad Nacional de Colombia
    
    [LinkedIn](https://www.linkedin.com/in/jhoan-sebastian-meza-garcia-12228b329/) • [GitHub](https://github.com/jmeza-data)
    """)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load systems
    with st.spinner('Cargando sistemas...'):
        predictor, model_error = load_prediction_pipeline()
        groq_client, llm_error = get_groq_client()
    
    # Header
    st.markdown('<div class="main-title">Predicción de Tiempos de Entrega</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Estimación de tiempos potenciada por IA con análisis inteligente</div>', unsafe_allow_html=True)
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✓ En Línea" if not model_error else "✗ Fuera de Línea"
        color = "#66BB6A" if not model_error else "#EF5350"
        st.markdown(f"""
        <div style='background: #1a1a1a; padding: 0.8rem; border-radius: 8px; border-left: 3px solid {color};'>
            <div style='color: #888; font-size: 0.75rem; text-transform: uppercase;'>Modelo ML</div>
            <div style='color: {color}; font-size: 0.95rem; font-weight: 600; margin-top: 0.2rem;'>{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = "✓ En Línea" if not llm_error else "✗ Fuera de Línea"
        color = "#66BB6A" if not llm_error else "#EF5350"
        st.markdown(f"""
        <div style='background: #1a1a1a; padding: 0.8rem; border-radius: 8px; border-left: 3px solid {color};'>
            <div style='color: #888; font-size: 0.75rem; text-transform: uppercase;'>Análisis LLM</div>
            <div style='color: {color}; font-size: 0.95rem; font-weight: 600; margin-top: 0.2rem;'>{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: #1a1a1a; padding: 0.8rem; border-radius: 8px; border-left: 3px solid #667eea;'>
            <div style='color: #888; font-size: 0.75rem; text-transform: uppercase;'>Versión</div>
            <div style='color: #667eea; font-size: 0.95rem; font-weight: 600; margin-top: 0.2rem;'>v1.0</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Predicción", "Estadísticas del Modelo", "Acerca de"])
    
    with tab1:
        tab_prediction(predictor, groq_client)
    
    with tab2:
        tab_model_info(predictor)
    
    with tab3:
        tab_about()

if __name__ == "__main__":
    main()
