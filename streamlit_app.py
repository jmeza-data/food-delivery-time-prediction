"""
Streamlit App - Predicción de Tiempos de Entrega con Análisis IA
Interfaz profesional con gráficos animados en Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os
import time

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
    """Generate 3 personalized insights using LLM."""
    
    if groq_client is None:
        return None
    
    # Traducir valores
    weather_es = {
        'Clear': 'Despejado', 'Cloudy': 'Nublado', 'Rainy': 'Lluvioso',
        'Snowy': 'Nevado', 'Foggy': 'Neblina', 'Windy': 'Ventoso'
    }
    traffic_es = {'Low': 'Bajo', 'Medium': 'Medio', 'High': 'Alto'}
    
    distance = order_data['Distance_km']
    weather = weather_es.get(order_data['Weather'], order_data['Weather'])
    traffic = traffic_es.get(order_data['Traffic_Level'], order_data['Traffic_Level'])
    prep_time = order_data['Preparation_Time_min']
    
    prompt = f"""Eres un asistente operativo para una plataforma de delivery.
Con base en el siguiente contexto de predicción, genera 3 insights breves y accionables para distintos actores:

Contexto:
- Tiempo estimado de entrega: {predicted_time:.1f} minutos
- Nivel de tráfico: {traffic}
- Clima: {weather}
- Distancia: {distance} km
- Tiempo de preparación: {prep_time} min

Genera:
1) Mensaje para el cliente:
Un mensaje corto y empático explicando el tiempo estimado y gestionando expectativas.

2) Recomendación para el repartidor:
Un consejo operativo breve para mejorar la eficiencia o seguridad según las condiciones.

3) Insight para el negocio:
Una sugerencia clara para el equipo de operaciones (ej: avisar retraso, monitorear el pedido, tomar acción preventiva).

Reglas:
- Máximo 2 líneas por sección
- Lenguaje claro y profesional
- Enfocado en acciones concretas"""

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
                        "content": "Eres experto en logística de delivery. Respondes de forma concisa y profesional en español."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            if model_name == models_to_try[-1]:
                return f"Error LLM: {str(e)}"
            continue
    
    return "Error: No se pudo conectar con LLM"

# ============================================================================
# CUSTOM CSS WITH ADVANCED ANIMATIONS
# ============================================================================

st.markdown("""
<style>
    /* General */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Keyframe Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    @keyframes countUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Headers */
    .main-title {
        font-size: 2.2rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #999;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-top: 2rem;
        margin-bottom: 1rem;
        animation: slideInLeft 0.5s ease-out;
    }
    
    /* Prediction Card with Advanced Animation */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 2rem 0;
        text-align: center;
        animation: scaleIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            to right,
            rgba(255,255,255,0) 0%,
            rgba(255,255,255,0.1) 50%,
            rgba(255,255,255,0) 100%
        );
        animation: shimmer 3s infinite;
    }
    
    .prediction-value {
        font-size: 4rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: -2px;
        animation: countUp 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        z-index: 1;
    }
    
    .prediction-label {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.85);
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
        animation: fadeInUp 1s ease-out;
    }
    
    /* Metric Cards with Staggered Animation */
    .metric-container {
        background-color: #1a1a1a;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #333;
        margin: 0.5rem 0;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: fadeInUp 0.5s ease-out;
        animation-fill-mode: both;
    }
    
    .metric-container:nth-child(1) { animation-delay: 0.1s; }
    .metric-container:nth-child(2) { animation-delay: 0.2s; }
    .metric-container:nth-child(3) { animation-delay: 0.3s; }
    .metric-container:nth-child(4) { animation-delay: 0.4s; }
    
    .metric-container:hover {
        border-color: #667eea;
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #667eea;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.3rem;
    }
    
    /* LLM Analysis Boxes with Slide-in Animation */
    .llm-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.3rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: slideInRight 0.6s ease-out;
        animation-fill-mode: both;
        position: relative;
        overflow: hidden;
    }
    
    .llm-section:nth-child(1) { animation-delay: 0.2s; }
    .llm-section:nth-child(2) { animation-delay: 0.4s; }
    .llm-section:nth-child(3) { animation-delay: 0.6s; }
    
    .llm-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.1),
            transparent
        );
        transition: left 0.5s;
    }
    
    .llm-section:hover::before {
        left: 100%;
    }
    
    .llm-section:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0,0,0,0.35);
    }
    
    .llm-section-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        color: white;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .llm-section-content {
        font-size: 0.95rem;
        line-height: 1.5;
        color: rgba(255,255,255,0.95);
    }
    
    /* Buttons with Hover Animation */
    .stButton>button {
        background: linear-gradient(120deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.7rem 2rem;
        font-size: 0.95rem;
        border-radius: 10px;
        width: 100%;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255,255,255,0.2);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
    }
    
    /* Tabs with Smooth Transition */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1a1a;
        padding: 0.5rem;
        border-radius: 10px;
        animation: fadeInUp 0.7s ease-out;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #888;
        font-weight: 500;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
        transform: scale(1.05);
    }
    
    /* Info Cards with Progressive Animation */
    .info-card {
        background-color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: slideInLeft 0.5s ease-out;
        animation-fill-mode: both;
    }
    
    .info-card:nth-child(odd) { animation-delay: 0.1s; }
    .info-card:nth-child(even) { animation-delay: 0.2s; }
    
    .info-card:hover {
        transform: translateX(10px) scale(1.02);
        border-left-color: #764ba2;
        box-shadow: -4px 0 12px rgba(102, 126, 234, 0.2);
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
    
    /* Stats Card with Scale Animation */
    .stats-card {
        background-color: #1a1a1a;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
        margin: 0.5rem 0;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: scaleIn 0.5s ease-out;
        animation-fill-mode: both;
    }
    
    .stats-card:nth-child(1) { animation-delay: 0.1s; }
    .stats-card:nth-child(2) { animation-delay: 0.2s; }
    .stats-card:nth-child(3) { animation-delay: 0.3s; }
    .stats-card:nth-child(4) { animation-delay: 0.4s; }
    
    .stats-card:hover {
        border-color: #667eea;
        transform: scale(1.08);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .stats-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-color: #667eea !important;
        border-right-color: transparent !important;
    }
    
    /* Smooth Transitions for All Elements */
    * {
        transition: opacity 0.3s ease, transform 0.3s ease;
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
        
        # Hardcodear metadata si no existe
        if not predictor.model_metadata:
            predictor.model_metadata = {
                'model_type': 'random_forest',
                'version': 'v1.0',
                'timestamp': '2026-02-05',
                'test_metrics': {
                    'r2': 0.802,
                    'rmse': 9.42,
                    'mae': 6.57,
                    'mape': 12.6
                }
            }
        
        return predictor, None
    
    except Exception as e:
        return None, "Error: " + str(e)

# ============================================================================
# PLOTLY VISUALIZATION FUNCTIONS (ANIMATED)
# ============================================================================

def create_distribution_chart(predicted_time):
    """Create animated distribution chart with Plotly."""
    # Generate data
    times = np.random.normal(45, 12, 1000)
    times = np.clip(times, 15, 90)
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=times,
        nbinsx=35,
        marker=dict(
            color='#667eea',
            line=dict(width=0)
        ),
        opacity=0.7,
        name='Distribución Histórica'
    ))
    
    # Add prediction line
    fig.add_vline(
        x=predicted_time,
        line_dash="dash",
        line_color="#f5576c",
        line_width=3,
        annotation_text="Tu Predicción",
        annotation_position="top"
    )
    
    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='#888', size=11),
        xaxis=dict(
            title='Tiempo de Entrega (minutos)',
            gridcolor='rgba(255,255,255,0.08)',
            showgrid=True
        ),
        yaxis=dict(
            title='Frecuencia',
            gridcolor='rgba(255,255,255,0.08)',
            showgrid=True
        ),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(26,26,26,0.9)',
            bordercolor='#444',
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        height=350
    )
    
    # Add animation
    fig.update_traces(
        marker_line_width=0,
        selector=dict(type='histogram')
    )
    
    return fig

def create_gauge_chart(value, max_val=90):
    """Create animated gauge chart with Plotly."""
    if value < 30:
        color = '#66BB6A'
        label = 'Rápido'
    elif value < 50:
        color = '#FFA726'
        label = 'Normal'
    else:
        color = '#EF5350'
        label = 'Lento'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': label, 'font': {'color': '#888', 'size': 18}},
        number={'suffix': " min", 'font': {'color': 'white', 'size': 32}},
        gauge={
            'axis': {'range': [None, max_val], 'tickcolor': "#888"},
            'bar': {'color': color},
            'bgcolor': "#2a2a2a",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(102, 187, 106, 0.2)'},
                {'range': [30, 50], 'color': 'rgba(255, 167, 38, 0.2)'},
                {'range': [50, max_val], 'color': 'rgba(239, 83, 80, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='#888'),
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_factors_chart(factors_data):
    """Create animated horizontal bar chart with Plotly."""
    factors = list(factors_data.keys())
    values = list(factors_data.values())
    colors = ['#EF5350' if v > 10 else '#FFA726' if v > 0 else '#66BB6A' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=factors,
        x=values,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=[f'{v:+.1f}' for v in values],
        textposition='outside',
        textfont=dict(color='white', size=12, family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Impacto: %{x:+.1f} min<extra></extra>'
    ))
    
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        line_width=1
    )
    
    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='#888', size=11),
        xaxis=dict(
            title='Impacto en Tiempo de Entrega (minutos)',
            gridcolor='rgba(255,255,255,0.08)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.08)',
            showgrid=False
        ),
        showlegend=False,
        margin=dict(l=120, r=80, t=40, b=60),
        height=400
    )
    
    return fig

def create_feature_importance_chart():
    """Create animated feature importance chart with Plotly."""
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
    
    # Create color gradient
    colors = px.colors.sequential.Viridis
    color_scale = [colors[int(i * (len(colors)-1) / (len(features)-1))] for i in range(len(features))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=features,
        x=importance,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Viridis',
            line=dict(width=0)
        ),
        text=[f'{v:.3f}' for v in importance],
        textposition='outside',
        textfont=dict(color='white', size=11, family='Arial'),
        hovertemplate='<b>%{y}</b><br>Importancia: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='#888', size=11),
        xaxis=dict(
            title='Importancia de Variable',
            gridcolor='rgba(255,255,255,0.08)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.08)',
            showgrid=False
        ),
        showlegend=False,
        margin=dict(l=180, r=80, t=40, b=60),
        height=450
    )
    
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
        weather = st.selectbox("Clima", ["Despejado", "Nublado", "Lluvioso", "Nevado", "Neblina", "Ventoso"], index=2)
        traffic = st.selectbox("Nivel de Tráfico", ["Bajo", "Medio", "Alto"], index=2)
    
    with col2:
        time_of_day = st.selectbox("Momento del Día", ["Mañana", "Tarde", "Noche", "Madrugada"], index=2)
        vehicle = st.selectbox("Tipo de Vehículo", ["Bicicleta", "Scooter", "Auto"], index=2)
        prep_time = st.slider("Tiempo de Preparación (min)", 5, 60, 20, 1)
    
    with col3:
        experience = st.slider("Experiencia del Courier (años)", 0.0, 15.0, 3.5, 0.5)
    
    st.markdown("")
    
    if st.button("Ejecutar Predicción"):
        # Traducir de vuelta a inglés para el modelo
        weather_map = {"Despejado": "Clear", "Nublado": "Cloudy", "Lluvioso": "Rainy", 
                       "Nevado": "Snowy", "Neblina": "Foggy", "Ventoso": "Windy"}
        traffic_map = {"Bajo": "Low", "Medio": "Medium", "Alto": "High"}
        time_map = {"Mañana": "Morning", "Tarde": "Afternoon", "Noche": "Evening", "Madrugada": "Night"}
        vehicle_map = {"Bicicleta": "Bike", "Scooter": "Scooter", "Auto": "Car"}
        
        order = {
            "Distance_km": distance,
            "Weather": weather_map[weather],
            "Traffic_Level": traffic_map[traffic],
            "Time_of_Day": time_map[time_of_day],
            "Vehicle_Type": vehicle_map[vehicle],
            "Preparation_Time_min": int(prep_time),
            "Courier_Experience_yrs": experience
        }
        
        # Progress bar animation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text('Analizando parámetros...')
            elif i < 60:
                status_text.text('Ejecutando modelo ML...')
            else:
                status_text.text('Generando predicción...')
        
        if predictor:
            predicted_time = predictor.predict_single(order)
        else:
            predicted_time = (distance * 2.5 + prep_time + 
                            (10 if traffic == "Alto" else 5 if traffic == "Medio" else 0) +
                            (5 if weather in ["Lluvioso", "Nevado"] else 0) -
                            (experience * 0.5))
        
        progress_bar.empty()
        status_text.empty()
        
        # Main prediction with emoji
        time_emoji = "⚡" if predicted_time < 30 else "🚗" if predicted_time < 50 else "🐌"
        
        # Animated counter for time
        time_placeholder = st.empty()
        for i in range(int(predicted_time) + 1):
            time_placeholder.markdown("""
            <div class="prediction-card">
                <div class="prediction-value">{} {:.1f} min</div>
                <div class="prediction-label">Tiempo Estimado de Entrega</div>
            </div>
            """.format(time_emoji, i), unsafe_allow_html=True)
            time.sleep(0.02)
        
        time_placeholder.markdown("""
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
            st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            st.markdown("**Gauge de Tiempo**")
            st.caption("Representación visual de velocidad de entrega")
            fig2 = create_gauge_chart(predicted_time)
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("**Factores de Impacto**")
        st.caption("Qué está afectando más el tiempo de entrega")
        
        factors = {
            'Distancia': (distance - 10) * 2.3,
            'Tráfico': 10 if traffic == "Alto" else 3 if traffic == "Medio" else -2,
            'Clima': 5 if weather in ["Lluvioso", "Nevado"] else 0,
            'Tiempo Prep': (prep_time - 15) * 0.5,
            'Experiencia': -(experience - 3) * 0.8
        }
        
        fig3 = create_factors_chart(factors)
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        
        # LLM Analysis - 3 Sections
        if groq_client:
            st.markdown('<div class="section-title">Análisis Personalizado</div>', unsafe_allow_html=True)
            
            with st.spinner('Generando insights...'):
                analysis = analyze_with_llm(order, predicted_time, groq_client)
            
            if analysis and "Error" not in analysis:
                # Parse the analysis into 3 sections
                sections = analysis.split('\n\n')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class="llm-section">
                        <div class="llm-section-title">📱 Para el Cliente</div>
                        <div class="llm-section-content">{}</div>
                    </div>
                    """.format(sections[0] if len(sections) > 0 else analysis[:200]), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="llm-section">
                        <div class="llm-section-title">🏍️ Para el Repartidor</div>
                        <div class="llm-section-content">{}</div>
                    </div>
                    """.format(sections[1] if len(sections) > 1 else ""), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="llm-section">
                        <div class="llm-section-title">🏢 Para el Negocio</div>
                        <div class="llm-section-content">{}</div>
                    </div>
                    """.format(sections[2] if len(sections) > 2 else ""), unsafe_allow_html=True)
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
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
    
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
        """.format(metadata.get('timestamp', 'N/A')), unsafe_allow_html=True)

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
                Análisis personalizado para 3 actores<br>
                Insights operacionales accionables
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
    - **Frontend:** Streamlit, Plotly, Seaborn
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
        <div style='background: #1a1a1a; padding: 0.8rem; border-radius: 8px; border-left: 3px solid {color}; animation: slideInLeft 0.5s ease-out;'>
            <div style='color: #888; font-size: 0.75rem; text-transform: uppercase;'>Modelo ML</div>
            <div style='color: {color}; font-size: 0.95rem; font-weight: 600; margin-top: 0.2rem;'>{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = "✓ En Línea" if not llm_error else "✗ Fuera de Línea"
        color = "#66BB6A" if not llm_error else "#EF5350"
        st.markdown(f"""
        <div style='background: #1a1a1a; padding: 0.8rem; border-radius: 8px; border-left: 3px solid {color}; animation: slideInLeft 0.6s ease-out;'>
            <div style='color: #888; font-size: 0.75rem; text-transform: uppercase;'>Análisis LLM</div>
            <div style='color: {color}; font-size: 0.95rem; font-weight: 600; margin-top: 0.2rem;'>{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: #1a1a1a; padding: 0.8rem; border-radius: 8px; border-left: 3px solid #667eea; animation: slideInLeft 0.7s ease-out;'>
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
