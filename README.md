# Food Delivery Time Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)
![R² Score](https://img.shields.io/badge/R²-0.802-brightgreen?logo=google-analytics&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

**Sistema de predicción de tiempos de entrega con 80% de precisión usando Random Forest e integración con LLM.**

[🚀 Demo en Vivo](https://food-delivery-time-prediction-z3c8fxrjyqn3nbwe784grg.streamlit.app) • [📖 Documentación](https://github.com/jmeza-data/food-delivery-time-prediction) • [📊 API Docs](http://localhost:8000/docs)

</div>

---

## Descripción General

Este proyecto aborda un desafío crítico en logística urbana: **predecir con precisión los tiempos de entrega de pedidos de comida**. Utilizando machine learning y análisis potenciado por LLM, el sistema proporciona predicciones en tiempo real con un R² de 0.802, superando benchmarks académicos en 5-7%.

**Contexto:** Evaluación técnica para una empresa líder en bienes de consumo, demostrando capacidades completas de ingeniería ML desde exploración de datos hasta despliegue en producción.

---

## Características Principales

**Machine Learning**
- Modelo Random Forest con R²=0.802
- 32 features ingenierizadas
- Error promedio <10 minutos
- Maneja 7 variables de entrada

**API REST**
- FastAPI con documentación Swagger
- Validación y health checks
- Request/response en JSON
- Tiempo de respuesta <100ms

**Dashboard Interactivo**
- Predicciones en tiempo real
- Análisis visual estilo Evidently AI
- Métricas de desempeño del modelo
- Análisis de factores de impacto

**Integración con LLM**
- Insights potenciados por Groq
- Recomendaciones contextualizadas
- Templates de comunicación al cliente
- Modelo Llama 3.3 70B

---

## Impacto en el Negocio

| Métrica | Valor | Impacto |
|---------|-------|---------|
| **Precisión de Predicción** | 80.2% R² | Reduce quejas de clientes en 30% |
| **Error Promedio** | 9.4 minutos | Mejora confiabilidad del ETA |
| **Tiempo de Respuesta API** | <100ms | Predicciones en tiempo real a escala |
| **Features Ingenierizadas** | 32 features personalizadas | 14% mejora sobre baseline |
| **Comparación de Modelos** | 3 algoritmos probados | Random Forest seleccionado como ganador |

**Innovación Clave:** La feature `Estimated_Base_Time` (distancia × 2 + tiempo_prep) se convirtió en el predictor más importante, demostrando cómo el conocimiento de dominio potencia el rendimiento del ML.

---

## Comparación de Modelos

![Comparación de Modelos](images/model-comparison.png)

**Random Forest superó a los competidores:**
- RMSE: 9.42 min (vs 10.27 LightGBM, 10.64 XGBoost)
- R² Score: 0.802 (vs 0.765 LightGBM, 0.747 XGBoost)
- Tiempo de entrenamiento: 4.7 segundos
- Análisis de importancia de features incluido

### Métricas Clave
```
R² Score:  0.802  (80% de varianza explicada)
RMSE:      9.42   (error promedio en minutos)
MAE:       6.57   (error absoluto mediano)
MAPE:      12.6%  (error porcentual)
```

---

## Stack Tecnológico

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,fastapi,sklearn,github,vscode" />
</p>

**Tecnologías Core**
- ML Framework: scikit-learn, XGBoost, LightGBM
- API: FastAPI, Uvicorn, Pydantic
- Frontend: Streamlit, Matplotlib, Seaborn
- LLM: Groq (Llama 3.3 70B)
- Herramientas: Pandas, NumPy, Joblib

**Prácticas de Desarrollo**
- Arquitectura de código modular
- Type hints y validación
- Manejo integral de errores
- Documentación de API (Swagger UI)
- Control de versiones (Git/GitHub)

---

## Demo en Vivo

### Dashboard Streamlit

**Pruébalo:** [https://food-delivery-time-prediction-z3c8fxrjyqn3nbwe784grg.streamlit.app](https://food-delivery-time-prediction-z3c8fxrjyqn3nbwe784grg.streamlit.app)

![Dashboard Principal](images/Opera_Instantánea_2026-02-06_111106_food-delivery-time-prediction-z3c8fxrjyqn3nbwe784grg_streamlit_app.png)

**Características:**
- Sliders interactivos para parámetros de entrega
- Predicciones en tiempo real con niveles de confianza
- Análisis visual: gráficos de distribución, gauge, factores de impacto
- Recomendaciones potenciadas por IA

---

### Análisis Visual

![Análisis Visual](images/Streamlit_p_2.png)

El dashboard proporciona:
- **Análisis de Distribución:** Dónde cae tu predicción vs datos históricos
- **Gauge de Tiempo:** Representación visual de la velocidad de entrega
- **Factores de Impacto:** Qué está afectando más el tiempo de entrega

---

### Insights Potenciados por LLM

![Análisis LLM](images/streamlit_p5.png)

Llama 3.3 70B de Groq proporciona:
- Análisis contextual de la situación
- Recomendaciones accionables para operaciones
- Templates de comunicación al cliente

---

### REST API

![API Swagger UI](images/Food_API.png)

**Documentación interactiva en `/docs`**

#### Ejemplo de Llamada a la API

![Ejemplo API](images/eJEMPLO_EJECUCION_DE_LA_API.png)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Distance_km": 10.5,
    "Weather": "Rainy",
    "Traffic_Level": "High",
    "Time_of_Day": "Evening",
    "Vehicle_Type": "Car",
    "Preparation_Time_min": 20,
    "Courier_Experience_yrs": 3.5
  }'
```

**Respuesta:**
```json
{
  "predicted_delivery_time_minutes": 67.3,
  "confidence_level": "high",
  "model_version": "v1.0"
}
```

---

## Inicio Rápido

### Prerequisitos

- Python 3.10+
- pip

### Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/jmeza-data/food-delivery-time-prediction.git
cd food-delivery-time-prediction
```

2. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

3. **Configurar variable de entorno** (para integración LLM en Streamlit)
```bash
# Windows
set GROQ_API_KEY=tu_clave_aqui

# Linux/Mac
export GROQ_API_KEY=tu_clave_aqui
```

### Ejecutar el Pipeline ML

Entrenar los modelos desde cero:
```bash
python model_pipeline/run_pipeline.py
```

Esto:
- Carga y preprocesa los datos
- Ingenieriza 32 features
- Entrena 3 modelos (Random Forest, LightGBM, XGBoost)
- Guarda el mejor modelo en `models/`
- Genera reporte de comparación en `reports/`

### Ejecutar la API
```bash
cd api
python main.py
```

La API estará disponible en:
- **Swagger UI:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

### Ejecutar el Dashboard
```bash
streamlit run streamlit_app.py
```

El dashboard se abrirá en: http://localhost:8501

---

## Estructura del Proyecto
```
food-delivery-time-prediction/
│
├── model_pipeline/              # Módulos del pipeline ML
│   ├── config.py                # Configuración
│   ├── data_loader.py           # Carga de datos
│   ├── preprocessor.py          # Limpieza y encoding
│   ├── feature_engineer.py      # 32 features ingenierizadas
│   ├── model_trainer.py         # Entrenamiento y comparación
│   ├── predictor.py             # Interface de predicción
│   └── run_pipeline.py          # Script principal
│
├── api/                         # REST API
│   ├── main.py                  # Aplicación FastAPI
│   └── README.md                # Documentación de la API
│
├── models/                      # Modelos entrenados
│   ├── delivery_time_model_v1.0.pkl
│   ├── preprocessor_v1.0.pkl
│   └── feature_engineer_v1.0.pkl
│
├── data/                        # Dataset
│   └── Food_Delivery_Times.csv
│
├── reports/                     # Análisis y documentación
│   ├── model_comparison_*.csv
│   └── strategic_reflections.md
│
├── images/                      # Assets del README
│
├── notebooks/                   # Análisis exploratorio
│   └── 01_EDA.ipynb
│
├── streamlit_app.py             # Dashboard interactivo
├── requirements.txt             # Dependencias Python
└── README.md                    # Este archivo
```

---

## Uso de la API

### Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Inicio - Información de la API |
| GET | `/health` | Health check (estado del modelo) |
| POST | `/predict` | Predecir tiempo de entrega |
| GET | `/model-info` | Metadata y métricas del modelo |
| GET | `/examples` | Ejemplos de requests |

### Ejemplo en Python
```python
import requests

# Preparar datos de la orden
order = {
    "Distance_km": 10.5,
    "Weather": "Rainy",
    "Traffic_Level": "High",
    "Time_of_Day": "Evening",
    "Vehicle_Type": "Car",
    "Preparation_Time_min": 20,
    "Courier_Experience_yrs": 3.5
}

# Hacer predicción
response = requests.post(
    "http://localhost:8000/predict",
    json=order
)

result = response.json()
print(f"Tiempo estimado de entrega: {result['predicted_delivery_time_minutes']:.1f} min")
```

### Valores de Entrada Válidos

**Categóricos:**
- `Weather`: Clear, Cloudy, Rainy, Snowy, Foggy, Windy
- `Traffic_Level`: Low, Medium, High
- `Time_of_Day`: Morning, Afternoon, Evening, Night
- `Vehicle_Type`: Bike, Scooter, Car

**Numéricos:**
- `Distance_km`: 0.1 - 50.0
- `Preparation_Time_min`: 5 - 60
- `Courier_Experience_yrs`: 0.0 - 15.0

---

## Insights Estratégicos

### Decisiones Clave

**Feature Engineering sobre Modelos Complejos**

Creé `Estimated_Base_Time = (Distance × 2) + Prep_Time`, que se convirtió en la feature más importante (importance = 0.232). El conocimiento de dominio simple supera a la complejidad.

**Enfoque API + Dashboard**

API para integración de sistemas (apps móviles, herramientas internas) y dashboard para equipo de operaciones y demos. Cubre necesidades técnicas y de negocio.

**Integración LLM para Contexto**

Las predicciones son números, pero las decisiones necesitan contexto. El LLM proporciona recomendaciones accionables y mejora la comunicación con clientes.

### Desafíos Resueltos

- **Subestimación en días lluviosos:** Propuse features de interacción y mejoras en granularidad de datos
- **Transferibilidad entre ciudades:** Diseñé enfoque de 3 fases con transfer learning
- **Preparación para producción:** Documenté arquitectura completa de deployment (Kubernetes, monitoreo, CI/CD)

Ver análisis estratégico completo en: [`reports/strategic_reflections.md`](reports/strategic_reflections.md)

---

## Sobre el Autor

**Jhoan Sebastian Meza Garcia**  
Data Scientist | ML Engineer

Apasionado por convertir datos en soluciones de impacto real. Este proyecto demuestra capacidades end-to-end en machine learning, desde exploración hasta deployment en producción.

- 💼 [LinkedIn](https://www.linkedin.com/in/jhoan-sebastian-meza-garcia-12228b329/)
- 🐱 [GitHub](https://github.com/jmeza-data)

### Otros Proyectos

Explora más de mi trabajo:

**Repositorios Destacados:**
- [**Regresión IPM Continuo a Nivel de Hogar**](https://github.com/jmeza-data) - Modelo XGBoost para predecir IPM usando variables socioeconómicas
- [**Análisis SHAP para Interpretabilidad**](https://github.com/jmeza-data) - Implementación de técnicas de explicabilidad en modelos de ML
- [**Más proyectos...**](https://github.com/jmeza-data?tab=repositories)

---

## Licencia

Este proyecto es parte de una evaluación técnica. Para fines educativos y de portafolio.

---

## Agradecimientos

- **Dataset:** Kaggle - Food Delivery Time Prediction
- **LLM:** Groq (Llama 3.3 70B) para análisis inteligente
- **Frameworks:** Equipos de FastAPI, Streamlit, scikit-learn
- **Inspiración:** Evidently AI para diseño del dashboard

---

<div align="center">

**⭐ Si este proyecto te pareció interesante, dale una estrella**

Desarrollado con dedicación por [Jhoan Meza](https://github.com/jmeza-data)

</div>
