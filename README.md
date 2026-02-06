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

Este proyecto aborda un desafío crítico en la logística urbana: predecir con precisión los tiempos de entrega de pedidos de comida. Mediante un sistema de machine learning de extremo a extremo, complementado con análisis inteligente apoyado por un LLM en la etapa de despliegue, el modelo genera predicciones en tiempo real con un R² de 0.802.
Además de estimar tiempos de entrega, la solución permite identificar las variables que más influyen en los retrasos y aporta insights relevantes tanto para la operación logística como para la experiencia del cliente.  

---

## Características Principales

**Machine Learning**
- Modelo Random Forest con R²=0.802
- 32 features 
- Error promedio <10 minutos
- Maneja 7 variables de entrada

**API REST**
- FastAPI con documentación Swagger
- Validación y health checks
- Request/response en JSON

**Dashboard Interactivo**
- Predicciones en tiempo real
- Análisis visual 
- Métricas de desempeño del modelo
- Análisis de factores de impacto

**Integración con LLM**
- Insights potenciados por Groq
- Recomendaciones contextualizadas
- Templates de comunicación al cliente
- Modelo usado Llama 3.3 70B, lo use porque los tokens son gratis y genera un plus en el analisis.

---

## Impacto en el Negocio

| Métrica | Resultado |
|---------|-----------|
| **R²** | 0.802 |
| **MAE** | ~9.4 minutos |
| **Tiempo de respuesta API** | <100 ms |
| **Features creadas** | 30+ variables derivadas |
| **Modelos evaluados** | LightGBM, XGBoost, Random Forest |
| **Modelo final** | Random Forest |

**Innovación de mi parte:** Hice una variable derivada que combina la distancia y el tiempo de preparación (Estimated_Base_Time = distancia × 2 + tiempo_prep) la cual se consolidó como uno de los predictores más influyentes del modelo.



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
Random Forest fue elegido como modelo final al obtener el mejor desempeño general en las métricas clave. Presentó el menor RMSE (9.42 min) y el mayor R² (0.802), superando consistentemente a LightGBM y XGBoost. Además de su precisión, mostró buena estabilidad, capacidad para capturar relaciones no lineales y una interpretación clara mediante la importancia de variables, lo que lo hace adecuado para un entorno operativo.

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
- Manejo integral de errores
- Documentación de API 
- Control de versiones 

---

## Demo en Vivo

### Dashboard Streamlit

**Pruébalo:** [https://food-delivery-time-prediction-z3c8fxrjyqn3nbwe784grg.streamlit.app](https://food-delivery-time-prediction-z3c8fxrjyqn3nbwe784grg.streamlit.app)
(Puede que se demore 1 minutito, dejalo cargando quedo muy bonito para que revises)


![Dashboard Principal](images/Opera.png)

**Características:**
- Sliders interactivos para parámetros de entrega
- Predicciones en tiempo real con niveles de confianza
- Análisis visual: gráficos de distribución, gauge y factores de impacto
- Recomendaciones potenciadas por IA

---

### Análisis Visual

![Análisis Visual](images/Streamlit_p2.png)

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

![Ejemplo API](images/ejemplo.png)
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

## Insights Estratégicos

### Decisiones Clave

**Feature Engineering sobre Modelos Complejos**

Diseñe la variable `Estimated_Base_Time = (Distance × 2) + Prep_Time`, que se convirtió en la feature más importante (importance = 0.232)

**Enfoque API + Dashboard**

Implemente una API para integración de sistemas y dashboard para equipo de operaciones y demos la cual cubre necesidades técnicas y de negocio.

**Integración LLM para Contexto**

Las predicciones son números pero las decisiones necesitan contexto, por eso el LLM proporciona recomendaciones accionables y mejora la comunicación con clientes.

### Desafíos Resueltos

- **Subestimación en días lluviosos:** Propuse features de interacción y mejoras en granularidad de datos
- **Transferibilidad entre ciudades:** Diseñé enfoque de 3 fases con transfer learning
- **Preparación para producción:** Documenté arquitectura completa de deployment

Ver análisis estratégico completo en: [`reports/strategic_reflections.md`](reports/strategic_reflections.md)


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

## Sobre el Autor

**Jhoan Sebastian Meza Garcia**  
Estudiante de economia | Universidad Nacional de Colombia

- 💼 [LinkedIn](https://www.linkedin.com/in/jhoan-sebastian-meza-garcia-12228b329/)
- 🐱 [GitHub](https://github.com/jmeza-data)  <- Tengo mas proyectos si quirees hechar un vistaso.

### Otros Proyectos

Explora más de mi trabajo:

**Repositorios Destacados:**
- [**Regresión IPM Continuo a Nivel de Hogar**](https://github.com/jmeza-data) - Modelo XGBoost para predecir IPM usando variables socioeconómicas
- [**Análisis SHAP para Interpretabilidad**](https://github.com/jmeza-data) - Implementación de técnicas de explicabilidad en modelos de ML
- [**Más proyectos...**](https://github.com/jmeza-data?tab=repositories)

yecto es parte de una evaluación técnica. Para fines educativos y de portafolio.


---

<div align="center">

**⭐ Si este proyecto te pareció interesante, dale una estrella**

Desarrollado con dedicación por [Jhoan Meza](https://github.com/jmeza-data)

</div>
