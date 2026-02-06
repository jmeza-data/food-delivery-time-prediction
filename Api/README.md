# Food Delivery Time Prediction API

API REST para predicción de tiempos de entrega usando Machine Learning.

## Características

- Predicciones en tiempo real con Random Forest (R² = 0.802)
- Validación automática de inputs
- Health checks integrados
- Documentación interactiva 
- Manejo de errores robusto

## Instalación
```bash
pip install fastapi uvicorn
```

## Ejecución
```bash
cd api
python main.py
```

La API estará disponible en `http://localhost:8000`

## Endpoints

### 1. Health Check
```
GET /health
```
Verifica que el modelo esté cargado y funcionando.

### 2. Predicción
```
POST /predict
```

**Body:**
```json
{
  "Distance_km": 10.5,
  "Weather": "Rainy",
  "Traffic_Level": "High",
  "Time_of_Day": "Evening",
  "Vehicle_Type": "Car",
  "Preparation_Time_min": 20,
  "Courier_Experience_yrs": 3.5
}
```

**Response:**
```json
{
  "predicted_delivery_time_minutes": 52.34,
  "confidence_level": "high",
  "model_version": "v1.0",
  "input_data": {...}
}
```

### 3. Información del Modelo
```
GET /model-info
```
Retorna métricas y metadata del modelo.

### 4. Ejemplos
```
GET /examples
```
Muestra casos de uso con diferentes escenarios.

## Valores Válidos

**Weather:** Clear, Cloudy, Rainy, Snowy, Foggy, Windy  
**Traffic_Level:** Low, Medium, High  
**Time_of_Day:** Morning, Afternoon, Evening, Night  
**Vehicle_Type:** Bike, Scooter, Car

**Rangos:**
- Distance_km: 0.1 - 50.0
- Preparation_Time_min: 5 - 60
- Courier_Experience_yrs: 0.0 - 15.0

## Documentación Interactiva

Visita `http://localhost:8000/docs` para probar la API directamente desde el navegador.

## Uso con Python
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "Distance_km": 10.5,
        "Weather": "Rainy",
        "Traffic_Level": "High",
        "Time_of_Day": "Evening",
        "Vehicle_Type": "Car",
        "Preparation_Time_min": 20,
        "Courier_Experience_yrs": 3.5
    }
)

result = response.json()
print(f"Tiempo estimado: {result['predicted_delivery_time_minutes']} minutos")
```

## Estructura del Proyecto
```
api/
├── main.py           # API implementation
└── README.md         # Esta documentación

model_pipeline/
├── predictor.py      # Predictor class
├── config.py         # Configuration
└── ...

models/
├── delivery_time_model_v1.0.pkl
├── preprocessor_v1.0.pkl
└── feature_engineer_v1.0.pkl
```

## Notas

- El modelo se carga automáticamente al iniciar
- Las predicciones incluyen nivel de confianza
- Los errores retornan códigos HTTP apropiados (422, 500, 503)
- La API usa la pipeline completa: preprocessing + feature engineering + predicción

