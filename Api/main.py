"""
API REST para predicción de tiempo de entrega de comida.
Construida con FastAPI.
"""

# IMPORTS
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import sys
from pathlib import Path

# Agregar el directorio padre al path para importar model_pipeline
sys.path.append(str(Path(__file__).parent.parent))

from model_pipeline import Predictor
from model_pipeline import config

# CONFIGURACIÓN DE LA API

# Crear la aplicación FastAPI
app = FastAPI(
    title="Food Delivery Time Prediction API",
    description="API para predecir el tiempo de entrega de pedidos de comida",
    version="1.0.0"
)

# Cargar el modelo al iniciar la API (solo una vez)
print("\n" + "="*70)
print("🔄 CARGANDO PIPELINE COMPLETA...")
print("="*70)

predictor = Predictor()
try:
    # Cargar modelo, preprocessor y feature engineer
    model_path = config.MODEL_DIR / config.MODEL_NAME
    preprocessor_path = config.MODEL_DIR / config.PREPROCESSOR_NAME
    feature_engineer_path = config.MODEL_DIR / config.FEATURE_ENGINEER_NAME
    
    print(f"\n Rutas:")
    print(f"   Modelo: {model_path}")
    print(f"   Preprocessor: {preprocessor_path}")
    print(f"   Feature Engineer: {feature_engineer_path}")
    
    predictor.load_pipeline(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        feature_engineer_path=feature_engineer_path,
        verbose=True
    )
    
    print("\n PIPELINE COMPLETA CARGADA EXITOSAMENTE")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n ERROR CARGANDO PIPELINE: {e}")
    print("="*70 + "\n")
    import traceback
    traceback.print_exc()
    predictor = None


# MODELOS DE DATOS 

class DeliveryOrder(BaseModel):
    """
    Esquema de datos para una orden de delivery.
    Define qué campos son necesarios y sus tipos.
    """
    Distance_km: float = Field(..., ge=0, le=100, description="Distancia en kilómetros")
    Weather: str = Field(..., description="Condición climática: Clear, Rainy, Snowy, Foggy, Windy")
    Traffic_Level: str = Field(..., description="Nivel de tráfico: Low, Medium, High")
    Time_of_Day: str = Field(..., description="Momento del día: Morning, Afternoon, Evening, Night")
    Vehicle_Type: str = Field(..., description="Tipo de vehículo: Bike, Scooter, Car")
    Preparation_Time_min: int = Field(..., ge=1, le=120, description="Tiempo de preparación en minutos")
    Courier_Experience_yrs: float = Field(..., ge=0, le=20, description="Años de experiencia del courier")
    
    class Config:
        # Ejemplo de datos válidos (se muestra en la documentación)
        schema_extra = {
            "example": {
                "Distance_km": 10.5,
                "Weather": "Rainy",
                "Traffic_Level": "High",
                "Time_of_Day": "Evening",
                "Vehicle_Type": "Car",
                "Preparation_Time_min": 20,
                "Courier_Experience_yrs": 3.5
            }
        }


class PredictionResponse(BaseModel):
    """
    Esquema de la respuesta de predicción.
    """
    predicted_delivery_time_minutes: float = Field(..., description="Tiempo estimado de entrega en minutos")
    confidence_level: str = Field(..., description="Nivel de confianza: low, medium, high")
    model_version: str = Field(..., description="Versión del modelo utilizado")
    input_data: dict = Field(..., description="Datos de entrada recibidos")


class HealthResponse(BaseModel):
    """
    Esquema de la respuesta de health check.
    """
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    feature_engineer_loaded: bool
    message: str


# Rutas de la API

@app.get("/")
def home():
    """
    Endpoint raíz - Información general de la API.
    
    Acceso: GET http://localhost:8000/
    """
    return {
        "message": "Food Delivery Time Prediction API",
        "status": "active",
        "version": "1.0.0",
        "description": "API para predecir tiempos de entrega usando Machine Learning",
        "endpoints": {
            "home": "/",
            "health": "/health",
            "predict": "/predict (POST)",
            "model_info": "/model-info",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "documentation": "Visit /docs for interactive API documentation"
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Endpoint de salud - Verifica que la API y el modelo estén funcionando.
    
    Acceso: GET http://localhost:8000/health
    """
    if predictor is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            preprocessor_loaded=False,
            feature_engineer_loaded=False,
            message="Predictor not initialized"
        )
    
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "unhealthy",
        model_loaded=predictor.model is not None,
        preprocessor_loaded=predictor.preprocessor is not None,
        feature_engineer_loaded=predictor.feature_engineer is not None,
        message="API is running and pipeline is ready" if predictor.is_loaded else "Pipeline not fully loaded"
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_delivery_time(order: DeliveryOrder):
    """
    Endpoint de predicción - Predice el tiempo de entrega.
    
    Acceso: POST http://localhost:8000/predict
    
    Body (JSON):
    {
        "Distance_km": 10.5,
        "Weather": "Rainy",
        "Traffic_Level": "High",
        "Time_of_Day": "Evening",
        "Vehicle_Type": "Car",
        "Preparation_Time_min": 20,
        "Courier_Experience_yrs": 3.5
    }
    
    Returns:
    {
        "predicted_delivery_time_minutes": 45.23,
        "confidence_level": "high",
        "model_version": "v1.0",
        "input_data": {...}
    }
    """
    # Verificar que el predictor esté cargado
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Pipeline is not loaded. Please check server logs and restart."
        )
    
    # Verificar que todos los componentes estén cargados
    if predictor.preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Preprocessor is not loaded. Pipeline incomplete."
        )
    
    if predictor.feature_engineer is None:
        raise HTTPException(
            status_code=503,
            detail="Feature engineer is not loaded. Pipeline incomplete."
        )
    
    try:
        # Convertir los datos a diccionario
        order_dict = order.dict()
        
        # Validar valores categóricos
        valid_weather = ["Clear", "Cloudy", "Rainy", "Snowy", "Foggy", "Windy"]
        valid_traffic = ["Low", "Medium", "High"]
        valid_time = ["Morning", "Afternoon", "Evening", "Night"]
        valid_vehicle = ["Bike", "Scooter", "Car"]
        
        if order_dict["Weather"] not in valid_weather:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid Weather value. Must be one of: {valid_weather}"
            )
        
        if order_dict["Traffic_Level"] not in valid_traffic:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid Traffic_Level value. Must be one of: {valid_traffic}"
            )
        
        if order_dict["Time_of_Day"] not in valid_time:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid Time_of_Day value. Must be one of: {valid_time}"
            )
        
        if order_dict["Vehicle_Type"] not in valid_vehicle:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid Vehicle_Type value. Must be one of: {valid_vehicle}"
            )
        
        # Hacer la predicción (el predictor maneja preprocessing y feature engineering)
        predicted_time = predictor.predict_single(order_dict)
        
        # Determinar nivel de confianza basado en el rango esperado
        if 15 <= predicted_time <= 70:
            confidence = "high"
        elif 10 <= predicted_time <= 90:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Retornar la respuesta
        return PredictionResponse(
            predicted_delivery_time_minutes=round(predicted_time, 2),
            confidence_level=confidence,
            model_version="v1.0",
            input_data=order_dict
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log del error completo
        import traceback
        error_detail = traceback.format_exc()
        print(f"\n ERROR EN PREDICCIÓN:\n{error_detail}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


@app.get("/model-info")
def get_model_info():
    """
    Endpoint de información del modelo.
    
    Acceso: GET http://localhost:8000/model-info
    
    Retorna información detallada sobre el modelo cargado,
    incluyendo métricas de performance y metadata.
    """
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Pipeline is not loaded"
        )
    
    try:
        info = predictor.get_model_info()
        
        # Agregar información adicional
        info['api_version'] = "1.0.0"
        info['pipeline_components'] = {
            'preprocessor': predictor.preprocessor is not None,
            'feature_engineer': predictor.feature_engineer is not None,
            'model': predictor.model is not None
        }
        
        return info
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model info: {str(e)}"
        )


@app.get("/examples")
def get_examples():
    """
    Endpoint que retorna ejemplos de órdenes para probar la API.
    
    Acceso: GET http://localhost:8000/examples
    """
    return {
        "examples": [
            {
                "name": "Orden fácil - Distancia corta, buen clima",
                "data": {
                    "Distance_km": 3.5,
                    "Weather": "Clear",
                    "Traffic_Level": "Low",
                    "Time_of_Day": "Morning",
                    "Vehicle_Type": "Bike",
                    "Preparation_Time_min": 10,
                    "Courier_Experience_yrs": 5.0
                },
                "expected_time": "~20-30 minutos"
            },
            {
                "name": "Orden normal - Distancia media",
                "data": {
                    "Distance_km": 10.5,
                    "Weather": "Clear",
                    "Traffic_Level": "Medium",
                    "Time_of_Day": "Afternoon",
                    "Vehicle_Type": "Scooter",
                    "Preparation_Time_min": 15,
                    "Courier_Experience_yrs": 3.0
                },
                "expected_time": "~35-45 minutos"
            },
            {
                "name": "Orden difícil - Distancia larga, mal clima",
                "data": {
                    "Distance_km": 18.5,
                    "Weather": "Snowy",
                    "Traffic_Level": "High",
                    "Time_of_Day": "Evening",
                    "Vehicle_Type": "Car",
                    "Preparation_Time_min": 30,
                    "Courier_Experience_yrs": 1.0
                },
                "expected_time": "~60-80 minutos"
            }
        ],
        "valid_values": {
            "Weather": ["Clear", "Cloudy", "Rainy", "Snowy", "Foggy", "Windy"],
            "Traffic_Level": ["Low", "Medium", "High"],
            "Time_of_Day": ["Morning", "Afternoon", "Evening", "Night"],
            "Vehicle_Type": ["Bike", "Scooter", "Car"]
        }
    }


# PUNTO DE ENTRADA

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print(" STARTING FOOD DELIVERY TIME PREDICTION API")
    print("="*70)
    print("\n Documentation available at: http://localhost:8000/docs")
    print(" Home: http://localhost:8000/")
    print(" Health check: http://localhost:8000/health")
    print(" Examples: http://localhost:8000/examples")
    print("\n" + "="*70 + "\n")
    
    # Iniciar el servidor
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Recarga automática al modificar el código
    )