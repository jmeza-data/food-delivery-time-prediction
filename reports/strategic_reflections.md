# Reflexiones Estratégicas - Predicción de Tiempos de Entrega 

## 1. Falla del Modelo en Días Lluviosos

El modelo subestima los tiempos de entrega cuando llueve. Un pedido que debería tardar 50 minutos lo predice en 40. Como solucion del problema desde defirentes frentes tanto el modelo como de las expectativas del negocio. No se trata de bajar las expectativas del negocio sino de que el modelo capture mejor el impacto de la lluvia.

Mejoras desde el modelo:

Primero, mejoraría las features de interacción. Para mi modelo hice `Distance_km_x_Weather_Rainy` la cual multiplica la distancia por si está lloviendo, capturando que los viajes largos se complican más bajo lluvia, asi mismo agregaría `Rain_Traffic_Combined`, la cual es una variable que detecta cuando llueve Y hay tráfico alto simultáneamente, donde esta combinación es peor que la suma de ambos factores por separado porque los carros van más lento y hay más congestión.


Mejoras desde los datos:

El problema de raíz puede estar en la granularidad. Actualmente solo tengo "Rainy" como categoría binaria. Necesito distinguir entre llovizna y aguacero, agregar visibilidad y velocidad del viento. Puede que parezca un poco costoso reunir todos estos datos pero se podria aplicar una API donde obtengamos esta información de lugares especificos.

Asi mismo, no bajaría las expectativas, pero cambiaría cómo comunicamos la información al cliente final, en lugar de "45 minutos" exactos, mostraría un rango "45-55 minutos" con un aviso de condiciones climáticas, **HICE UN EJEMPLO EN FORMA DE LLM EN STREAMLIT**. 


## 2. Transferibilidad: De Mumbai a São Paulo

El desafio en este caso es que tengo un modelo que funciona bien en Mumbai y necesito deployarlo en São Paulo. Las ciudades son completamente diferentes, asi mismo, Mumbai tiene tráfico caótico pero predecible y São Paulo tiene picos distintos y es más dispersa geográficamente, asi mismo, el clima también cambia dastricamente.Por ultimo, la flota vehicular tampoco es igual debido a sus diferencias estructurales. 

Como primera solución hariamos una adaptación por 3 fases, en la primera fase haria una Validación Controladam en donde no  deployaría a ciegas. Primero colectaría 2-3 semanas de datos reales en São Paulo, usaría el modelo de Mumbai solo como baseline interno, no para mostrar al cliente.

Como segunda fase, no entrenaría desde cero porque el modelo de Mumbai ya sabe conceptos universales sobre entregas. Identificaría qué features son universales como  lo son la distancia y experiencia. Ademas cuales  son locales como lo puede ser el trafico en cada zona. 

Por ultima fase, como una solución a largo plazo sería un modelo de dos capas. La primera capa captura patrones universales de entregas, entrenada con datos de todas las ciudades. La segunda capa aprende los ajustes específicos de São Paulo. Asi mismo, implementaría monitoreo continuo con alertas automáticas si el RMSE de São Paulo supera al de Mumbai por más de 30%. 

## 3. Uso de Herramientas GenAI

Utilice Claude en algunos partes que me ayudaron a potenciar mi trabajo, presento una lista de como use la IA:

**Estructura de código (20%):** Generé templates iniciales para las clases principales (Preprocessor, FeatureEngineer, ModelTrainer). Revisé cada una manualmente y ajusté a las necesidades específicas del proyecto.

**Feature engineering (10%):** Pedí ideas sobre features de interacción útiles. Implementé solo las que tenían sentido de dominio y evalué su importancia después del entrenamiento.

**Debugging (30%):** Resolví errores de imports relativos, problemas con f-strings en Streamlit, y bugs varios. Probé cada solución y entendí el problema antes de aplicarla.

**Steeamlit:** En el momento, no se como programar una pagina web, entonces le pedi a la IA, que me ayudara a estructurar el codigo, yo le di las ideas como quisiera que se viera, como se deberian ver los reportes y agregue el uso de LLM para una salida mas profesional.

No use la IA en el análisis exploratorio fue completamente manual. Necesitaba entender los datos a profundidad, ademas me gusta mucho graficar. La selección de modelos vino a partir de varias pruebas que hice y metricas de desempeño.

Lo más importante: el modelo logró R²=0.802, superando benchmarks académicos de 0.70-0.76. Esto valida que las decisiones técnicas fueron correctas, no solo código que "correr".

## 4. Insight del Que Estoy Orgulloso

La feature `Estimated_Base_Time` terminó siendo la más importante del modelo (importance = 0.232) la cual no pense desde un inicio. En este caso se define como `(Distance_km * 2) + Preparation_Time_min`. Básicamente, dos minutos por kilómetro más el tiempo que tarda la cocina. Le da al modelo un punto de partida realista para que solo tenga que ajustar según las condiciones específicas (tráfico, clima, experiencia del courier) en lugar de aprender todo desde cero. 

También estoy orgulloso de haber construido toda la infraestructura de deployment con poca experiencia previa, nunca había creado una API REST antes, pero implementé una completa con FastAPI, donde aprende que era un endpoints de predicción, health checks, validación de inputs, y manejo de errores. Asi mismo, aprendí sobre arquitectura de APIs, autenticación, y rate limiting en el proceso. 

Por ultimo desarrollé una aplicación en Streamli, donde integré visualizaciones interactivas, métricas en tiempo real, y conecté un LLM (Groq) para análisis inteligente de las predicciones. Todo esto lo aprendí haciendo, investigando documentación y resolviendo problemas sobre la marcha.

Lo que más valoro es que no me quedé solo en entrenar un modelo. Construí un sistema completo que alguien puede usar de verdad.

## 5. Deployment a Producción

Para llevar el modelo a producción diseñaría una arquitectura en capas que permita escalabilidad y mantenimiento fácil, la idea es separar responsabilidades: frontend para usuarios, API para predicciones, y servicios de soporte para monitoreo y almacenamiento.

### Arquitectura Propuesta

El flujo sería así: las aplicaciones (móvil o dashboard web) se comunican con un API Gateway que maneja autenticación y rate limiting. Este gateway redirige las peticiones al servicio de predicción construido en FastAPI, el cual corre en Kubernetes con múltiples réplicas para alta disponibilidad. Los modelos se guardan en S3, las features pre-computadas en Redis para acceso rápido, y todos los logs van a CloudWatch para monitoreo.

El servicio de predicción tendría 3 réplicas mínimas corriendo simultáneamente con auto-scaling basado en uso de CPU. Cada 30 segundos se ejecutan health checks para asegurar que todo funciona correctamente.


**API Gateway y Seguridad**

Implementaría diferentes API keys según el tipo de cliente: una para la app móvil, otra para el dashboard web, y otra para servicios internos. Cada cliente tendría un límite de 100 requests por minuto para evitar abuso. FastAPI tiene decoradores que hacen esto bastante simple de implementar.

**Versionado y Testing A/B**

No puedo simplemente reemplazar el modelo actual con uno nuevo sin estar seguro que funciona mejor. Tendría un ModelManager que decide qué versión usar según el user_id. Por ejemplo, 90% del tráfico va al modelo actual probado, y 10% al modelo candidato nuevo. Cada predicción se registra con su versión, y después de una semana comparo las métricas. Si el nuevo modelo es mejor, lo promociono para todos.

**Feature Store**

Muchas features se pueden pre-calcular y reutilizar. Por ejemplo, la velocidad promedio de un courier o el tráfico histórico de una zona no cambian cada minuto. Guardaría estas features en Redis con un tiempo de vida de 1-2 horas. Esto reduce la latencia de 200ms a 50ms porque no tengo que recalcular todo en cada request.

**Sistema de Logs**

Cada vez que el modelo hace una predicción, guardo en S3 toda la información: qué datos entraron, qué predijo el modelo, qué versión se usó, y cuánto tardó. Cuando la entrega se completa, actualizo ese log con el tiempo real que tomó. Esto me permite calcular qué tan bien está funcionando el modelo en producción y detectar si está empeorando.

**Detección de Drift**

Los datos cambian con el tiempo. Puede que en verano haya más tráfico, o que una nueva zona se vuelva popular. Programaría un job semanal en Airflow que compara la distribución de features de la última semana contra los datos de entrenamiento usando Evidently. Si el drift score supera 0.5, dispara un re-entrenamiento automático del modelo.

También monitorearía métricas de negocio: si el RMSE sube 20% o las quejas de clientes aumentan, algo anda mal y mando una alerta.

**Containerización**

Todo el código iría en un contenedor Docker. El Dockerfile instalaría las dependencias, copiaría el código, y expondría el puerto 8000. Cada 30 segundos Docker verifica que el servicio esté vivo. La imagen se subiría a AWS ECR.

En Kubernetes definiría un deployment con mínimo 3 réplicas y máximo 10. Si el uso de CPU supera 70%, automáticamente crea más réplicas para manejar la carga.

### Pipeline de Deploy Automático

Usaría GitHub Actions para automatizar todo. Cuando hago push a la rama main, GitHub corre los tests automáticamente. Si pasan, construye la imagen Docker, la sube al registro con el hash del commit como tag, y actualiza el deployment en Kubernetes sin downtime.

Lo importante es que si las métricas de la nueva versión empeoran en los primeros 10 minutos (latencia alta, más errores), hay un rollback automático a la versión anterior.

### Monitoreo Continuo

Tendría un dashboard en Grafana mostrando en tiempo real: latencia (percentiles 50, 95 y 99), número de requests por segundo, tasa de error, y RMSE calculado cuando las entregas se completan.

Las alertas irían a Slack si: latencia > 500ms, error rate > 1%, RMSE > 12 min, o se detecta drift en los datos.

### Realidad de Costos

En AWS con 3 instancias t3.medium, Redis en ElastiCache, y S3 para logs, el costo sería aproximadamente $330 por mes para 100,000 requests al día. Si escala a 1 millón de requests diarios, necesitaría ~10 pods y costaría unos $800 mensuales. Sigue siendo económico comparado con el valor que genera tener predicciones precisas.

### Plan de Implementación

Las primeras dos semanas las dedicaría a dockerizar la aplicación, configurar Kubernetes, y tener logging básico funcionando. En las semanas 3-4 implementaría el feature store y el sistema de versionado. Las semanas 5-6 serían para drift detection y alertas. Las últimas dos semanas completaría el CI/CD y automatización de retraining.

En dos meses tendría un sistema robusto listo para producción real. Lo más importante es que cada componente se puede mejorar incrementalmente sin tener que rehacer todo.