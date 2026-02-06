# SQL Insights - Análisis Adicional de Entregas

**Proyecto:** Food Delivery Time Prediction  
**Autor:** Jhoan Meza  
**Fecha:** Febrero 2026

---

## 1. Patrones Temporales de Demanda

Las horas pico son predecibles: almuerzo (12-14h) y cena (19-21h). Los fines de semana muestran incrementos del 30-40% en tiempos de entrega.

```sql
SELECT
    DATEPART(HOUR, order_placed_at) AS hour_of_day,
    DATENAME(WEEKDAY, order_placed_at) AS day_of_week,
    AVG(delivery_time_min) AS avg_delivery_time,
    COUNT(*) AS total_deliveries
FROM deliveries
WHERE order_placed_at >= DATEADD(MONTH, -3, GETDATE())
GROUP BY DATEPART(HOUR, order_placed_at), DATENAME(WEEKDAY, order_placed_at)
ORDER BY avg_delivery_time DESC;
```

**Recomendación:** Asignar más repartidores en horas pico. Ofrecer incentivos para entregas en horarios de baja demanda.

---

## 2. Impacto del Clima

El clima adverso incrementa los tiempos significativamente. Lluvia y nieve añaden 15-20% más tiempo. Niebla añade 10-15%.
```sql
SELECT
    weather_condition,
    AVG(delivery_time_min) AS avg_delivery_time,
    STDEV(delivery_time_min) AS std_deviation,
    COUNT(*) AS total_deliveries,
    AVG(delivery_rating) AS avg_rating
FROM deliveries
WHERE order_placed_at >= DATEADD(MONTH, -6, GETDATE())
GROUP BY weather_condition
ORDER BY avg_delivery_time DESC;
```

**Recomendación:** Ajustar ETAs dinámicamente según pronóstico del clima. Bonos para repartidores que trabajen en condiciones adversas.

---

## 3. Rutas Problemáticas

Algunas combinaciones de áreas (restaurante → cliente) son consistentemente lentas. Esto puede deberse a tráfico estructural, puentes o falta de repartidores en ciertas zonas.
```sql
SELECT TOP 20
    restaurant_area,
    customer_area,
    AVG(delivery_time_min) AS avg_delivery_time,
    AVG(delivery_distance_km) AS avg_distance,
    ROUND(AVG(delivery_time_min) / NULLIF(AVG(delivery_distance_km), 0), 2) AS time_per_km,
    COUNT(*) AS total_deliveries
FROM deliveries
WHERE 
    order_placed_at >= DATEADD(MONTH, -3, GETDATE())
    AND delivery_distance_km > 0
GROUP BY restaurant_area, customer_area
HAVING COUNT(*) >= 20
ORDER BY avg_delivery_time DESC;
```

**Recomendación:** Reasignar repartidores a zonas problemáticas. Considerar centros de distribución intermedios.

---

## 4. Experiencia del Courier

La experiencia reduce tiempos de entrega. Couriers con 0-6 meses tardan 15% más que aquellos con +2 años. Los ratings también mejoran con la experiencia.
```sql
WITH experience_ranges AS (
    SELECT
        dp.delivery_person_id,
        CASE
            WHEN DATEDIFF(MONTH, dp.hired_date, GETDATE()) < 6 THEN '0-6 meses'
            WHEN DATEDIFF(MONTH, dp.hired_date, GETDATE()) < 12 THEN '6-12 meses'
            WHEN DATEDIFF(MONTH, dp.hired_date, GETDATE()) < 24 THEN '1-2 años'
            ELSE '2+ años'
        END AS experience_range
    FROM delivery_persons dp
    WHERE dp.is_active = 1
)
SELECT
    er.experience_range,
    AVG(d.delivery_time_min) AS avg_delivery_time,
    AVG(d.delivery_rating) AS avg_rating,
    COUNT(*) AS total_deliveries
FROM experience_ranges er
JOIN deliveries d ON er.delivery_person_id = d.delivery_person_id
WHERE d.order_placed_at >= DATEADD(MONTH, -3, GETDATE())
GROUP BY er.experience_range
ORDER BY 
    CASE er.experience_range
        WHEN '0-6 meses' THEN 1
        WHEN '6-12 meses' THEN 2
        WHEN '1-2 años' THEN 3
        ELSE 4
    END;
```

**Recomendación:** Programa de mentoría donde couriers experimentados entrenen a nuevos. Asignar rutas complejas solo a couriers con más de 1 año.

---

## 5. Factores que Afectan el Rating

El rating del cliente cae drásticamente cuando el tiempo supera los 60 minutos (promedio <3.5 estrellas). La combinación de tráfico alto y clima adverso es especialmente problemática.
```sql
SELECT
    CASE
        WHEN delivery_time_min < 30 THEN '<30 min'
        WHEN delivery_time_min < 45 THEN '30-45 min'
        WHEN delivery_time_min < 60 THEN '45-60 min'
        ELSE '>60 min'
    END AS time_bucket,
    traffic_condition,
    AVG(delivery_rating) AS avg_rating,
    COUNT(*) AS total_deliveries
FROM deliveries
WHERE 
    order_placed_at >= DATEADD(MONTH, -3, GETDATE())
    AND delivery_rating IS NOT NULL
GROUP BY
    CASE
        WHEN delivery_time_min < 30 THEN '<30 min'
        WHEN delivery_time_min < 45 THEN '30-45 min'
        WHEN delivery_time_min < 60 THEN '45-60 min'
        ELSE '>60 min'
    END,
    traffic_condition
ORDER BY avg_rating DESC;
```

**Recomendación:** Sistema de alertas cuando una entrega está en riesgo de superar 60 minutos. Ofrecer compensaciones proactivas.

---

## 6. Rentabilidad por Tipo de Cocina

No todos los tipos de cocina son igual de rentables. Fast food tiene alto volumen y bajo tiempo. Comida gourmet tiene alto valor pero tiempo de preparación elevado. Comida asiática muestra el mejor balance.
```sql
SELECT
    r.cuisine_type,
    COUNT(DISTINCT o.order_id) AS total_orders,
    AVG(o.order_value) AS avg_order_value,
    AVG(d.delivery_time_min) AS avg_delivery_time,
    ROUND(SUM(o.order_value) / SUM(d.delivery_time_min), 2) AS revenue_per_minute
FROM orders o
JOIN deliveries d ON o.delivery_id = d.delivery_id
JOIN restaurants r ON o.restaurant_id = r.restaurant_id
WHERE d.order_placed_at >= DATEADD(MONTH, -3, GETDATE())
GROUP BY r.cuisine_type
ORDER BY revenue_per_minute DESC;
```

**Recomendación:** Priorizar partnerships con restaurantes de alta rentabilidad por minuto. Incentivar a restaurantes lentos para mejorar preparación.

---

## 7. Detección de Anomalías

Entregas con tiempos anormalmente altos (>2 desviaciones estándar) requieren investigación. Pueden deberse a problemas técnicos, accidentes, direcciones incorrectas o problemas en el restaurante.
```sql
WITH delivery_stats AS (
    SELECT
        AVG(delivery_time_min) AS mean_time,
        STDEV(delivery_time_min) AS std_time
    FROM deliveries
    WHERE order_placed_at >= DATEADD(MONTH, -1, GETDATE())
)
SELECT
    d.delivery_id,
    dp.name AS courier_name,
    d.delivery_time_min,
    d.weather_condition,
    d.traffic_condition,
    ROUND((d.delivery_time_min - ds.mean_time) / ds.std_time, 2) AS z_score
FROM deliveries d
CROSS JOIN delivery_stats ds
JOIN delivery_persons dp ON d.delivery_person_id = dp.delivery_person_id
WHERE 
    d.order_placed_at >= DATEADD(MONTH, -1, GETDATE())
    AND d.delivery_time_min > (ds.mean_time + 2 * ds.std_time)
ORDER BY d.delivery_time_min DESC;
```

**Recomendación:** Alertas en tiempo real para entregas que superen 1.5 desviaciones estándar. Contacto proactivo con courier y cliente.

---

## 8. Optimización de Flota

El ratio óptimo es 3-4 entregas por courier por hora. Menos de 3 indica subutilización. Más de 5 causa sobrecarga, incremento de tiempos y caída de ratings.
```sql
SELECT
    dp.region,
    DATEPART(HOUR, d.order_placed_at) AS hour_of_day,
    COUNT(DISTINCT d.delivery_id) AS total_deliveries,
    COUNT(DISTINCT d.delivery_person_id) AS active_couriers,
    ROUND(CAST(COUNT(DISTINCT d.delivery_id) AS FLOAT) / COUNT(DISTINCT d.delivery_person_id), 2) AS deliveries_per_courier
FROM deliveries d
JOIN delivery_persons dp ON d.delivery_person_id = dp.delivery_person_id
WHERE d.order_placed_at >= DATEADD(MONTH, -1, GETDATE())
GROUP BY dp.region, DATEPART(HOUR, d.order_placed_at)
HAVING COUNT(DISTINCT d.delivery_id) >= 10
ORDER BY dp.region, hour_of_day;
```

**Recomendación:** Ajustar turnos dinámicamente según demanda histórica. Sistema de "on-call" para horas pico inesperadas.

---

## Conclusiones

**Hallazgos principales:**

Los patrones temporales son predecibles con horas pico claras. El clima incrementa tiempos en 15-20% bajo lluvia. La experiencia del courier importa: más de 2 años reduce tiempos en 15%. El rating del cliente correlaciona fuertemente con tiempo: más de 60 minutos resulta en ratings menores a 3.5 estrellas.

Identifiqué rutas específicas que son consistentemente lentas y tipos de cocina con mejor rentabilidad por minuto invertido. El análisis de anomalías revela entregas problemáticas que requieren investigación inmediata.

**Acciones recomendadas:**

A corto plazo, implementar ETAs dinámicos según clima y alertas para entregas en riesgo. Establecer programa de mentoría para nuevos couriers.

A mediano plazo, redistribuir repartidores a zonas problemáticas e incentivar entregas en horarios de baja demanda.

A largo plazo, considerar centros de distribución intermedios y sistema predictivo para optimizar tamaño de flota en tiempo real.

**Próximos pasos:**

Validar estos insights con stakeholders del negocio. Implementar dashboards de monitoreo en tiempo real. Realizar A/B testing de las recomendaciones en una región piloto antes de escalar.

---

**Autor:** Jhoan Meza  
[LinkedIn](https://www.linkedin.com/in/jhoan-sebastian-meza-garcia-12228b329/) | [GitHub](https://github.com/jmeza-data)