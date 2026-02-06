--  CREACION DE TABLAS
   

CREATE TABLE delivery_persons (
    delivery_person_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    region VARCHAR(50),
    hired_date DATE,
    is_active BIT
);

CREATE TABLE restaurants (
    restaurant_id VARCHAR(50) PRIMARY KEY,
    area VARCHAR(50),
    name VARCHAR(100),
    cuisine_type VARCHAR(50),
    avg_preparation_time_min FLOAT
);

CREATE TABLE deliveries (
    delivery_id VARCHAR(50) PRIMARY KEY,
    delivery_person_id VARCHAR(50) NOT NULL,
    restaurant_area VARCHAR(50),
    customer_area VARCHAR(50),
    delivery_distance_km FLOAT,
    delivery_time_min INT,
    order_placed_at DATETIME,
    weather_condition VARCHAR(50),
    traffic_condition VARCHAR(50),
    delivery_rating FLOAT,
    CONSTRAINT FK_deliveries_person
        FOREIGN KEY (delivery_person_id)
        REFERENCES delivery_persons(delivery_person_id)
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    delivery_id VARCHAR(50) NOT NULL,
    restaurant_id VARCHAR(50) NOT NULL,
    customer_id VARCHAR(50),
    order_value FLOAT,
    items_count INT,
    CONSTRAINT FK_orders_delivery
        FOREIGN KEY (delivery_id)
        REFERENCES deliveries(delivery_id),
    CONSTRAINT FK_orders_restaurant
        FOREIGN KEY (restaurant_id)
        REFERENCES restaurants(restaurant_id)
);


-- 1. Top 5 customer areas con mayor tiempo promedio en los ultimos 30 dias

SELECT TOP 5
    customer_area,
    AVG(delivery_time_min) AS avg_delivery_time
FROM deliveries
WHERE order_placed_at >= DATEADD(DAY, -30, GETDATE())
GROUP BY customer_area
ORDER BY avg_delivery_time DESC;

-- 2. Tiempo promedio por tr�fico, �rea del restaurante y tipo de cocina
SELECT
    d.traffic_condition,
    r.area AS restaurant_area,
    r.cuisine_type,
    AVG(d.delivery_time_min) AS avg_delivery_time
FROM deliveries d
JOIN orders o
    ON d.delivery_id = o.delivery_id
JOIN restaurants r
    ON o.restaurant_id = r.restaurant_id
GROUP BY
    d.traffic_condition,
    r.area,
    r.cuisine_type
ORDER BY avg_delivery_time DESC;

-- 3. Top 10 repartidores m�s r�pidos (>= 50 entregas y activos)
SELECT TOP 10
    d.delivery_person_id,
    AVG(d.delivery_time_min) AS avg_delivery_time,
    COUNT(*) AS total_deliveries
FROM deliveries d
JOIN delivery_persons dp
    ON d.delivery_person_id = dp.delivery_person_id
WHERE dp.is_active = 1
GROUP BY d.delivery_person_id
HAVING COUNT(*) >= 50
ORDER BY avg_delivery_time ASC;

-- 4) �rea de restaurante m�s rentable (�ltimos 3 meses)
SELECT TOP 1
    r.area,
    SUM(o.order_value) AS total_revenue
FROM orders o
JOIN restaurants r
    ON o.restaurant_id = r.restaurant_id
JOIN deliveries d
    ON o.delivery_id = d.delivery_id
WHERE d.order_placed_at >= DATEADD(MONTH, -3, GETDATE())
GROUP BY r.area
ORDER BY total_revenue DESC;

-- 5. Tendencia mensual del tiempo promedio por repartidor

WITH monthly_stats AS (
    SELECT
        d.delivery_person_id,
        FORMAT(d.order_placed_at, 'yyyy-MM') AS year_month,
        AVG(d.delivery_time_min) AS avg_time,
        COUNT(*) AS deliveries_count
    FROM deliveries d
    JOIN delivery_persons dp
        ON d.delivery_person_id = dp.delivery_person_id
    WHERE dp.is_active = 1
    GROUP BY d.delivery_person_id, FORMAT(d.order_placed_at, 'yyyy-MM')
    HAVING COUNT(*) >= 10  -- Al menos 10 entregas por mes
)
SELECT
    m1.delivery_person_id,
    m1.year_month AS recent_month,
    m1.avg_time AS recent_avg_time,
    m2.year_month AS previous_month,
    m2.avg_time AS previous_avg_time,
    (m1.avg_time - m2.avg_time) AS time_increase
FROM monthly_stats m1
JOIN monthly_stats m2
    ON m1.delivery_person_id = m2.delivery_person_id
    AND DATEADD(MONTH, 1, CAST(m2.year_month + '-01' AS DATE)) = CAST(m1.year_month + '-01' AS DATE)
WHERE m1.avg_time > m2.avg_time  -- Solo cuando aumenta
ORDER BY time_increase DESC;