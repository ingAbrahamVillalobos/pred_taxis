# Proyecto: Predicción de Demanda de Taxis (Series Temporales)

## Introducción

Este proyecto, desarrollado como parte de un bootcamp de Data Scientist, aborda la predicción de la demanda de taxis en aeropuertos para la compañía Sweet Lift Taxi. El objetivo es construir un modelo que anticipe la cantidad de pedidos de taxis para la próxima hora, permitiendo a la empresa optimizar la asignación de conductores durante las horas pico.

## Problema de Negocio

Sweet Lift Taxi necesita una herramienta predictiva precisa para gestionar eficientemente su flota en aeropuertos. La meta es predecir la demanda horaria de taxis con una métrica de Error Cuadrático Medio (RMSE) no superior a 48 en el conjunto de prueba.

## Metodología

El proyecto se estructura en las siguientes fases:

### 1. Preparación de Datos

* **Carga de datos:** Se cargó el archivo `taxi.csv`, estableciendo la columna de tiempo como índice y parseando las fechas.
* **Remuestreo:** Los datos originales, con una frecuencia de 10 minutos, fueron remuestreados a intervalos de una hora (`1H`) para ajustarse al requerimiento del problema, sumando el número de pedidos por hora.
* **Inspección inicial:** Se verificó la información general del DataFrame remuestreado (`.info()`, `.describe()`) y la ausencia de duplicados en el índice.

### 2. Análisis Exploratorio de Datos (EDA)

* Se visualizó la serie temporal del número de pedidos por hora.
* **Observaciones clave:**
    * **Tendencia creciente:** Se observó un aumento significativo en los pedidos a partir de julio/agosto.
    * **Variabilidad:** Se notó una considerable dispersión en los valores horarios, especialmente hacia el final del periodo.
    * **Picos:** Existencia de picos puntuales de pedidos (superiores a 400), posiblemente debido a eventos específicos o anomalías.

### 3. Generación de Características

Se implementó la función `make_features` para enriquecer el dataset con variables temporales y rezagadas, cruciales para el modelado de series temporales:

* **Características de tiempo:** `year`, `month`, `day`, `dayofweek`, `hour`.
* **Lags (rezagos):** Se crearon `lag_1` a `lag_6` (6 lags) para capturar la dependencia de pedidos anteriores.
* **Media móvil:** Se calculó una `rolling_mean` con un tamaño de ventana de 10 para suavizar la serie y capturar tendencias a corto plazo.

### 4. División de Datos y Entrenamiento de Modelos

* Los datos se dividieron en conjuntos de entrenamiento (90%) y prueba (10%), manteniendo el orden cronológico (`shuffle=False`).
* Se eliminaron las filas con valores `NaN` generados por los rezagos y la media móvil en el conjunto de entrenamiento.
* Se entrenaron y evaluaron cuatro modelos de regresión:
    * **Regresión Lineal:** Modelo base para establecer una referencia.
    * **RandomForestRegressor:** Un modelo de ensamble robusto.
    * **CatBoostRegressor:** Un potente algoritmo de boosting.
    * **LGBMRegressor:** Otro algoritmo de boosting de alto rendimiento, conocido por su eficiencia.

### 5. Evaluación y Comparación de Modelos

La métrica principal de evaluación fue el **RMSE (Root Mean Squared Error)** en el conjunto de prueba. Los resultados obtenidos fueron:

* **Regresión Lineal:** RMSE = 53.16
* **RandomForestRegressor:** RMSE = 46.50
* **CatBoostRegressor:** RMSE = 46.80
* **LGBMRegressor:** **RMSE = 43.74** (¡El mejor resultado!)

## Conclusión y Recomendación

El objetivo de este proyecto era construir un modelo capaz de predecir la cantidad de pedidos de taxis por hora con un RMSE no superior a 48.

De los modelos evaluados, el modelo entrenado con **LightGBM** demostró ser el más preciso, obteniendo un **RMSE de 43.74**. Este resultado no solo es el mejor entre los modelos probados, sino que también cumple y supera el umbral requerido de RMSE ≤ 48.

Por lo tanto, se recomienda el uso del modelo **LightGBM** para la predicción operativa de la demanda horaria de taxis por parte de Sweet Lift Taxi. Este modelo permitirá a la compañía anticipar con mayor precisión las horas pico y optimizar la asignación de sus conductores.

## Tecnologías Utilizadas

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `sklearn` (LinearRegression, RandomForestRegressor, mean_squared_error, train_test_split)
* `catboost` (CatBoostRegressor)
* `lightgbm` (LGBMRegressor)
