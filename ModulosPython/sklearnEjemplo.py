import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importamos las clases y funciones específicas de scikit-learn que vamos a usar
from sklearn.model_selection import train_test_split # Para dividir los datos
from sklearn.linear_model import LinearRegression    # Nuestro modelo de regresión lineal
from sklearn.metrics import mean_squared_error, r2_score # Para evaluar el modelo
from sklearn.preprocessing import OneHotEncoder      # Para convertir texto a números (categorías)
from sklearn.compose import ColumnTransformer        # Para aplicar transformaciones a columnas específicas
from sklearn.pipeline import Pipeline                # Para encadenar pasos de preprocesamiento y modelado

plt.style.use('seaborn-v0_8-darkgrid') # Estilo bonito para los gráficos

# --- PASO 1: Preparar los datos (Simulación del registro de tus ventas) ---
# Vamos a crear un DataFrame de Pandas para simular tus datos de ventas.
# En la vida real, cargarías esto desde un archivo CSV o una base de datos.

data = {
    'Tamaño': ['S', 'M', 'L', 'S', 'XL', 'M', 'L', 'XS', 'M', 'S', 'L', 'XL'],
    'Calidad_Material': [7, 8, 9, 6, 10, 7, 8, 5, 9, 7, 6, 9],
    'Precio_Venta': [15.0, 18.0, 22.0, 14.0, 25.0, 17.5, 20.0, 12.0, 21.0, 16.0, 19.0, 23.0]
}
df_camisas = pd.DataFrame(data)

print("DataFrame de Ventas de Camisas:")
print(df_camisas)
print("-" * 50)

# --- PASO 2: Preprocesamiento de los datos ---
# Nuestros modelos de Machine Learning (como LinearRegression) trabajan con números,
# no con texto. La columna 'Tamaño' es de texto (categórica), así que necesitamos
# convertirla a números. Usaremos OneHotEncoder.

# Identificamos las columnas categóricas y numéricas.
columnas_categoricas = ['Tamaño']
columnas_numericas = ['Calidad_Material']
target = 'Precio_Venta' # La variable que queremos predecir

# Creamos un "transformador" para las columnas categóricas
# OneHotEncoder: Convierte cada categoría en una nueva columna binaria (0 o 1).
#   - handle_unknown='ignore': Si aparece un tamaño que el modelo nunca ha visto, simplemente lo ignora.
#   - sparse_output=False: Devuelve un array denso de NumPy, más fácil de manejar para este ejemplo.
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Creamos un ColumnTransformer para aplicar transformaciones a columnas específicas.
# Esto es muy útil cuando tienes diferentes tipos de columnas que requieren diferentes preprocesamientos.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', one_hot_encoder, columnas_categoricas) # Aplica one_hot_encoder a las 'columnas_categoricas'
        # Podrías añadir otras transformaciones aquí, como StandardScaler para 'columnas_numericas'
    ],
    remainder='passthrough' # Mantiene las columnas numéricas sin transformar.
)
# Explicación del preprocessor:
# Esto es como tener una "fábrica de datos" que dice: "para la columna 'Tamaño', usa el OneHotEncoder,
# y el resto de las columnas (como 'Calidad_Material'), pásalas directamente sin modificar".


# --- PASO 3: Dividir los datos en conjuntos de entrenamiento y prueba ---
# Separamos las características (X) de la variable objetivo (y).
X = df_camisas[['Tamaño', 'Calidad_Material']]
y = df_camisas[target]

# train_test_split(): Dividimos nuestros datos. El 20% será para probar el modelo, 80% para entrenarlo.
#   - test_size=0.2: 20% de los datos para prueba.
#   - random_state=42: Asegura que la división sea la misma cada vez que corres el código. ¡Es importante para reproducibilidad!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamaño de X_train: {X_train.shape}")
print(f"Tamaño de X_test:  {X_test.shape}")
print(f"Tamaño de y_train: {y_train.shape}")
print(f"Tamaño de y_test:  {y_test.shape}")
print("-" * 50)


# --- PASO 4: Construir el Pipeline (Encadenar Preprocesamiento y Modelo) ---
# Un Pipeline es una forma de encadenar varios pasos de preprocesamiento y un estimador (modelo).
# Es una práctica recomendada en sklearn para mantener tu código limpio y evitar fugas de datos.

modelo_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # Primer paso: aplicar el preprocesamiento
    ('regressor', LinearRegression()) # Segundo paso: aplicar el modelo de Regresión Lineal
])
# Explicación del Pipeline:
# Cuando llamas a .fit() en el pipeline, primero se ejecuta el 'preprocessor' en tus datos de entrenamiento,
# y luego, los datos ya transformados se pasan al 'regressor' para el entrenamiento.
# Cuando llamas a .predict(), los datos de prueba pasan por el 'preprocessor' y luego por el 'regressor'.


# --- PASO 5: Entrenar el modelo ---
# modelo_pipeline.fit(): Entrenamos nuestro modelo usando los datos de entrenamiento.
# El pipeline se encarga automáticamente de aplicar el preprocesamiento antes de entrenar la regresión lineal.
print("Entrenando el modelo de Regresión Lineal...")
modelo_pipeline.fit(X_train, y_train)
print("¡Modelo entrenado exitosamente!")
print("-" * 50)


# --- PASO 6: Realizar predicciones ---
# modelo_pipeline.predict(): Usamos el modelo entrenado para hacer predicciones en los datos de prueba.
predicciones = modelo_pipeline.predict(X_test)

print("Valores Reales (y_test) vs. Predicciones:")
resultados = pd.DataFrame({'Real': y_test, 'Predicho': predicciones.round(2)})
print(resultados)
print("-" * 50)


# --- PASO 7: Evaluar el modelo ---
# Usamos las métricas de sklearn.metrics para ver qué tan bueno es nuestro modelo.

mse = mean_squared_error(y_test, predicciones)
#   - mean_squared_error(): Nos da el Error Cuadrático Medio. Un valor más bajo es mejor.
#     Nos dice, en promedio, qué tan lejos están nuestras predicciones de los valores reales.
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")

r2 = r2_score(y_test, predicciones)
#   - r2_score(): Nos da el R-cuadrado. Un valor más cercano a 1.0 es mejor.
#     Nos dice qué porcentaje de la variabilidad en el precio de venta es explicada por nuestro modelo.
print(f"Coeficiente R^2: {r2:.2f}")
print("-" * 50)

# --- PASO 8: Visualizar las predicciones (Opcional, pero muy útil) ---
# Graficamos los valores reales vs. los predichos para ver el ajuste.

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicciones, alpha=0.7, color='purple', s=100, edgecolor='black')
#   - plt.scatter(): Gráfico de dispersión. Queremos ver si los puntos se alinean con la diagonal.
#     Si 'y_test' (real) es igual a 'predicciones', todos los puntos estarían en la línea diagonal.

# Añadimos una línea diagonal perfecta para comparar (donde Predicho = Real)
min_val = min(y_test.min(), predicciones.min())
max_val = max(y_test.max(), predicciones.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
#   - plt.plot(): Dibuja una línea. Aquí, una línea roja punteada para la referencia.

plt.title('Valores Reales vs. Predicciones del Modelo', fontsize=16)
plt.xlabel('Precio Real de Venta (Bs/$)', fontsize=12)
plt.ylabel('Precio Predicho (Bs/$)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# --- PASO 9: Hacer una predicción para una nueva camisa ---
# ¡Aquí está la utilidad práctica! Predicciones para casos nuevos.
nueva_camisa = pd.DataFrame([['XS', 8], ['XL', 7]], columns=['Tamaño', 'Calidad_Material'])
#   - Creamos un nuevo DataFrame con los datos de la camisa nueva.
#     Es CRUCIAL que las columnas tengan los mismos nombres y el mismo orden que las que usaste para entrenar X.

precio_predicho_nueva_camisa = modelo_pipeline.predict(nueva_camisa)
print(f"Predicción para una camisa XS con calidad 8: {precio_predicho_nueva_camisa[0]:.2f} Bs/$")
print(f"Predicción para una camisa XL con calidad 7: {precio_predicho_nueva_camisa[1]:.2f} Bs/$")
