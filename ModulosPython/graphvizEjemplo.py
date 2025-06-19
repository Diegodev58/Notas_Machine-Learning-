import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importamos las clases de scikit-learn necesarias para el árbol de decisión
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importamos las funciones de sklearn para exportar y la librería graphviz
from sklearn.tree import export_graphviz # Esta es la clave para conectar con graphviz
import graphviz # La librería de Python que interactúa con el software Graphviz

plt.style.use('seaborn-v0_8-darkgrid')

# --- PASO 1: Crear un dataset de ejemplo (simulando datos de clientes) ---
# Vamos a simular un dataset de clientes con su uso de datos, número de quejas y su satisfacción.
# Queremos predecir si un cliente es 'Satisfecho' (1) o 'Insatisfecho' (0).

data = {
    'Uso_Datos_GB': [10, 5, 25, 12, 30, 8, 15, 2, 28, 7, 20, 3, 18, 9, 22],
    'Num_Quejas': [1, 3, 0, 2, 0, 4, 1, 5, 0, 3, 1, 4, 0, 2, 0],
    'Satisfaccion': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # 1: Satisfecho, 0: Insatisfecho
}
df_clientes = pd.DataFrame(data)

# Mapeamos los números de 'Satisfaccion' a nombres legibles para el gráfico
mapeo_satisfaccion = {0: 'Insatisfecho', 1: 'Satisfecho'}
df_clientes['Satisfaccion_Etiqueta'] = df_clientes['Satisfaccion'].map(mapeo_satisfaccion)

print("DataFrame de Clientes:")
print(df_clientes)
print("-" * 50)

# --- PASO 2: Preparar los datos para el modelo ---
X = df_clientes[['Uso_Datos_GB', 'Num_Quejas']] # Características (variables independientes)
y = df_clientes['Satisfaccion'] # Variable objetivo (lo que queremos predecir)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Tamaño del conjunto de entrenamiento (X_train): {X_train.shape}")
print(f"Tamaño del conjunto de prueba (X_test):  {X_test.shape}")
print("-" * 50)

# --- PASO 3: Entrenar el Árbol de Decisión ---
# Instanciamos el clasificador de Árbol de Decisión.
# max_depth=3: Limitamos la profundidad para que el árbol no sea muy grande y sea fácil de visualizar.
# random_state=42: Para que el resultado del árbol sea siempre el mismo.
modelo_arbol = DecisionTreeClassifier(max_depth=3, random_state=42)

print("Entrenando el Árbol de Decisión...")
modelo_arbol.fit(X_train, y_train)
print("¡Árbol de Decisión entrenado exitosamente!")
print("-" * 50)

# --- PASO 4: Evaluar el modelo (opcional, pero buena práctica) ---
predicciones = modelo_arbol.predict(X_test)
precision = accuracy_score(y_test, predicciones)
print(f"Precisión del modelo en el conjunto de prueba: {precision:.2f}")
print("-" * 50)

# --- PASO 5: Exportar el Árbol de Decisión a formato DOT con `export_graphviz` ---
# Aquí es donde entra `export_graphviz` de scikit-learn.
# Genera un archivo en formato DOT, que es un lenguaje de descripción de gráficos.

dot_data = export_graphviz(
    modelo_arbol,
    out_file=None, # Le decimos que no guarde en un archivo, sino que devuelva el texto DOT
    feature_names=X.columns, # Los nombres de tus columnas para las decisiones
    class_names=['Insatisfecho', 'Satisfecho'], # Los nombres de las clases de tu variable objetivo
    filled=True, # Rellena los nodos con colores de clase
    rounded=True, # Dibuja los nodos con esquinas redondeadas
    special_characters=True # Maneja caracteres especiales
)
# Explicación de export_graphviz():
# Esta función toma tu modelo de árbol entrenado y lo convierte en una cadena de texto
# que describe el árbol en el lenguaje DOT. Cada nodo, cada flecha, cada decisión.

print("Contenido DOT generado (primeras 200 caracteres):")
print(dot_data[:200] + "...\n") # Mostramos solo un pedazo para no saturar la salida
print("-" * 50)

# --- PASO 6: Visualizar el Árbol usando `graphviz` ---
# Ahora usamos la librería `graphviz` de Python para tomar ese texto DOT y renderizarlo.

graph = graphviz.Source(dot_data)
#   - graphviz.Source(): Crea un objeto de gráfico a partir de la cadena de texto DOT.
#     Es como decirle a Graphviz: "Aquí está la descripción de mi diagrama".

graph.render("arbol_satisfaccion_clientes", format='png', view=True)
#   - graph.render(): Renderiza el gráfico y lo guarda en un archivo.
#     - "arbol_satisfaccion_clientes": El nombre base del archivo de salida (ej. arbol_satisfaccion_clientes.png)
#     - format='png': El formato de imagen en el que quieres guardar el árbol (puedes usar 'pdf', 'svg', etc.)
#     - view=True: ¡Importante! Abre automáticamente el archivo generado con el visor predeterminado de tu sistema.
#       Si no pones 'view=True', el archivo se guardará, pero no se abrirá automáticamente.

print("¡Árbol de decisión visualizado y guardado como 'arbol_satisfaccion_clientes.png'!")
print("Se abrirá automáticamente en tu visor de imágenes por defecto.")
