import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # ¡Importamos Pandas, el mejor amigo de Seaborn!

# Para que los gráficos se vean bonitos con el estilo de Seaborn
plt.style.use('seaborn-v0_8-darkgrid')

# --- PASO 1: Definir los datos y convertirlos a un DataFrame de Pandas ---
# Es el mismo escenario, pero ahora lo ponemos en un diccionario para luego hacerlo DataFrame
datos_bodega = {
    'Producto': ['Harina PAN', 'Arroz', 'Pasta', 'Aceite', 'Café', 'Azúcar', 'Leche', 'Huevos'],
    'Precio_Bs': np.array([1.50, 2.75, 0.99, 5.00, 3.20, 1.80, 4.50, 2.00]),
    'Cantidad_Vendida': np.array([10, 5, 20, 2, 8, 15, 3, 12])
}
# Agregamos más productos para que el histograma y otros gráficos sean más interesantes.

df_ventas = pd.DataFrame(datos_bodega)
#   - pd.DataFrame(): Esta es la función de Pandas para crear un DataFrame.
#     Un DataFrame es como una hoja de cálculo o una tabla de base de datos en Python,
#     con filas y columnas con nombres. ¡Es súper útil para organizar tus datos!

# Calculamos el ingreso total por cada producto y lo añadimos como una nueva columna
df_ventas['Ingreso_Generado'] = df_ventas['Precio_Bs'] * df_ventas['Cantidad_Vendida']
#   - df_ventas['Nueva_Columna'] = ... : Así de fácil se crea una nueva columna en un DataFrame.
#     Pandas también hace las operaciones elemento a elemento automáticamente, como NumPy.

print("DataFrame de Ventas de la Bodega:")
print(df_ventas)
print("-" * 50)


# --- PASO 2: Visualizar la Distribución de Precios con sns.histplot (con KDE) ---

plt.figure(figsize=(10, 6))
# sns.histplot(): Dibujamos un histograma de los precios, pero con la curva de densidad (KDE).
#   - data=df_ventas: Le decimos a Seaborn que use nuestro DataFrame para obtener los datos.
#   - x='Precio_Bs': Especificamos qué columna del DataFrame queremos en el eje X.
#   - kde=True: ¡Aquí la magia! Superpone una Estimación de Densidad del Kernel.
#     Es una curva suave que nos muestra la forma de la distribución de los precios de forma más clara que solo las barras.
#   - bins=5: Agrupamos los precios en 5 rangos.
#   - edgecolor='black': Borde para las barras.
#   - alpha=0.7: Transparencia para las barras.
sns.histplot(data=df_ventas, x='Precio_Bs', kde=True, bins=5, edgecolor='black', alpha=0.7)

plt.title('Distribución de Precios de Productos con KDE', fontsize=16)
plt.xlabel('Precio (Bs/$)', fontsize=12)
plt.ylabel('Frecuencia / Densidad', fontsize=12)
plt.show()

# Explicación:
# El histograma muestra cuántos productos caen en ciertos rangos de precios.
# La línea (KDE) te da una idea más fluida de dónde se concentran tus precios.
# Por ejemplo, si hay una joroba alta alrededor de 2 Bs/$, significa que tienes muchos productos en ese rango de precios.


# --- PASO 3: Comparar Ingresos Generados por Producto con sns.barplot ---

plt.figure(figsize=(12, 7))
# sns.barplot(): Dibuja un gráfico de barras donde la altura de la barra representa una medida numérica
#                (por defecto, la media, pero aquí es directa ya que tenemos una fila por producto).
#   - data=df_ventas: Nuestro DataFrame.
#   - x='Producto': Los nombres de los productos en el eje X.
#   - y='Ingreso_Generado': La nueva columna que creamos, que será la altura de las barras.
#   - palette='viridis': Un conjunto de colores predefinidos y atractivos de Seaborn.
sns.barplot(data=df_ventas, x='Producto', y='Ingreso_Generado', palette='viridis')

plt.title('Ingresos Generados por Cada Producto', fontsize=16)
plt.xlabel('Producto', fontsize=12)
plt.ylabel('Ingreso Total (Bs/$)', fontsize=12)
plt.xticks(rotation=45, ha='right') # Rotamos las etiquetas del eje X para que no se superpongan
plt.grid(axis='y', linestyle='--', alpha=0.7) # Cuadrícula en Y
plt.tight_layout() # Asegura que todo el gráfico se vea bien
plt.show()

# Explicación:
# Este gráfico de barras es similar al de Matplotlib, pero Seaborn le da un estilo más pulcro
# y es más fácil de integrar con DataFrames. Puedes ver rápidamente qué productos son tus "caballos de batalla"
# en términos de ingresos. ¡El "Pasta" está vendiendo duro por lo visto!


# --- PASO 4: Analizar la relación entre Precio, Cantidad y Ingreso con sns.scatterplot ---

plt.figure(figsize=(10, 8))
# sns.scatterplot(): Dibuja un gráfico de dispersión, pero con la capacidad de mapear
#                    más variables a las características visuales (color, tamaño).
#   - data=df_ventas: Nuestro DataFrame.
#   - x='Precio_Bs': Precio en el eje X.
#   - y='Cantidad_Vendida': Cantidad vendida en el eje Y.
#   - hue='Producto': ¡Aquí está lo bueno! Colorea cada punto según el nombre del producto.
#     Seaborn automáticamente crea una leyenda.
#   - size='Ingreso_Generado': ¡Otra genialidad! El tamaño de cada punto se basa en el ingreso generado.
#     Los productos que generan más ingreso tendrán un punto más grande.
#   - sizes=(50, 1000): Rango de tamaño para los puntos (del más pequeño al más grande).
#   - alpha=0.8: Transparencia.
sns.scatterplot(data=df_ventas, x='Precio_Bs', y='Cantidad_Vendida',
                hue='Producto', size='Ingreso_Generado', sizes=(100, 1000), alpha=0.8)

plt.title('Relación Precio vs. Cantidad Vendida por Producto (Tamaño = Ingreso)', fontsize=16)
plt.xlabel('Precio Unitario (Bs/$)', fontsize=12)
plt.ylabel('Cantidad Vendida', fontsize=12)
plt.legend(title='Producto', bbox_to_anchor=(1.05, 1), loc='upper left') # Mueve la leyenda fuera del gráfico
#   - plt.legend(): Añade la leyenda.
#   - bbox_to_anchor=(1.05, 1), loc='upper left': Mueve la leyenda fuera del área del gráfico
#     para que no tape los puntos. Esto es un truco de Matplotlib que se usa a menudo con Seaborn.
plt.grid(True)
plt.tight_layout() # Asegura que la leyenda no se corte
plt.show()

# Explicación:
# Este scatter plot es mucho más informativo.
# - El eje X te dice el precio.
# - El eje Y te dice la cantidad vendida.
# - El COLOR te dice qué producto es.
# - El TAMAÑO del punto te dice cuánto ingreso generó ese producto.
# Puedes ver rápidamente que "Aceite" es caro pero se vende poco, mientras que "Pasta" es barato, se vende mucho y ¡genera buen ingreso!
