import numpy as np
import matplotlib.pyplot as plt

# --- PASO 1: Definir los datos de tus ventas (usando NumPy) ---
# Primero, definimos nuestros datos, como ya sabemos hacerlo con NumPy.
# Damos nombres a los productos para que sea más fácil entender el gráfico.
nombres_productos = ['Harina PAN', 'Arroz', 'Pasta', 'Aceite', 'Café']
precios_productos = np.array([1.50, 2.75, 0.99, 5.00, 3.20])
cantidades_vendidas = np.array([10, 5, 20, 2, 8])

print(f"Productos: {nombres_productos}")
print(f"Precios:   {precios_productos}")
print(f"Cantidades: {cantidades_vendidas}")

# Calculamos las ventas por producto (ingreso generado por cada uno)
ventas_por_producto = precios_productos * cantidades_vendidas
print(f"Ventas por cada producto: {ventas_por_producto.round(2)}")

# --- PASO 2: Visualizar la distribución de precios con un Histograma ---

plt.figure(figsize=(8, 5)) # Creamos una figura (el lienzo) de 8 pulgadas de ancho por 5 de alto.
#   - plt.figure(): Sirve para iniciar un nuevo gráfico. Es como agarrar una hoja en blanco.
#     El `figsize` nos ayuda a controlar el tamaño de la imagen que se va a generar.

plt.hist(precios_productos, bins=len(precios_productos), edgecolor='black', alpha=0.7)
#   - plt.hist(): Dibuja un histograma.
#     - `precios_productos`: El array de datos que queremos analizar.
#     - `bins=len(precios_productos)`: Aquí le decimos que cree una barra por cada precio único.
#       Si tuvieras muchos precios diferentes, usarías un número fijo de `bins` (ej., `bins=5`)
#       para agruparlos en rangos.
#     - `edgecolor='black'`: Pone un borde negro a cada barra para que se distingan mejor.
#     - `alpha=0.7`: Hace las barras un poco transparentes, útil si se superponen.

plt.title('Distribución de Precios de Productos', fontsize=16)
#   - plt.title(): Coloca un título al gráfico para que sepamos de qué se trata.
#     Es como ponerle un encabezado a tu informe.

plt.xlabel('Precio (Bs/$)', fontsize=12)
#   - plt.xlabel(): Etiqueta el eje horizontal (eje X). ¡Es crucial para entender qué representa!

plt.ylabel('Número de Productos', fontsize=12)
#   - plt.ylabel(): Etiqueta el eje vertical (eje Y). Nos dice qué estamos midiendo en este eje.

plt.grid(axis='y', linestyle='--', alpha=0.7)
#   - plt.grid(): Añade una cuadrícula al gráfico.
#     - `axis='y'`: Solo muestra líneas de cuadrícula en el eje Y.
#     - `linestyle='--'`: Hace que las líneas sean punteadas.
#     - `alpha=0.7`: Las hace un poco transparentes para que no distraigan.

plt.show() # ¡Importantísimo! Muestra el gráfico en pantalla. Si no lo pones, no verás nada.
#   - plt.show(): Renderiza y muestra el gráfico actual. Piensa que es el botón de "imprimir" o "mostrar" el gráfico.

# --- PASO 3: Comparar el Monto de Ventas por Producto con un Gráfico de Barras ---

plt.figure(figsize=(10, 6)) # Creamos otra figura para este nuevo gráfico.

# Creamos un gráfico de barras.
plt.bar(nombres_productos, ventas_por_producto, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
#   - plt.bar(): Dibuja un gráfico de barras.
#     - `nombres_productos`: Los nombres de las categorías para cada barra (eje X).
#     - `ventas_por_producto`: Los valores (altura de las barras, eje Y).
#     - `color`: Podemos pasar una lista de colores para que cada barra tenga un color diferente.

plt.title('Ingresos por Tipo de Producto', fontsize=18)
plt.xlabel('Producto', fontsize=14)
plt.ylabel('Ingreso Generado (Bs/$)', fontsize=14)
plt.xticks(rotation=45, ha='right') # Rota las etiquetas del eje X para que no se superpongan
#   - plt.xticks(): Personaliza las "marcas" en el eje X.
#     - `rotation=45`: Rota los nombres de los productos 45 grados para que quepan bien.
#     - `ha='right'`: Alinea el texto a la derecha del punto de la marca.

plt.grid(axis='y', linestyle=':', alpha=0.6) # Cuadrícula en Y, punteada
plt.tight_layout() # Ajusta automáticamente los parámetros de la subtrama para un diseño ajustado.
#   - plt.tight_layout(): A veces, las etiquetas o títulos pueden salirse del gráfico.
#     Esta función intenta ajustarlo todo para que se vea bien.

plt.show()

# --- PASO 4: Analizar la relación entre Precio y Cantidad con un Gráfico de Dispersión ---

plt.figure(figsize=(9, 7)) # Otra figura más.

plt.scatter(precios_productos, cantidades_vendidas, s=cantidades_vendidas*30, alpha=0.8, color='darkblue')
#   - plt.scatter(): Dibuja un gráfico de dispersión.
#     - `precios_productos`: Eje X.
#     - `cantidades_vendidas`: Eje Y.
#     - `s=cantidades_vendidas*30`: ¡Aquí la magia! El tamaño de cada punto es proporcional a la cantidad vendida.
#       Los productos que se vendieron más, tendrán un punto más grande. Esto nos da una dimensión extra en el gráfico.
#     - `alpha=0.8`: Un poco de transparencia para ver si los puntos se superponen.
#     - `color='darkblue'`: Un solo color para todos los puntos.

# Para añadir etiquetas a los puntos (opcional, pero útil)
for i, txt in enumerate(nombres_productos):
    plt.annotate(txt, (precios_productos[i], cantidades_vendidas[i]), textcoords="offset points", xytext=(0,10), ha='center')
    # Esto es un poco más avanzado, pero sirve para poner el nombre del producto al lado de cada punto.
    # - plt.annotate(): Añade texto a un punto específico del gráfico.
    #   - `txt`: El texto a mostrar (nombre del producto).
    #   - `(precios_productos[i], cantidades_vendidas[i])`: Las coordenadas del punto.
    #   - `xytext=(0,10)`: Desplaza el texto 10 puntos hacia arriba del punto.
    #   - `ha='center'`: Alinea el texto al centro.


plt.title('Relación entre Precio y Cantidad Vendida (Tamaño = Cantidad)', fontsize=16)
plt.xlabel('Precio Unitario (Bs/$)', fontsize=12)
plt.ylabel('Cantidad Vendida', fontsize=12)
plt.grid(True)
plt.show()
