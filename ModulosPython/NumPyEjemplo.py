import numpy as np

# --- PASO 1: Definir los datos de tus ventas con NumPy Arrays ---
# Imagina que estos son los precios unitarios de 5 productos diferentes.
precios_productos = np.array([1.50, 2.75, 0.99, 5.00, 3.20])
#   - np.array(): ¡Aquí estamos usando la función clave para crear nuestros arrays!
#     Sirve para transformar una lista de Python en un array de NumPy, que es mucho más eficiente
#     para operaciones numéricas grandes.

# Y estas son las cantidades vendidas de cada uno de esos 5 productos.
cantidades_vendidas = np.array([10, 5, 20, 2, 8])
#   - Fíjate que el orden de los precios y las cantidades debe coincidir para cada producto.
#     Por ejemplo, el producto de 1.50 bolívares (o dólares, tú eliges) se vendió 10 veces.

print(f"Precios de los productos: {precios_productos}")
print(f"Cantidades vendidas:     {cantidades_vendidas}")

# --- PASO 2: Calcular estadísticas básicas de los precios ---

# Promedio de precios
promedio_precios = np.mean(precios_productos)
#   - np.mean(): Esta función calcula la media aritmética de todos los elementos en el array.
#     Nos da una idea del precio típico de tus productos.
print(f"\nPrecio promedio de tus productos: {promedio_precios:.2f}")

# Precio más alto
precio_maximo = np.max(precios_productos)
#   - np.max(): ¡Sencillito! Encuentra el valor más grande dentro del array.
#     Útil para saber cuál es tu producto más costoso.
print(f"El producto más caro cuesta: {precio_maximo:.2f}")

# Precio más bajo
precio_minimo = np.min(precios_productos)
#   - np.min(): Lo opuesto a np.max(), encuentra el valor más pequeño.
#     Así sabes cuál es tu producto más económico.
print(f"El producto más barato cuesta: {precio_minimo:.2f}")

# Desviación estándar de los precios
desviacion_precios = np.std(precios_productos)
#   - np.std(): Calcula qué tan dispersos están los precios con respecto al promedio.
#     Si es un número alto, tus precios varían mucho; si es bajo, son más consistentes.
print(f"La dispersión de los precios (desviación estándar) es: {desviacion_precios:.2f}")

# --- PASO 3: Calcular el total de productos vendidos ---

total_productos_vendidos = np.sum(cantidades_vendidas)
#   - np.sum(): ¡Suma todos los elementos del array!
#     Te da la cantidad total de unidades que moviste.
print(f"\nCantidad total de productos vendidos: {total_productos_vendidos} unidades")

# --- PASO 4: Calcular el monto total de ventas (¡la caja!) ---

# Para calcular el monto de venta de cada producto, multiplicamos el precio por la cantidad.
# ¡NumPy hace esto elemento a elemento automáticamente!
ventas_por_producto = precios_productos * cantidades_vendidas
#   - Cuando multiplicas dos arrays de la misma forma en NumPy, se multiplican elemento a elemento.
#     Es decir, (precio_1 * cantidad_1), (precio_2 * cantidad_2), y así.
print(f"Ventas por cada tipo de producto: {ventas_por_producto:.2f}")

# Ahora, sumamos todas esas ventas individuales para obtener el total.
monto_total_ventas = np.sum(ventas_por_producto)
#   - Otra vez np.sum() para sumar el resultado de la multiplicación anterior.
#     ¡Este es tu ingreso bruto por esos productos!
print(f"Monto total de ventas de la semana: {monto_total_ventas:.2f} Bs/$\n")

# --- PASO 5: Un poquito más avanzado: ¿Qué porcentaje de mis ventas representa el producto más caro? ---

# Primero, encontramos el índice (posición) del precio máximo.
indice_precio_maximo = np.argmax(precios_productos)
#   - np.argmax(): Esta es una función útil que te devuelve el *índice* (la posición)
#     del valor máximo en un array. Esto nos permite saber "cuál" producto es el más caro.
print(f"El producto más caro está en la posición (índice) {indice_precio_maximo}")

# Usamos ese índice para obtener la venta específica de ese producto.
venta_producto_mas_caro = ventas_por_producto[indice_precio_maximo]
print(f"La venta del producto más caro fue de: {venta_producto_mas_caro:.2f}")

# Calculamos el porcentaje.
porcentaje_venta_max = (venta_producto_mas_caro / monto_total_ventas) * 100
print(f"El producto más caro representa el {porcentaje_venta_max:.2f}% del monto total de ventas.")
