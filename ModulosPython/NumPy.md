### **Funciones Clave de NumPy (Para tus Notas)**

Para cada función, asumiremos que ya tienes un array NumPy, por ejemplo:
`mi_array = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])`
`mi_array_2d = np.array([[1, 2, 3], [4, 5, 6]])`

---

1.  **`numpy.array()`**
    * **¿Para qué sirve?** Sirve para crear un array de NumPy a partir de una lista o tupla de Python. Es el punto de partida para casi todo en NumPy.
    * **Sintaxis:** `numpy.array(objeto_tipo_lista_o_tupla)`
    * **Ejemplo:**
        ```python
        import numpy as np
        lista_de_numeros = [1, 2, 3, 4, 5]
        mi_primer_array = np.array(lista_de_numeros)
        print(f"Array creado: {mi_primer_array}")
        # Salida: Array creado: [1 2 3 4 5]
        ```

2.  **`numpy.mean()`**
    * **¿Para qué sirve?** Sirve para calcular la media aritmética (el promedio) de los elementos en un array.
    * **Sintaxis:** `numpy.mean(array)`
    * **Ejemplo:**
        ```python
        import numpy as np
        mi_array = np.array([10, 20, 30, 40, 50])
        promedio = np.mean(mi_array)
        print(f"El promedio es: {promedio}")
        # Salida: El promedio es: 30.0
        ```
    * **Nota:** También puedes especificar un `axis` para calcular la media a lo largo de filas o columnas en arrays 2D. Por ejemplo: `np.mean(mi_array_2d, axis=0)` (media por columna) o `np.mean(mi_array_2d, axis=1)` (media por fila).

3.  **`numpy.std()`**
    * **¿Para qué sirve?** Sirve para calcular la desviación estándar de los elementos en un array. La desviación estándar te dice qué tan dispersos están los datos con respecto a la media.
    * **Sintaxis:** `numpy.std(array)`
    * **Ejemplo:**
        ```python
        import numpy as np
        mi_array = np.array([10, 11, 10, 12, 9, 10]) # Datos poco dispersos
        desviacion = np.std(mi_array)
        print(f"La desviación estándar es: {desviacion:.2f}")
        # Salida: La desviación estándar es: 0.91
        ```
    * **Nota:** También puedes usar el parámetro `axis` para arrays multidimensionales.

4.  **`numpy.max()`**
    * **¿Para qué sirve?** Sirve para encontrar el valor máximo dentro de un array.
    * **Sintaxis:** `numpy.max(array)`
    * **Ejemplo:**
        ```python
        import numpy as np
        mi_array = np.array([5, 12, 3, 20, 8])
        valor_maximo = np.max(mi_array)
        print(f"El valor máximo es: {valor_maximo}")
        # Salida: El valor máximo es: 20
        ```

5.  **`numpy.min()`**
    * **¿Para qué sirve?** Sirve para encontrar el valor mínimo dentro de un array.
    * **Sintaxis:** `numpy.min(array)`
    * **Ejemplo:**
        ```python
        import numpy as np
        mi_array = np.array([5, 12, 3, 20, 8])
        valor_minimo = np.min(mi_array)
        print(f"El valor mínimo es: {valor_minimo}")
        # Salida: El valor mínimo es: 3
        ```

6.  **`numpy.sum()`**
    * **¿Para qué sirve?** Sirve para calcular la suma de todos los elementos en un array.
    * **Sintaxis:** `numpy.sum(array)`
    * **Ejemplo:**
        ```python
        import numpy as np
        mi_array = np.array([1, 2, 3, 4, 5])
        suma_total = np.sum(mi_array)
        print(f"La suma total es: {suma_total}")
        # Salida: La suma total es: 15
        ```
    * **Nota:** También puedes usar el parámetro `axis` para sumar a lo largo de filas o columnas.

7.  **`numpy.median()`**
    * **¿Para qué sirve?** Sirve para calcular la mediana de los elementos en un array. La mediana es el valor central de un conjunto de datos ordenado.
    * **Sintaxis:** `numpy.median(array)`
    * **Ejemplo:**
        ```python
        import numpy as np
        mi_array = np.array([1, 7, 3, 9, 5]) # Desordenado
        mediana = np.median(mi_array)
        print(f"La mediana es: {mediana}")
        # Salida: La mediana es: 5.0 (porque ordenado sería [1, 3, 5, 7, 9])
        ```

8.  **`numpy.percentile()`**
    * **¿Para qué sirve?** Sirve para calcular el percentil N de los elementos en un array. Un percentil te dice el valor por debajo del cual cae un determinado porcentaje de las observaciones.
    * **Sintaxis:** `numpy.percentile(array, q)`
        * `array`: El array de datos.
        * `q`: El percentil a calcular (un valor entre 0 y 100).
    * **Ejemplo:**
        ```python
        import numpy as np
        mi_array = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        percentil_25 = np.percentile(mi_array, 25)
        print(f"El percentil 25 es: {percentil_25}")
        # Salida: El percentil 25 es: 32.5 (25% de los valores son 32.5 o menos)
        ```

9.  **`numpy.random.rand()`**
    * **¿Para qué sirve?** Sirve para crear un array de forma específica con números aleatorios uniformemente distribuidos entre 0 y 1.
    * **Sintaxis:** `numpy.random.rand(d1, d2, ...)` (donde d1, d2 son las dimensiones)
    * **Ejemplo:**
        ```python
        import numpy as np
        array_aleatorio_1d = np.random.rand(5) # 5 números aleatorios entre 0 y 1
        array_aleatorio_2d = np.random.rand(2, 3) # Matriz 2x3 de números aleatorios
        print(f"Array aleatorio 1D:\n{array_aleatorio_1d.round(2)}")
        print(f"Array aleatorio 2D:\n{array_aleatorio_2d.round(2)}")
        ```

10. **`numpy.random.randn()`**
    * **¿Para qué sirve?** Sirve para crear un array de forma específica con números aleatorios que siguen una distribución normal estándar (media 0, desviación estándar 1).
    * **Sintaxis:** `numpy.random.randn(d1, d2, ...)`
    * **Ejemplo:**
        ```python
        import numpy as np
        array_normal = np.random.randn(5) # 5 números aleatorios de una distribución normal estándar
        print(f"Array con distribución normal estándar:\n{array_normal.round(2)}")
        ```

11. **`numpy.random.normal()`**
    * **¿Para qué sirve?** Sirve para crear un array de forma específica con números aleatorios que siguen una distribución normal con una media y desviación estándar dadas.
    * **Sintaxis:** `numpy.random.normal(loc=media, scale=desviacion_estandar, size=forma)`
        * `loc`: La media de la distribución.
        * `scale`: La desviación estándar de la distribución.
        * `size`: La forma del array resultante.
    * **Ejemplo:**
        ```python
        import numpy as np
        mi_array_normal = np.random.normal(loc=170, scale=5, size=10) # 10 números con media 170 y desv. est. 5
        print(f"Array con distribución normal personalizada:\n{mi_array_normal.round(2)}")
        ```

12. **`numpy.linspace()`**
    * **¿Para qué sirve?** Sirve para crear una secuencia de números espaciados uniformemente en un intervalo específico. Es muy útil para generar datos para gráficos o para funciones.
    * **Sintaxis:** `numpy.linspace(inicio, fin, num=cantidad_de_puntos)`
    * **Ejemplo:**
        ```python
        import numpy as np
        secuencia = np.linspace(0, 10, num=5) # 5 puntos entre 0 y 10 (incluidos)
        print(f"Secuencia: {secuencia}")
        # Salida: Secuencia: [ 0.   2.5  5.   7.5 10. ]
        ```

13. **`.reshape()` (Método de los arrays de NumPy)**
    * **¿Para qué sirve?** Sirve para cambiar la forma (dimensiones) de un array sin cambiar sus datos. Es crucial cuando un algoritmo espera una forma específica (como un array 2D para las características).
    * **Sintaxis:** `array.reshape(dimension1, dimension2, ...)`
        * `-1` en una dimensión le dice a NumPy que calcule esa dimensión automáticamente.
    * **Ejemplo:**
        ```python
        import numpy as np
        array_1d = np.array([1, 2, 3, 4, 5, 6])
        array_2d_columna = array_1d.reshape(-1, 1) # Lo convierte en una columna (6 filas, 1 columna)
        array_2d_matriz = array_1d.reshape(2, 3) # Lo convierte en una matriz de 2 filas por 3 columnas
        print(f"Array 1D original:\n{array_1d}")
        print(f"Array 2D como columna:\n{array_2d_columna}")
        print(f"Array 2D como matriz 2x3:\n{array_2d_matriz}")
        ```

14. **`.flatten()` (Método de los arrays de NumPy)**
    * **¿Para qué sirve?** Sirve para convertir un array multidimensional (como una matriz) en un array unidimensional (una sola fila).
    * **Sintaxis:** `array.flatten()`
    * **Ejemplo:**
        ```python
        import numpy as np
        mi_array_2d = np.array([[1, 2, 3], [4, 5, 6]])
        array_aplanado = mi_array_2d.flatten()
        print(f"Array 2D original:\n{mi_array_2d}")
        print(f"Array aplanado:\n{array_aplanado}")
        # Salida: Array aplanado: [1 2 3 4 5 6]
        ```

---

¡Uff, mi pana! Con esas funciones en tus notas, ya tienes una base sólida para empezar a manipular datos como un pro con NumPy. Son las más comunes y las que más vas a ver y usar en el día a día del análisis de datos y el Machine Learning. ¡Sigue así que vas por buen camino!
