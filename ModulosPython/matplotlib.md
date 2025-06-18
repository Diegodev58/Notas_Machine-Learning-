### **Funciones Clave de `matplotlib.pyplot` (Para tus Notas)**

Para usar estas funciones, generalmente las importamos como `plt`.
`import matplotlib.pyplot as plt`

---

1.  **`plt.figure()`**
    * **¿Para qué sirve?** Sirve para crear una nueva figura (una ventana o lienzo) donde se dibujarán uno o más gráficos. Es como el "marco" de tu obra de arte.
    * **Sintaxis:** `plt.figure(figsize=(ancho, alto))`
        * `figsize`: Una tupla que especifica el ancho y alto de la figura en pulgadas.
    * **Ejemplo:**
        ```python
        import matplotlib.pyplot as plt
        # Crea una figura con un tamaño de 10 pulgadas de ancho por 6 de alto
        plt.figure(figsize=(10, 6))
        # Después de esto, irían las funciones para dibujar los gráficos dentro de esta figura.
        # plt.show()
        ```

2.  **`plt.plot()`**
    * **¿Para qué sirve?** Sirve para dibujar gráficos de línea. Es ideal para mostrar tendencias o relaciones continuas entre dos variables.
    * **Sintaxis:** `plt.plot(x, y, color='...', linewidth=..., label='...')`
        * `x`: Los valores para el eje horizontal.
        * `y`: Los valores para el eje vertical.
        * `color`: El color de la línea (ej. `'red'`, `'blue'`, `'green'`).
        * `linewidth`: El grosor de la línea.
        * `label`: La etiqueta para la línea, que aparecerá en la leyenda.
    * **Ejemplo:**
        ```python
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.linspace(0, 10, 100) # 100 puntos entre 0 y 10
        y = np.sin(x) # Función seno
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, color='blue', linewidth=2, label='Función Seno')
        # plt.legend()
        # plt.show()
        ```

3.  **`plt.scatter()`**
    * **¿Para qué sirve?** Sirve para dibujar diagramas de dispersión, donde cada punto representa una observación. Es excelente para visualizar la relación entre dos variables numéricas y detectar patrones.
    * **Sintaxis:** `plt.scatter(x, y, s=..., c='...', alpha=..., label='...')`
        * `x`: Los valores para el eje horizontal.
        * `y`: Los valores para el eje vertical.
        * `s`: El tamaño de los puntos (ej. `50`).
        * `c`: El color de los puntos (ej. `'red'`). También puede ser un array de colores para cada punto.
        * `alpha`: La transparencia de los puntos (de 0.0 a 1.0). Útil para ver la densidad de puntos superpuestos.
        * `label`: La etiqueta para el grupo de puntos, que aparecerá en la leyenda.
    * **Ejemplo:**
        ```python
        import matplotlib.pyplot as plt
        import numpy as np
        x_puntos = np.random.rand(50) * 10
        y_puntos = np.random.rand(50) * 10
        plt.figure(figsize=(8, 6))
        plt.scatter(x_puntos, y_puntos, s=100, c='purple', alpha=0.7, label='Puntos Aleatorios')
        # plt.legend()
        # plt.show()
        ```

4.  **`plt.hist()`**
    * **¿Para qué sirve?** Sirve para crear histogramas. Los histogramas muestran la distribución de una variable numérica, dividiéndola en "bins" (barras) y contando cuántas observaciones caen en cada bin.
    * **Sintaxis:** `plt.hist(datos, bins=..., alpha=..., color='...', edgecolor='...')`
        * `datos`: El array de números cuya distribución quieres ver.
        * `bins`: El número de barras o los bordes de los bins.
        * `alpha`: La transparencia de las barras.
        * `color`: El color de las barras.
        * `edgecolor`: El color del borde de las barras (ej. `'black'` para que se distingan).
    * **Ejemplo:**
        ```python
        import matplotlib.pyplot as plt
        import numpy as np
        datos_edades = np.random.normal(loc=30, scale=5, size=100) # Edades simuladas
        plt.figure(figsize=(8, 5))
        plt.hist(datos_edades, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        # plt.show()
        ```

5.  **`plt.title()`**
    * **¿Para qué sirve?** Sirve para añadir un título principal al gráfico.
    * **Sintaxis:** `plt.title('Mi Título del Gráfico', fontsize=...)`
        * `fontsize`: El tamaño de la fuente del título.
    * **Ejemplo:**
        ```python
        import matplotlib.pyplot as plt
        # ... (código para crear un gráfico) ...
        plt.title('Distribución de Alturas de Personas', fontsize=16)
        # plt.show()
        ```

6.  **`plt.xlabel()`**
    * **¿Para qué sirve?** Sirve para añadir una etiqueta al eje horizontal (eje X) del gráfico.
    * **Sintaxis:** `plt.xlabel('Nombre del Eje X', fontsize=...)`
    * **Ejemplo:**
        ```python
        import matplotlib.pyplot as plt
        # ... (código para crear un gráfico) ...
        plt.xlabel('Altura (cm)', fontsize=12)
        # plt.show()
        ```

7.  **`plt.ylabel()`**
    * **¿Para qué sirve?** Sirve para añadir una etiqueta al eje vertical (eje Y) del gráfico.
    * **Sintaxis:** `plt.ylabel('Nombre del Eje Y', fontsize=...)`
    * **Ejemplo:**
        ```python
        import matplotlib.pyplot as plt
        # ... (código para crear un gráfico) ...
        plt.ylabel('Frecuencia', fontsize=12)
        # plt.show()
        ```

8.  **`plt.legend()`**
    * **¿Para qué sirve?** Sirve para mostrar una leyenda en el gráfico. La leyenda es un cuadro que explica qué significa cada línea, punto o barra, usando las `label` que definiste en `plt.plot()`, `plt.scatter()`, etc.
    * **Sintaxis:** `plt.legend(loc='...')`
        * `loc`: La ubicación de la leyenda (ej. `'upper right'`, `'lower left'`, `'best'`).
    * **Ejemplo:**
        ```python
        import matplotlib.pyplot as plt
        # ... (código para crear un gráfico con labels) ...
        plt.legend(loc='best') # Ubica la leyenda en la mejor posición automáticamente
        # plt.show()
        ```

9.  **`plt.grid()`**
    * **¿Para qué sirve?** Sirve para añadir una cuadrícula al gráfico, lo que facilita la lectura de los valores.
    * **Sintaxis:** `plt.grid(True)`
    * **Ejemplo:**
        ```python
        import matplotlib.pyplot as plt
        # ... (código para crear un gráfico) ...
        plt.grid(True)
        # plt.show()
        ```

10. **`plt.show()`**
    * **¿Para qué sirve?** ¡Es una de las funciones más importantes! Sirve para mostrar todos los gráficos que has creado. Si no llamas a `plt.show()`, los gráficos no aparecerán en tu pantalla o en tu notebook.
    * **Sintaxis:** `plt.show()`
    * **Ejemplo:**
        ```python
        import matplotlib.pyplot as plt
        # ... (todo el código para configurar tu gráfico) ...
        plt.show() # ¡Siempre al final para que se vea el gráfico!
        ```

11. **`plt.style.use()`**
    * **¿Para qué sirve?** Sirve para aplicar un estilo predefinido a todos tus gráficos. Matplotlib viene con varios estilos que cambian la apariencia (colores, fondos, tipografías) de tus gráficos de forma global.
    * **Sintaxis:** `plt.style.use('nombre_del_estilo')`
        * Un estilo común es `'seaborn-v0_8-darkgrid'` (que hace que los gráficos se parezcan a los de Seaborn con cuadrícula oscura). Otros incluyen `'ggplot'`, `'fivethirtyeight'`, `'dark_background'`, etc.
    * **Ejemplo:**
        ```python
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid') # Aplica este estilo a todos los gráficos siguientes
        # ... (código para crear gráficos) ...
        ```

---

¡Ahí lo tienes, mi pana! Con estas funciones de `matplotlib.pyplot`, ya puedes empezar a crear visualizaciones claras y atractivas de tus datos. La clave es ir probando y combinándolas.

¡Sigue así, que vas a dominar esto de la programación y el análisis de datos! ¡Cualquier otra cosa que necesites, me avisas!
