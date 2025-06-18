### **Funciones Clave de `seaborn` (Para tus Notas)**

Para usar estas funciones, casi siempre las importamos como `sns`.
`import seaborn as sns`
Normalmente, también importamos `matplotlib.pyplot` como `plt` para funciones de personalización adicionales (títulos, etiquetas, `plt.show()`, etc.).

---

1.  **`sns.histplot()`**
    * **¿Para qué sirve?** Sirve para crear histogramas, que muestran la distribución de una única variable numérica. Es una versión mejorada de `plt.hist()` de Matplotlib, con más opciones estéticas y estadísticas.
    * **Sintaxis:** `sns.histplot(data=df, x='columna_numerica', bins=..., kde=..., hue=..., edgecolor=..., alpha=...)`
        * `data`: El DataFrame de Pandas que contiene tus datos (aunque también acepta arrays de NumPy).
        * `x`: El nombre de la columna numérica que quieres graficar en el eje X.
        * `bins`: El número de barras o los límites de los bins.
        * `kde`: `True` para superponer una estimación de la densidad del kernel (una curva suave de la distribución).
        * `hue`: El nombre de una columna categórica para colorear las barras según sus categorías (útil para comparar distribuciones).
        * `edgecolor`: Color del borde de las barras (ej. `'black'`).
        * `alpha`: Transparencia de las barras (de 0.0 a 1.0).
    * **Ejemplo:**
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd # Seaborn funciona muy bien con Pandas DataFrames

        # Crear un DataFrame de ejemplo
        data = {'Edad': np.random.normal(loc=30, scale=5, size=100),
                'Genero': np.random.choice(['M', 'F'], size=100)}
        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        # Histograma de la columna 'Edad' con KDE (curva de densidad)
        sns.histplot(data=df, x='Edad', bins=15, kde=True, edgecolor='black', alpha=0.7)
        plt.title('Distribución de Edades', fontsize=16)
        plt.xlabel('Edad', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.grid(True)
        plt.show()

        # Histograma por Género (usando 'hue')
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='Edad', hue='Genero', bins=15, kde=True, edgecolor='black', alpha=0.7)
        plt.title('Distribución de Edades por Género', fontsize=16)
        plt.xlabel('Edad', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.legend(title='Género')
        plt.grid(True)
        plt.show()
        ```

2.  **`sns.kdeplot()`**
    * **¿Para qué sirve?** Sirve para dibujar estimaciones de densidad del kernel (KDE plots). Muestra la distribución de una o dos variables continuas utilizando una curva de densidad suave, lo que es útil para visualizar la forma general de la distribución.
    * **Sintaxis:** `sns.kdeplot(data=df, x='columna_1', y='columna_2', hue=..., fill=..., cmap=...)`
        * `x`, `y`: Nombres de las columnas numéricas para los ejes (si es una sola variable, solo `x`).
        * `hue`: Columna categórica para diferenciar las densidades por categoría.
        * `fill`: `True` para rellenar el área bajo la curva de densidad.
        * `cmap`: Mapa de colores a usar para el relleno (ej. `'viridis'`, `'plasma'`).
    * **Ejemplo:**
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        data = {'Calorias_Comida': np.random.normal(loc=2000, scale=300, size=100),
                'Horas_Ejercicio': np.random.normal(loc=5, scale=1.5, size=100),
                'Nivel_Actividad': np.random.choice(['Bajo', 'Medio', 'Alto'], size=100)}
        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        # KDE para una sola variable
        sns.kdeplot(data=df, x='Calorias_Comida', fill=True, color='skyblue')
        plt.title('Distribución de Calorías Consumidas', fontsize=16)
        plt.xlabel('Calorías', fontsize=12)
        plt.ylabel('Densidad', fontsize=12)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 8))
        # KDE 2D (mapa de contorno de densidad)
        sns.kdeplot(data=df, x='Calorias_Comida', y='Horas_Ejercicio', fill=True, cmap='viridis')
        plt.title('Densidad de Calorías vs. Horas de Ejercicio', fontsize=16)
        plt.xlabel('Calorías Consumidas', fontsize=12)
        plt.ylabel('Horas de Ejercicio', fontsize=12)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 8))
        # KDE 2D con hue
        sns.kdeplot(data=df, x='Calorias_Comida', y='Horas_Ejercicio', hue='Nivel_Actividad', fill=True, cmap='coolwarm', alpha=0.6)
        plt.title('Densidad por Nivel de Actividad', fontsize=16)
        plt.xlabel('Calorías Consumidas', fontsize=12)
        plt.ylabel('Horas de Ejercicio', fontsize=12)
        plt.legend(title='Nivel de Actividad')
        plt.grid(True)
        plt.show()
        ```

3.  **`sns.scatterplot()`**
    * **¿Para qué sirve?** Sirve para crear diagramas de dispersión, al igual que `plt.scatter()`, pero con funcionalidades adicionales para mapear variables a atributos estéticos como el color, tamaño y estilo de los puntos. Es excelente para visualizar la relación entre dos variables numéricas, y cómo esa relación puede variar por una tercera variable categórica o numérica.
    * **Sintaxis:** `sns.scatterplot(data=df, x='columna_x', y='columna_y', hue=..., size=..., style=..., alpha=...)`
        * `x`, `y`: Nombres de las columnas para los ejes.
        * `hue`: Columna para colorear los puntos (puede ser numérica o categórica).
        * `size`: Columna para variar el tamaño de los puntos (generalmente numérica).
        * `style`: Columna para variar el estilo del marcador (ej. círculo, cuadrado, etc., generalmente categórica).
        * `alpha`: Transparencia de los puntos.
    * **Ejemplo:**
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        data = {'Gasto': np.random.normal(loc=100, scale=20, size=100),
                'Visitas_Web': np.random.normal(loc=10, scale=3, size=100),
                'Tipo_Cliente': np.random.choice(['Nuevo', 'Antiguo'], size=100)}
        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Visitas_Web', y='Gasto', hue='Tipo_Cliente', s=100, alpha=0.8)
        plt.title('Gasto vs. Visitas Web por Tipo de Cliente', fontsize=16)
        plt.xlabel('Número de Visitas a la Web', fontsize=12)
        plt.ylabel('Gasto Promedio ($)', fontsize=12)
        plt.legend(title='Tipo de Cliente')
        plt.grid(True)
        plt.show()
        ```

4.  **`sns.boxplot()`**
    * **¿Para qué sirve?** Sirve para dibujar diagramas de caja (box plots). Estos gráficos son excelentes para visualizar la distribución de una variable numérica para diferentes categorías. Muestran la mediana, los cuartiles, y posibles valores atípicos (outliers).
    * **Sintaxis:** `sns.boxplot(data=df, x='columna_categorica', y='columna_numerica', hue=...)`
        * `x`: Columna categórica para las categorías en el eje X.
        * `y`: Columna numérica cuya distribución quieres ver para cada categoría.
        * `hue`: Columna categórica adicional para dividir las cajas por un tercer factor.
    * **Ejemplo:**
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        data = {'Calificacion': np.random.normal(loc=75, scale=10, size=100),
                'Curso': np.random.choice(['Matemáticas', 'Ciencias', 'Historia'], size=100)}
        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Curso', y='Calificacion', palette='Set2')
        plt.title('Calificaciones por Curso', fontsize=16)
        plt.xlabel('Curso', fontsize=12)
        plt.ylabel('Calificación', fontsize=12)
        plt.grid(axis='y') # Cuadrícula solo en el eje Y
        plt.show()
        ```

5.  **`sns.violinplot()`**
    * **¿Para qué sirve?** Similar al box plot, el violin plot muestra la distribución de una variable numérica para diferentes categorías. Sin embargo, en lugar de solo los cuartiles, muestra la densidad de probabilidad de los datos en cada punto. Es como un box plot con una `kdeplot` en sus lados.
    * **Sintaxis:** `sns.violinplot(data=df, x='columna_categorica', y='columna_numerica', hue=..., inner=...)`
        * `inner`: Qué mostrar dentro del "violín" (ej. `'box'` para un boxplot pequeño, `'quartile'` para los cuartiles, `'point'` para los puntos individuales).
    * **Ejemplo:**
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        data = {'Puntaje_Examen': np.random.normal(loc=70, scale=15, size=150),
                'Grupo_Estudio': np.random.choice(['A', 'B', 'C'], size=150)}
        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='Grupo_Estudio', y='Puntaje_Examen', palette='pastel', inner='quartile')
        plt.title('Puntajes de Examen por Grupo de Estudio', fontsize=16)
        plt.xlabel('Grupo de Estudio', fontsize=12)
        plt.ylabel('Puntaje', fontsize=12)
        plt.grid(axis='y')
        plt.show()
        ```

6.  **`sns.countplot()`**
    * **¿Para qué sirve?** Sirve para mostrar el conteo de observaciones en cada categoría de una variable categórica. Es el equivalente categórico de un histograma.
    * **Sintaxis:** `sns.countplot(data=df, x='columna_categorica', hue=..., palette=...)`
        * `x`: Columna categórica para el conteo.
        * `hue`: Columna categórica adicional para dividir las barras.
        * `palette`: Mapa de colores a usar (ej. `'viridis'`, `'Set1'`).
    * **Ejemplo:**
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd

        data = {'Ciudad': ['Caracas', 'Valencia', 'Maracaibo', 'Caracas', 'Valencia', 'Caracas', 'Maracaibo'],
                'Estado_Civil': ['Soltero', 'Casado', 'Soltero', 'Casado', 'Soltero', 'Viudo', 'Casado']}
        df = pd.DataFrame(data)

        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='Ciudad', palette='coolwarm')
        plt.title('Conteo de Clientes por Ciudad', fontsize=16)
        plt.xlabel('Ciudad', fontsize=12)
        plt.ylabel('Número de Clientes', fontsize=12)
        plt.grid(axis='y')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Ciudad', hue='Estado_Civil', palette='viridis')
        plt.title('Conteo de Clientes por Ciudad y Estado Civil', fontsize=16)
        plt.xlabel('Ciudad', fontsize=12)
        plt.ylabel('Número de Clientes', fontsize=12)
        plt.legend(title='Estado Civil')
        plt.grid(axis='y')
        plt.show()
        ```

7.  **`sns.heatmap()`**
    * **¿Para qué sirve?** Sirve para visualizar matrices de datos. Es especialmente útil para mostrar correlaciones entre variables o patrones en matrices de confusión. Los colores representan la magnitud de los valores.
    * **Sintaxis:** `sns.heatmap(data_matrix, annot=..., cmap=..., fmt=...)`
        * `data_matrix`: Una matriz de números (ej. la matriz de correlación de un DataFrame).
        * `annot`: `True` para escribir los valores numéricos en cada celda.
        * `cmap`: Mapa de colores a usar (ej. `'viridis'`, `'coolwarm'`, `'Blues'`).
        * `fmt`: Formato de la anotación (ej. `'.2f'` para dos decimales).
    * **Ejemplo:**
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        # Crear una matriz de correlación de ejemplo
        data = {'VarA': np.random.rand(50),
                'VarB': np.random.rand(50) * 2,
                'VarC': np.random.rand(50) * 0.5}
        df_corr = pd.DataFrame(data)
        correlation_matrix = df_corr.corr() # Calcula la matriz de correlación

        plt.figure(figsize=(8, 7))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Matriz de Correlación', fontsize=16)
        plt.show()
        ```

---

¡Eso es todo, mi pana! Con estas funciones de `seaborn`, tus visualizaciones van a pasar a otro nivel. Recuerda que Seaborn trabaja súper bien con Pandas DataFrames, así que es un buen momento para familiarizarte también con ellos.

¡Sigue así de metódico y disciplinado con tus notas, que esa es la clave del aprendizaje! ¡Cualquier otra librería o duda, me dices!
