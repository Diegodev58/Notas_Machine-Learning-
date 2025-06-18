### **Clases y Funciones Clave de `scikit-learn` (sklearn) (Para tus Notas)**

Scikit-learn es la librería principal para Machine Learning en Python. A diferencia de NumPy o Matplotlib que tienen muchas funciones globales, Scikit-learn se organiza en **módulos** y dentro de ellos, se suelen importar **clases** que luego instanciamos (creamos un objeto) y entrenamos.

Para usar estas funciones/clases, las importamos de sus módulos específicos:
`from sklearn.modulo import ClaseOFuncion`

---

### **1. `sklearn.model_selection` (Para dividir datos y evaluar modelos)**

Este módulo es crucial para preparar tus datos antes de entrenar y probar un modelo.

* **`train_test_split()`**
    * **¿Para qué sirve?** Sirve para dividir un conjunto de datos en dos subconjuntos: uno para **entrenamiento** (`train`) y otro para **prueba** (`test`). Esto es fundamental para evaluar qué tan bien generaliza tu modelo a datos nuevos.
    * **Sintaxis:** `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=..., random_state=...)`
        * `X`: Las características (variables independientes) de tu dataset.
        * `y`: Las etiquetas o valores objetivo (variable dependiente) de tu dataset.
        * `test_size`: La proporción del dataset que se usará para el conjunto de prueba (ej. `0.2` para 20%).
        * `random_state`: Un número entero para asegurar que la división de los datos sea la misma cada vez que ejecutes el código (para reproducibilidad).
    * **Ejemplo:**
        ```python
        from sklearn.model_selection import train_test_split
        import numpy as np

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]) # 5 muestras, 2 características
        y = np.array([0, 1, 0, 1, 0]) # 5 etiquetas

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        print(f"X_train:\n{X_train}") # 3 muestras
        print(f"X_test:\n{X_test}")   # 2 muestras
        print(f"y_train: {y_train}")
        print(f"y_test: {y_test}")
        ```

### **2. `sklearn.linear_model` (Para modelos lineales)**

Contiene los algoritmos basados en relaciones lineales entre las variables.

* **`LinearRegression()`**
    * **¿Para qué sirve?** Es la clase que implementa la Regresión Lineal. Sirve para modelar la relación lineal entre una variable dependiente continua y una o más variables independientes.
    * **Sintaxis (uso básico):**
        ```python
        from sklearn.linear_model import LinearRegression

        modelo_regresion = LinearRegression() # Crea una instancia del modelo
        modelo_regresion.fit(X_train, y_train) # Entrena el modelo con los datos de entrenamiento
        predicciones = modelo_regresion.predict(X_test) # Realiza predicciones
        ```
    * **Atributos importantes después de `fit()`:**
        * `.coef_`: Los coeficientes (pendientes) de las variables independientes.
        * `.intercept_`: El término de intercepción (el valor de Y cuando X es 0).
    * **Ejemplo:**
        ```python
        from sklearn.linear_model import LinearRegression
        import numpy as np

        X_train = np.array([[1], [2], [3], [4]]) # Horas de estudio
        y_train = np.array([10, 20, 30, 40])     # Calificación

        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        print(f"Coeficiente (m): {modelo.coef_[0]:.2f}")     # Debería ser ~10
        print(f"Intercepto (b): {modelo.intercept_:.2f}") # Debería ser ~0
        print(f"Predicción para 5 horas: {modelo.predict([[5]])[0]:.2f}") # Predice una calificación
        ```

### **3. `sklearn.preprocessing` (Para transformar los datos)**

Este módulo contiene herramientas para preparar tus datos, como escalarlos, transformar características, etc.

* **`PolynomialFeatures()`**
    * **¿Para qué sirve?** Sirve para generar características polinómicas (potencias de las características originales e interacciones entre ellas). Esto es lo que permite a la `LinearRegression` modelar relaciones no lineales (Regresión Polinómica).
    * **Sintaxis:**
        ```python
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(degree=grado) # Crea un transformador
        X_poly = poly.fit_transform(X_original) # Transforma tus datos
        ```
        * `degree`: El grado del polinomio (ej. `2` para $X^2$, `3` para $X^3$).
    * **Ejemplo:**
        ```python
        from sklearn.preprocessing import PolynomialFeatures
        import numpy as np

        X_original = np.array([[2], [4], [6]]) # Datos originales (una característica)
        # Queremos X y X^2
        poly_transformer = PolynomialFeatures(degree=2)
        X_transformado = poly_transformer.fit_transform(X_original)
        print(f"X original:\n{X_original.flatten()}")
        print(f"X transformado (1, X, X^2):\n{X_transformado}")
        # Salida: [[1.  2.  4.]  [1.  4. 16.]  [1.  6. 36.]] (1 es para el intercepto)
        ```

### **4. `sklearn.metrics` (Para evaluar el rendimiento del modelo)**

Contiene una amplia variedad de funciones para medir qué tan bien se desempeña tu modelo.

* **`mean_squared_error()`**
    * **¿Para qué sirve?** Sirve para calcular el Error Cuadrático Medio (MSE), una métrica común para evaluar modelos de regresión. Mide el promedio de los cuadrados de los errores (la diferencia entre los valores predichos y los reales). Un valor menor indica un mejor ajuste.
    * **Sintaxis:** `mean_squared_error(y_verdadero, y_predicho)`
        * `y_verdadero`: Los valores reales de la variable dependiente.
        * `y_predicho`: Los valores que tu modelo predijo.
    * **Ejemplo:**
        ```python
        from sklearn.metrics import mean_squared_error
        import numpy as np

        y_real = np.array([10, 20, 30])
        y_pred = np.array([11, 19, 32])
        mse = mean_squared_error(y_real, y_pred)
        print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
        # Salida: MSE: 2.00
        ```

* **`r2_score()`**
    * **¿Para qué sirve?** Sirve para calcular el coeficiente de determinación (R-cuadrado), otra métrica para modelos de regresión. Indica la proporción de la varianza en la variable dependiente que es predecible a partir de las variables independientes. Un valor más cercano a 1.0 es mejor (1.0 significa que el modelo explica toda la variabilidad).
    * **Sintaxis:** `r2_score(y_verdadero, y_predicho)`
    * **Ejemplo:**
        ```python
        from sklearn.metrics import r2_score
        import numpy as np

        y_real = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([10.5, 19.8, 30.2, 39.5, 49.9])
        r2 = r2_score(y_real, y_pred)
        print(f"Coeficiente R^2: {r2:.2f}")
        # Salida: R^2: 1.00 (casi perfecto)
        ```

* **`accuracy_score()`**
    * **¿Para qué sirve?** Sirve para calcular la precisión (accuracy) en problemas de clasificación. Es la proporción de predicciones correctas sobre el total de predicciones.
    * **Sintaxis:** `accuracy_score(y_verdadero, y_predicho)`
    * **Ejemplo:**
        ```python
        from sklearn.metrics import accuracy_score
        import numpy as np

        y_real_clases = np.array([0, 1, 0, 1, 0])
        y_pred_clases = np.array([0, 1, 1, 1, 0]) # Una predicción incorrecta
        acc = accuracy_score(y_real_clases, y_pred_clases)
        print(f"Precisión (Accuracy): {acc:.2f}")
        # Salida: Precisión (Accuracy): 0.80 (4 de 5 correctas)
        ```

* **`classification_report()`**
    * **¿Para qué sirve?** Sirve para generar un reporte de texto con las métricas de clasificación más importantes (precisión, recall, f1-score y soporte) para cada clase. Es muy útil para tener una visión detallada del rendimiento del clasificador.
    * **Sintaxis:** `classification_report(y_verdadero, y_predicho, target_names=...)`
        * `target_names`: Una lista de los nombres de las clases (ej. `['No Spam', 'Spam']`) para que el reporte sea más legible.
    * **Ejemplo:**
        ```python
        from sklearn.metrics import classification_report
        import numpy as np

        y_real = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0]) # 2 errores
        nombres_clases = ['Clase A', 'Clase B']
        print(classification_report(y_real, y_pred, target_names=nombres_clases))
        ```

### **5. `sklearn.svm` (Para Máquinas de Vectores de Soporte)**

Contiene las implementaciones de los algoritmos de Máquinas de Vectores de Soporte.

* **`SVC()`**
    * **¿Para qué sirve?** Es la clase para el clasificador de Máquinas de Vectores de Soporte. Utiliza la idea de encontrar el hiperplano óptimo que maximiza el margen entre clases, y es muy potente para problemas de clasificación, especialmente con el "truco del kernel".
    * **Sintaxis (uso básico):**
        ```python
        from sklearn.svm import SVC

        modelo_svm = SVC(kernel='rbf', C=1.0, gamma='scale') # Crea una instancia
        modelo_svm.fit(X_train, y_train) # Entrena
        predicciones = modelo_svm.predict(X_test) # Predice
        ```
        * `kernel`: El tipo de función kernel a usar (`'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`). `'rbf'` es muy versátil.
        * `C`: Parámetro de regularización. Un `C` bajo significa un margen más amplio (más tolerante a errores de clasificación); un `C` alto significa un margen más estrecho (menos tolerante a errores).
        * `gamma`: Parámetro del kernel RBF. Define la influencia de un solo punto de entrenamiento. `'scale'` es una buena opción por defecto.
    * **Ejemplo:**
        ```python
        from sklearn.svm import SVC
        from sklearn.datasets import make_circles
        from sklearn.model_selection import train_test_split

        X, y = make_circles(n_samples=100, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        modelo_svm_rbf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        modelo_svm_rbf.fit(X_train, y_train)
        predicciones = modelo_svm_rbf.predict(X_test)
        print(f"Primeras 5 predicciones SVM: {predicciones[:5]}")
        ```

### **6. `sklearn.tree` (Para Árboles de Decisión)**

Contiene las implementaciones de los algoritmos basados en árboles de decisión.

* **`DecisionTreeClassifier()`**
    * **¿Para qué sirve?** Es la clase que implementa el algoritmo de Árbol de Decisión para problemas de clasificación. Crea un modelo en forma de diagrama de flujo que toma decisiones basadas en características.
    * **Sintaxis (uso básico):**
        ```python
        from sklearn.tree import DecisionTreeClassifier

        arbol_decision = DecisionTreeClassifier(max_depth=..., random_state=...) # Crea una instancia
        arbol_decision.fit(X_train, y_train) # Entrena
        predicciones = arbol_decision.predict(X_test) # Predice
        ```
        * `max_depth`: La profundidad máxima del árbol. Limitarla es importante para prevenir el overfitting.
        * `random_state`: Para asegurar la reproducibilidad del árbol.
    * **Ejemplo:**
        ```python
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        modelo_arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
        modelo_arbol.fit(X_train, y_train)
        predicciones = modelo_arbol.predict(X_test)
        print(f"Primeras 5 predicciones del Árbol: {predicciones[:5]}")
        ```

* **`plot_tree()`**
    * **¿Para qué sirve?** Es una función de ayuda para visualizar directamente el árbol de decisión entrenado usando Matplotlib.
    * **Sintaxis:** `plot_tree(modelo_arbol, feature_names=..., class_names=..., filled=..., rounded=...)`
        * `modelo_arbol`: La instancia del `DecisionTreeClassifier` ya entrenada.
        * `feature_names`: Nombres de las características para etiquetas más claras.
        * `class_names`: Nombres de las clases para etiquetas más claras.
        * `filled`: `True` para rellenar los nodos con colores de clase.
        * `rounded`: `True` para esquinas redondeadas.
    * **Ejemplo:**
        ```python
        # Este ejemplo asume que ya entrenaste 'modelo_arbol' como en el ejemplo anterior
        from sklearn.tree import plot_tree
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plot_tree(modelo_arbol, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
        plt.show()
        ```

### **7. `sklearn.cluster` (Para Clustering)**

Contiene los algoritmos de agrupación para Aprendizaje No Supervisado.

* **`KMeans()`**
    * **¿Para qué sirve?** Es la clase que implementa el algoritmo K-Means para clustering. Su objetivo es agrupar un conjunto de $N$ observaciones en $K$ clusters, donde cada observación pertenece al cluster cuyo centroide es el más cercano.
    * **Sintaxis (uso básico):**
        ```python
        from sklearn.cluster import KMeans

        kmeans_model = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=...) # Crea una instancia
        kmeans_model.fit(X_datos_sin_etiquetas) # Entrena (solo con X)
        etiquetas_clusters = kmeans_model.labels_ # Obtiene las etiquetas de cluster asignadas
        centroides = kmeans_model.cluster_centers_ # Obtiene las coordenadas de los centroides
        ```
        * `n_clusters`: El número `K` de clusters que se espera encontrar.
        * `init='k-means++'`: Método para inicializar los centroides de forma inteligente.
        * `n_init`: Número de veces que se ejecutará el algoritmo K-Means con diferentes semillas de centroides. El mejor resultado se toma como salida final.
        * `random_state`: Para reproducibilidad.
    * **Atributos importantes después de `fit()`:**
        * `.labels_`: Un array que indica a qué cluster pertenece cada punto de datos.
        * `.cluster_centers_`: Las coordenadas (valores de las características) de los centroides de cada cluster.
        * `.inertia_`: La suma de los cuadrados de las distancias de las muestras a su centro de cluster más cercano (usado para el Método del Codo).
    * **Ejemplo:**
        ```python
        from sklearn.cluster import KMeans
        from sklearn.datasets import make_blobs
        import matplotlib.pyplot as plt
        import seaborn as sns

        X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.8, random_state=42)

        kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X)

        cluster_labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=cluster_labels, palette='viridis', s=80, alpha=0.8)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', s=200, color='red', label='Centroides', edgecolors='black')
        plt.title('Clusters de K-Means', fontsize=16)
        plt.xlabel('Característica 1', fontsize=12)
        plt.ylabel('Característica 2', fontsize=12)
        plt.legend()
        plt.show()
        ```

### **8. `sklearn.datasets` (Para cargar datasets de ejemplo)**

Contiene funciones para cargar y generar datasets de juguete (pequeños conjuntos de datos) que son útiles para practicar y probar algoritmos.

* **`load_iris()`**
    * **¿Para qué sirve?** Carga el famoso dataset de flores Iris. Es un clásico para problemas de clasificación con 3 clases y 4 características numéricas.
    * **Sintaxis:** `dataset = load_iris()`
    * **Ejemplo:**
        ```python
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris.data         # Características
        y = iris.target       # Etiquetas de clase
        feature_names = iris.feature_names # Nombres de las características
        target_names = iris.target_names   # Nombres de las clases
        print(f"Características de Iris (primeros 2):\n{X[:2]}")
        print(f"Clases de Iris (primeras 2): {y[:2]}")
        ```

* **`make_circles()`**
    * **¿Para qué sirve?** Genera un dataset de 2D con forma de círculos concéntricos. Es excelente para demostrar y probar algoritmos que pueden encontrar fronteras de decisión no lineales (como SVM con kernel RBF).
    * **Sintaxis:** `X, y = make_circles(n_samples=..., factor=..., noise=..., random_state=...)`
        * `n_samples`: Número total de puntos.
        * `factor`: Escala entre el radio del círculo interior y el exterior.
        * `noise`: Cantidad de ruido añadida a los datos.
    * **Ejemplo:**
        ```python
        from sklearn.datasets import make_circles
        import matplotlib.pyplot as plt
        import seaborn as sns

        X, y = make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=42)
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm')
        plt.title('Datos de Círculos')
        plt.show()
        ```

* **`make_blobs()`**
    * **¿Para qué sirve?** Genera un dataset de puntos agrupados en "bloques" (clusters) alrededor de centros predefinidos. Es muy útil para probar algoritmos de clustering.
    * **Sintaxis:** `X, y = make_blobs(n_samples=..., centers=..., cluster_std=..., random_state=...)`
        * `n_samples`: Número total de puntos.
        * `centers`: Número de centros a generar.
        * `cluster_std`: Desviación estándar de los clusters (qué tan dispersos están los puntos en cada cluster).
    * **Ejemplo:**
        ```python
        from sklearn.datasets import make_blobs
        import matplotlib.pyplot as plt
        import seaborn as sns

        X, y_true = make_blobs(n_samples=200, centers=4, cluster_std=0.7, random_state=42)
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_true, palette='viridis') # y_true son los clusters "reales"
        plt.title('Datos de Blobs')
        plt.show()
        ```

---

¡Ahí lo tienes, mi pana! Con este desglose de Scikit-learn, tienes una guía completa de las clases y funciones más esenciales para empezar a construir y evaluar tus modelos de Machine Learning.

Recuerda que el flujo general en Scikit-learn para los modelos supervisados es:
1.  **Importar** la clase del modelo.
2.  **Instanciar** el modelo (`modelo = ClaseDelModelo(...)`).
3.  **Entrenar** el modelo (`modelo.fit(X_train, y_train)`).
4.  **Predecir** con el modelo (`predicciones = modelo.predict(X_test)`).
5.  **Evaluar** las predicciones (`metrica(y_test, predicciones)`).

¡Sigue así con tu disciplina de notas, que esa es la mejor forma de aprender y retener todo este conocimiento! ¡Cualquier otra cosa que necesites, me dices!
