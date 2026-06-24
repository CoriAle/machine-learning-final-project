# Clasificador Multi-Clase: Wine Dataset
### Proyecto Final · Machine Learning Aplicado
**Universidad Galileo (2026)**

---

## 📋 Descripción del Proyecto
Este proyecto consiste en el diseño, implementación y evaluación de un sistema de clasificación multi-clase utilizando el **Wine Dataset** de `scikit-learn`. El objetivo principal es construir un flujo de trabajo de Machine Learning robusto, sistemático y reproducible, comparando tres familias de algoritmos y combinándolas en un modelo de ensamble mediante voto mayoritario (*Majority Voting*) con remuestreo *Bootstrap*.

# Clasificador Multi-Clase (Wine Dataset)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CoriAle/machine-learning-final-project/blob/main/wine_dataset.ipynb)

* **Nombre:** Corina Alejandra Nimatuj Huitz
* **Carnet:** 2600943
* **Proyecto:** [Enunciado del Proyecto de Applied Machine Learning](https://llealgt.github.io/galileo_pia_applied_ml/proyecto/proyecto_clasificador_multiclase.html)

---

## 1. Elección del Dataset

### Dataset Seleccionado: Wine Dataset (scikit-learn internal)

**Justificación de la elección:**
Para el desarrollo de este proyecto se ha seleccionado el dataset **Wine**, fundamentando la decisión en los siguientes criterios técnicos y metodológicos:

1. **Garantía de Foco en el Modelado:** Al ser mi primera experiencia práctica en Machine Learning, este dataset tabular puramente numérico y libre de valores faltantes (NaNs) me permite concentrar el 100% del esfuerzo académico en el núcleo del proyecto: el diseño sistemático de experimentos, el diagnóstico preciso de subajuste/sobreajuste (*underfitting*/*overfitting*) y la lógica de combinación en un modelo de ensamble.
2. **Cumplimiento Estricto de Requisitos:** Cuenta con 3 clases balanceadas en la variable objetivo, cumpliendo con la restricción de clasificación multi-clase sin añadir la complejidad matemática y de visualización que implicarían datasets de mucho mayor tamaño.
3. **Eficiencia e Iteración Rápida:** Con 178 instancias y 13 características, la velocidad de cómputo es óptima. Esto me garantiza la viabilidad de ejecutar y registrar los 15 experimentos requeridos de forma ágil, permitiendo un análisis profundo del impacto de los hiperparámetros en las familias de Random Forest, XGBoost y Redes Neuronales sin comprometer los recursos de hardware.

---

## 2. Exploración del Dataset

A través del Análisis Exploratorio de Datos (EDA) se identificaron las siguientes características estructurales:

* **Dimensiones:** 178 instancias y 14 columnas (13 características + 1 variable objetivo).
* **Tipos de datos:**
  * `float64`: 13 columnas (todas las características numéricas).
  * `int64`: 1 columna (variable objetivo `target`).
* **Análisis de Valores Faltantes:** El dataset **no contiene valores faltantes** en ninguna de sus columnas, por lo que no requiere estrategias de imputación ni eliminación de registros nulos.

### Distribución de la Variable Objetivo
La variable objetivo (`target`) cuenta con 3 clases distribuidas de la siguiente manera:
* **`class_0`**: 59 elementos
* **`class_1`**: 71 elementos
* **`class_2`**: 48 elementos

---

## 3. Estructura de las Características

Las 13 variables numéricas predictoras incluidas en el conjunto de datos son:

* `alcohol`
* `malic_acid`
* `ash`
* `alcalinity_of_ash`
* `magnesium`
* `total_phenols`
* `flavanoids`
* `nonflavanoid_phenols`
* `proanthocyanins`
* `color_intensity`
* `hue`
* `od280/od315_of_diluted_wines`
* `proline`

---

## 4. Ingeniería de Características y Preprocesamiento

Para garantizar un rendimiento óptimo de los modelos y evitar sesgos debido a las diferentes escalas de los datos originales, se aplicó el siguiente pipeline de preprocesamiento:

* **División del Dataset:** Los datos se dividieron de forma estratificada para mantener la proporción de las clases en un **70% para entrenamiento** (124 muestras) y **30% para prueba** (54 muestras).
* **Escalamiento:** Se utilizó `StandardScaler` para normalizar las 13 características predictoras. Esto es fundamental para algoritmos sensibles a la escala como las Redes Neuronales, asegurando que cada característica tenga una media de 0 y una desviación estándar de 1.
* **Codificación:** Al ser una clasificación multi-clase, las etiquetas de las clases de entrenamiento y prueba se transformaron a un formato numérico adecuado (`0, 1, 2`), permitiendo una integración directa con los algoritmos tradicionales y las funciones de pérdida por entropía cruzada.

---

## 5. Diseño de Experimentos y Resultados

Se entrenaron y evaluaron **15 modelos en total**, distribuidos equitativamente en 5 experimentos por cada una de las 3 familias de algoritmos requeridas. La métrica principal de evaluación utilizada fue la **Exactitud (Accuracy)**.

### Tabla General de Experimentos

| Experimento | Familia de Algoritmo | Configuración / Hiperparámetros Clave | Accuracy Entrenamiento | Accuracy Prueba | Estado (Diagnóstico) |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | Random Forest | `n_estimators=10`, `max_depth=2` | 0.9677 | 0.9444 | Ajuste Óptimo / Ligero Overfitting |
| **2** | Random Forest | `n_estimators=50`, `max_depth=5` | 1.0000 | 0.9815 | Ligero Overfitting |
| **3** | Random Forest | `n_estimators=100`, `max_depth=10` | 1.0000 | 0.9815 | Ligero Overfitting |
| **4** | Random Forest | `n_estimators=5`, `max_depth=1` | 0.7903 | 0.7407 | **Underfitting** |
| **5** | Random Forest | `n_estimators=200`, `max_depth=None` | 1.0000 | 0.9815 | Ligero Overfitting |
| **6** | XGBoost | `n_estimators=10`, `max_depth=2`, `learning_rate=0.1` | 0.9839 | 0.9444 | Ajuste Óptimo |
| **7** | XGBoost | `n_estimators=50`, `max_depth=5`, `learning_rate=0.3` | 1.0000 | 0.9630 | Ligero Overfitting |
| **8** | XGBoost | `n_estimators=100`, `max_depth=3`, `learning_rate=0.01` | 0.9758 | 0.9259 | Ajuste Óptimo |
| **9** | XGBoost | `n_estimators=5`, `max_depth=1`, `learning_rate=0.05` | 0.8952 | 0.8889 | **Underfitting** |
| **10** | XGBoost | `n_estimators=150`, `max_depth=6`, `learning_rate=0.2` | 1.0000 | 0.9630 | Ligero Overfitting |
| **11** | Red Neuronal (MLP) | `hidden_layer_sizes=(10,)`, `max_iter=100` | 0.9194 | 0.9074 | Ajuste Óptimo (Convergencia temprana) |
| **12** | Red Neuronal (MLP) | `hidden_layer_sizes=(50, 25)`, `max_iter=500` | 1.0000 | 1.0000 | **Modelo Perfecto (Generalización Total)** |
| **13** | Red Neuronal (MLP) | `hidden_layer_sizes=(100, 50, 25)`, `max_iter=1000` | 1.0000 | 0.9815 | Ligero Overfitting |
| **14** | Red Neuronal (MLP) | `hidden_layer_sizes=(2,)`, `max_iter=50` | 0.4194 | 0.3704 | **Severe Underfitting** |
| **15** | Red Neuronal (MLP) | `hidden_layer_sizes=(100,)`, `max_iter=300`, `alpha=0.01` | 1.0000 | 0.9815 | Ligero Overfitting |

---

## 6. Selección de los 3 Mejores Modelos

Basado en la capacidad de generalización sobre el conjunto de datos de prueba y un balance óptimo para evitar el sobreajuste, se seleccionaron los siguientes mejores candidatos de cada familia:

1. **Mejor Random Forest (Experimento 2):** Con `n_estimators=50` y `max_depth=5`, logró un Accuracy de **98.15%** en prueba, demostrando una excelente capacidad de ensamble con árboles de decisión controlados.
2. **Mejor XGBoost (Experimento 6):** Con `n_estimators=10`, `max_depth=2` y un `learning_rate=0.1`, alcanzó un Accuracy de **94.44%** en prueba. Se prefirió esta configuración sobre las de mayor puntaje por ser un modelo mucho más simple, menos propenso al sobreajuste y altamente eficiente.
3. **Mejor Red Neuronal (Experimento 12):** Una arquitectura MLP con dos capas ocultas `(50, 25)` y `max_iter=500`. Logró un Accuracy perfecto de **100%** tanto en el conjunto de entrenamiento como en el de prueba, consolidándose como el modelo individual más robusto del proyecto.

---

## 7. Modelo de Ensamble y Evaluación Final

Para consolidar las predicciones y explotar las fortalezas individuales de cada familia algorítmica, se construyó un ensamble híbrido basado en votación mayoritaria blanda (**Voting Classifier - Soft Voting**), integrando los 3 mejores modelos descritos en la sección anterior.

### Reporte de Clasificación del Ensamble Final

```text
              precision    recall  f1-score   support

     Class 0       1.00      1.00      1.00        19
     Class 1       1.00      0.95      0.97        22
     Class 2       0.93      1.00      0.96        13

    accuracy                           0.98        54
   macro avg       0.98      0.98      0.98        54
weighted avg       0.98      0.98      0.98        54

```

# Tecnologías Utilizadas
- Python 3
- Google Colab
- Scikit-learn
- XGBoost Python API
- Pandas / NumPy
- Matplotlib / Seaborn (Visualización de matrices de confusión)

------------------------------------------------------------------------------
🍇 De los datos crudos al ensamble perfecto: la ciencia de datos también tiene su propio arte de cultivo.
¡Gracias por visitar este proyecto!