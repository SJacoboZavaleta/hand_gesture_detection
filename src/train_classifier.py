"""
Entrenamiento del Clasificador de Lenguaje de Señas
-------------------------------------------------
Este script entrena un modelo de Random Forest para clasificar gestos de lenguaje de señas.
Utiliza datos preprocesados de landmarks de manos generados previamente y guarda
el modelo entrenado para su uso posterior en la clasificación en tiempo real.

Proceso:
1. Carga de datos preprocesados
2. División en conjuntos de entrenamiento y prueba
3. Entrenamiento del modelo
4. Evaluación del rendimiento
5. Guardado del modelo
"""

# === IMPORTACIÓN DE BIBLIOTECAS ===
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from pathlib import Path

# === CARGA Y PREPARACIÓN DE DATOS ===
try:
    # Cargar datos preprocesados
    with open(Path(__file__).parent / 'data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    
    # Convertir a arrays numpy para mejor rendimiento
    data = np.asarray(data_dict['data'])     # Matriz de features (42 coordenadas normalizadas por muestra)
    labels = np.asarray(data_dict['labels'])  # Vector de etiquetas (A, B, L, etc.)
    
    # === DIVISIÓN DE DATOS ===
    # Separación en conjuntos de entrenamiento (80%) y prueba (20%)
    x_train, x_test, y_train, y_test = train_test_split(
        data, 
        labels, 
        test_size=0.2,        # 20% para pruebas
        shuffle=True,         # Mezclar datos aleatoriamente
        stratify=labels,      # Mantener proporción de clases
        random_state=42       # Semilla para reproducibilidad
    )

    # === ENTRENAMIENTO DEL MODELO ===
    # Inicialización del clasificador Random Forest
    model = RandomForestClassifier(
        n_estimators=100,     # Número de árboles en el bosque
        random_state=42,      # Semilla para reproducibilidad
        n_jobs=-1            # Usar todos los núcleos disponibles
    )

    # Entrenamiento con datos de entrenamiento
    print("Iniciando entrenamiento del modelo...")
    model.fit(x_train, y_train)

    # === EVALUACIÓN DEL MODELO ===
    # Predicciones en conjunto de prueba
    y_predict = model.predict(x_test)
    
    # Cálculo y muestra de precisión
    score = accuracy_score(y_predict, y_test)
    print('Precisión del modelo: {:.2f}%'.format(score * 100))

    # === GUARDADO DEL MODELO ===
    print("Guardando modelo entrenado...")
    model_path = Path(__file__).parent / 'model.p'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model}, f)

    print("Entrenamiento completado exitosamente!")

except FileNotFoundError:
    print("Error: No se encontró el archivo data.pickle")
except Exception as e:
    print(f"Error durante el entrenamiento: {e}")