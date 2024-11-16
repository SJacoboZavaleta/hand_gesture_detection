# Importación de bibliotecas necesarias
import pickle
from sklearn.ensemble import RandomForestClassifier  # Clasificador de bosque aleatorio
from sklearn.model_selection import train_test_split  # Para dividir datos en entrenamiento y prueba
from sklearn.metrics import accuracy_score  # Para evaluar la precisión del modelo
import numpy as np

# Carga de datos desde el archivo pickle que contiene las características extraídas
# El archivo data.pickle debe contener un diccionario con 'data' (características) y 'labels' (etiquetas)
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Conversión de los datos y etiquetas a arrays de numpy para su procesamiento
data = np.asarray(data_dict['data'])     # Matriz de características normalizadas de las manos
labels = np.asarray(data_dict['labels'])  # Vector de etiquetas correspondientes

# División de datos en conjuntos de entrenamiento y prueba
# test_size=0.2: 80% para entrenamiento, 20% para prueba
# shuffle=True: mezcla aleatoria de los datos
# stratify=labels: mantiene la proporción de clases en ambos conjuntos
x_train, x_test, y_train, y_test = train_test_split(data, labels, 
                                                   test_size=0.2, 
                                                   shuffle=True, 
                                                   stratify=labels)

# Inicialización del modelo RandomForest con una semilla aleatoria para reproducibilidad
model = RandomForestClassifier(random_state=42)

# Entrenamiento del modelo con los datos de entrenamiento
model.fit(x_train, y_train)

# Realización de predicciones sobre el conjunto de prueba
y_predict = model.predict(x_test)

# Cálculo de la precisión del modelo
score = accuracy_score(y_predict, y_test)

# Impresión del resultado de la evaluación
print('{}% de las muestras fueron clasificadas correctamente!'.format(score * 100))

# Guardado del modelo entrenado en un archivo pickle
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()