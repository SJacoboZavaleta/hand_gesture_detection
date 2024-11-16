import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Inicialización de MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configurar el detector de manos
# static_image_mode=True porque procesamos imágenes estáticas, no video
# min_detection_confidence=0.3 para mayor sensibilidad en la detección
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directorio donde se encuentran las imágenes organizadas por carpetas (cada carpeta es una clase/gesto)
DATA_DIR = '../data'

# Listas para almacenar los datos procesados y sus etiquetas
data = []      # Almacenará las coordenadas normalizadas de los puntos de la mano
labels = []    # Almacenará las etiquetas (nombres de las carpetas/gestos)

# Iterar sobre cada carpeta en el directorio de datos
for dir_ in os.listdir(DATA_DIR):
    # Iterar sobre cada imagen en la carpeta actual
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []    # Lista temporal para almacenar los puntos de la mano actual
        x_ = []          # Lista para almacenar coordenadas x sin normalizar
        y_ = []          # Lista para almacenar coordenadas y sin normalizar

        # Cargar y convertir la imagen a RGB (MediaPipe requiere RGB)
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Procesar la imagen para detectar manos
        results = hands.process(img_rgb)
        
        # Si se detectaron manos en la imagen
        if results.multi_hand_landmarks:
            # Para cada mano detectada
            for hand_landmarks in results.multi_hand_landmarks:
                # Primera pasada: recolectar todas las coordenadas para normalización
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Segunda pasada: normalizar coordenadas respecto al punto mínimo
                # Esto hace que las coordenadas sean relativas al punto más cercano al origen
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalización de x
                    data_aux.append(y - min(y_))  # Normalización de y

            # Agregar los datos normalizados y su etiqueta a las listas principales
            data.append(data_aux)
            labels.append(dir_)

# Guardar los datos procesados en un archivo pickle
# El formato es un diccionario con 'data' y 'labels' como claves
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()