"""
Clasificador de Lenguaje de Señas en Tiempo Real
----------------------------------------------
Este script implementa un clasificador de gestos de lenguaje de señas usando la cámara web.
Utiliza MediaPipe para la detección de manos y un modelo pre-entrenado para clasificar
los gestos en letras del alfabeto (A, B, L).

Características principales:
- Detección de manos en tiempo real
- Clasificación de gestos
- Visualización de resultados con OpenCV
- Salir del programa presionando 'q'
"""

# Importación de bibliotecas necesarias
import pickle
import cv2
import mediapipe as mp
import numpy as np

# === INICIALIZACIÓN ===
# Carga del modelo pre-entrenado
model_dict = pickle.load(open('../model.p', 'rb'))
model = model_dict['model']

# Configuración de la captura de video
cap = cv2.VideoCapture(0)

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicialización del detector de manos
# static_image_mode=False para optimizar detección en video
# min_detection_confidence=0.3 para balance entre precisión y velocidad
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Mapeo de predicciones numéricas a letras
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# === VERIFICACIÓN DE CÁMARA ===
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

# === BUCLE PRINCIPAL DE PROCESAMIENTO ===
try:
    while True:
        # Inicialización de arrays para coordenadas
        data_aux = []  # Almacenará coordenadas normalizadas
        x_ = []       # Coordenadas x para normalización
        y_ = []       # Coordenadas y para normalización

        # Captura y verificación del frame
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame")
            break

        H, W, _ = frame.shape  # Dimensiones del frame

        # Preprocesamiento de imagen para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # === PROCESAMIENTO DE MANOS DETECTADAS ===
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Visualización de puntos y conexiones de la mano
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,  
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Primera pasada: recolección de coordenadas
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Segunda pasada: normalización de coordenadas
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # === PREDICCIÓN Y VISUALIZACIÓN ===
            if len(data_aux) == 42:  # Verificación de dimensionalidad correcta
                # Cálculo del rectángulo delimitador
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Predicción y visualización
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Dibujo de resultados en el frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                           cv2.LINE_AA)
            else:
                print(f"Número incorrecto de features: {len(data_aux)} (se esperaban 42)")

        # Mostrar resultado
        cv2.imshow('Sign Language Classifier', frame)

        # Verificar si se debe salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error durante la ejecución: {e}")

finally:
    # === LIMPIEZA Y CIERRE ===
    print("Liberando recursos...")
    cap.release()
    cv2.destroyAllWindows()