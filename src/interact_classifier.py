# Importación de bibliotecas necesarias
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Carga del modelo entrenado desde el archivo pickle
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Inicialización de la captura de video
cap = cv2.VideoCapture(0)

# Configuración de MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicialización del detector de manos
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Diccionario de etiquetas para mapear predicciones numéricas a letras
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

try:
    while True:
        # Arrays para almacenar las coordenadas de los puntos de referencia
        data_aux = []
        x_ = []
        y_ = []

        # Captura del frame actual
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame")
            break

        H, W, _ = frame.shape

        # Conversión del frame a RGB (requerido por MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesamiento de la imagen para detectar manos
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Dibujo de los puntos de referencia y conexiones de la mano
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  
                    hand_landmarks,  
                    mp_hands.HAND_CONNECTIONS,  
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Recolección de coordenadas x,y de todos los puntos de referencia
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalización de coordenadas respecto al punto mínimo
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            ## Verificar que tengamos exactamente 42 features antes de hacer la predicción
            if len(data_aux) == 42:
                # Cálculo de las coordenadas del rectángulo delimitador
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Realización de la predicción usando el modelo
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Dibujo del rectángulo y la letra predicha
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                           cv2.LINE_AA)
            else:
                print(f"Número incorrecto de features: {len(data_aux)} (se esperaban 42)")

        # Mostrar el frame procesado
        cv2.imshow('Sign Language Classifier', frame)

        # Esperar por la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error durante la ejecución: {e}")

finally:
    # Liberación de recursos
    print("Liberando recursos...")
    cap.release()
    cv2.destroyAllWindows()