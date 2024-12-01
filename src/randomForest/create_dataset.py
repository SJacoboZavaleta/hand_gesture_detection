"""
Crea un dataset de entrenamiento para el clasificador de lenguaje de señas.

Este script procesa las imágenes captiuradas previamente desde la cámara web para crear un dataset de entrenamiento
para el clasificador de lenguaje de señas. Las imágenes se organizan en carpetas numeradas (0, 1, 2) correspondientes a diferentes gestos.

Input:
- Las imágenes capturadas previamente desde la cámara web (doro, natalia, sergio y fergus) fueron combinadas en una carpeta llamada `webcam_data`.
- Carpetas de las imagenes: 0, 1, 2, ... correspondientes a diferentes gestos (a, b y c, ...).  

Output:
- Un dataset de entrenamiento con las imágenes procesadas y las etiquetas correspondientes.	
- Las etiquetas correspondientes a cada imagen se obtienen de las carpetas de las imagenes.

Funcionamiento:
1. Procesa las imágenes capturadas previamente desde la cámara web.
2. Extrae los landmarks de las manos detectadas en las imágenes.
3. Normaliza las coordenadas de los landmarks.
4. Crea un dataset de entrenamiento con las imágenes procesadas y las etiquetas correspondientes.
5. Guarda el dataset en un archivo .pickle.

Funciones:
1. process_hand_landmarks: Procesa los landmarks de una mano y devuelve las coordenadas normalizadas.
2. create_dataset: Crea un dataset de entrenamiento con las imágenes procesadas y las etiquetas correspondientes.
3. save_dataset: Guarda el dataset en unified_webcam_dataset_info.pickle.
"""

import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_hand_landmarks(hand_landmarks):
    """
    Procesa los landmarks de una mano y devuelve las coordenadas normalizadas.
    Asegura que siempre devuelve un vector de 42 características (21 puntos * 2 coordenadas).
    """
    data_aux = []
    x_ = []
    y_ = []
    
    # Asegurarse de que tenemos exactamente 21 landmarks
    if len(hand_landmarks.landmark) != 21:
        return None
    
    # Primera pasada: recolectar todas las coordenadas
    for landmark in hand_landmarks.landmark:
        x_.append(landmark.x)
        y_.append(landmark.y)
    
    # Segunda pasada: normalizar coordenadas
    for landmark in hand_landmarks.landmark:
        data_aux.append(landmark.x - min(x_))
        data_aux.append(landmark.y - min(y_))
    
    # Verificar que tenemos exactamente 42 características
    if len(data_aux) != 42:
        return None
        
    return data_aux

def create_dataset():
    # Inicialización de MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,  # Solo detectar una mano
        min_detection_confidence=0.3
    )

    # Directorio de datos
    DATA_DIR = str(Path(__file__).parent.parent.parent / 'data' / 'webcam_data' / 'unified_data')
    
    data = []
    labels = []
    skipped_images = 0
    processed_images = 0
    
    # Obtener total de imágenes para la barra de progreso
    total_images = sum(len(os.listdir(os.path.join(DATA_DIR, dir_))) for dir_ in os.listdir(DATA_DIR))
    
    print(f"Iniciando procesamiento de {total_images} imágenes...")
    
    # Barra de progreso principal
    with tqdm(total=total_images, desc="Procesando imágenes") as pbar:
        for dir_ in sorted(os.listdir(DATA_DIR)):  # Ordenar para consistencia
            for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
                # Cargar y procesar imagen
                img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
                if img is None:
                    print(f"Error al cargar imagen: {img_path}")
                    skipped_images += 1
                    pbar.update(1)
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    # Tomar solo la primera mano detectada
                    hand_landmarks = results.multi_hand_landmarks[0]
                    processed_data = process_hand_landmarks(hand_landmarks)
                    
                    if processed_data is not None:
                        data.append(processed_data)
                        labels.append(dir_)
                        processed_images += 1
                    else:
                        skipped_images += 1
                else:
                    skipped_images += 1
                
                pbar.update(1)
    
    # Convertir a numpy arrays para verificar la forma
    data = np.array(data)
    labels = np.array(labels)
    
    print("\nResumen del procesamiento:")
    print(f"Total de imágenes procesadas exitosamente: {processed_images}")
    print(f"Imágenes omitidas: {skipped_images}")
    print(f"Forma final de los datos: {data.shape}")
    print(f"Forma final de las etiquetas: {labels.shape}")
    
    # Verificar que todos los datos tienen la misma longitud
    if len(data) > 0:
        expected_length = len(data[0])
        if not all(len(x) == expected_length for x in data):
            raise ValueError("Error: Datos inconsistentes detectados")
    
    # Guardar los datos procesados
    data_path = Path(__file__).parent / 'unified_webcam_dataset_info.pickle'
    with open(data_path, 'wb') as f:
        pickle.dump({
            'data': data.tolist(),  # Convertir a lista para mejor compatibilidad
            'labels': labels.tolist()
        }, f)
    
    print(f"\nDataset creado exitosamente")
    print(f"Archivo guardado en: {data_path}")

if __name__ == "__main__":
    create_dataset()