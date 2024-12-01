"""
Recolector de Imágenes para Dataset de Lenguaje de Señas
------------------------------------------------------
Este script captura imágenes desde la cámara web para crear un dataset de entrenamiento
para el clasificador de lenguaje de señas. Las imágenes se organizan en carpetas
numeradas (0, 1, 2) correspondientes a diferentes gestos.

Funcionamiento:
1. Crea directorios para cada clase si no existen
2. Para cada clase:
   - Espera a que el usuario presione 'q' para iniciar la captura
   - Captura 100 imágenes automáticamente
   - Guarda las imágenes en la carpeta correspondiente
"""

# === IMPORTACIÓN DE BIBLIOTECAS ===
import os
import cv2
import time
from pathlib import Path

# === CONFIGURACIÓN INICIAL ===
# Directorio para almacenar las imágenes
DATA_DIR = str(Path(__file__).parent.parent / 'data' / 'new_webcam_data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Parámetros de configuración
number_of_classes = 29        # Número de gestos diferentes a capturar
dataset_size = 100          # Cantidad de imágenes por gesto
capture_interval = 0.1      # Tiempo entre capturas (segundos)

# === INICIALIZACIÓN DE LA CÁMARA ===
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception('No se pudo abrir la cámara')

    # === CAPTURA DE IMÁGENES POR CLASE ===
    for j in range(number_of_classes):
        # Crear directorio para la clase actual
        class_dir = os.path.join(DATA_DIR, str(j))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print(f'\n=== Recolectando datos para el gesto {j} ===')
        print('Prepara tu mano y presiona "q" cuando estés listo')

        # Esperar a que el usuario esté listo
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Error: No se pudo leer el frame')
                continue
                
            # Mostrar instrucciones en pantalla
            cv2.putText(frame, 'Listo? Presiona "Q"!', (100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('Captura de Gestos', frame)
            
            # Verificar si el usuario presionó 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # === CAPTURA AUTOMÁTICA DE IMÁGENES ===
        print(f'\nIniciando captura automática de {dataset_size} imágenes...')
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print('Error: No se pudo leer el frame')
                continue
                
            # Mostrar vista previa
            cv2.imshow('Captura de Gestos', frame)
            cv2.waitKey(1)
            
            # Guardar imagen
            save_path = os.path.join(class_dir, f'{counter}.jpg')
            cv2.imwrite(save_path, frame)

            counter += 1
            print(f'Gesto {j}: Capturada imagen {counter} de {dataset_size}')

            # Pausa entre capturas
            time.sleep(capture_interval)

except Exception as e:
    print(f"Error durante la captura: {e}")

finally:
    # === LIMPIEZA Y CIERRE ===
    print("\nLiberando recursos...")
    cap.release()
    cv2.destroyAllWindows()
    print('¡Recolección de datos completada!')
    print(f'Las imágenes se guardaron en: {DATA_DIR}')