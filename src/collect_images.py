# Importación de bibliotecas necesarias
import os
import cv2
import time
#pip install opencv-python

# Configuración del directorio de datos
# Usando rutas relativas
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Parámetros de configuración
number_of_classes = 3
dataset_size = 100
capture_interval = 0.1  # Tiempo en segundos entre cada captura

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # Usar cámara predeterminada

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print('Error: No se pudo abrir la cámara')
    exit()

# Crear directorios para cada clase
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Recolectando datos para la clase {j}')

    # Esperar a que el usuario esté listo
    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error: No se pudo leer el frame')
            continue
            
        cv2.putText(frame, 'Listo? Presiona "Q"!', (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Capturar imágenes para la clase actual
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print('Error: No se pudo leer el frame')
            continue
            
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        
        # Guardar la imagen
        save_path = os.path.join(DATA_DIR, str(j), f'{counter}.jpg')
        cv2.imwrite(save_path, frame)

        counter += 1
        print(f'Clase {j}: Capturada imagen {counter} de {dataset_size}')

        # Esperar el intervalo definido antes de la siguiente captura
        time.sleep(capture_interval)
# Liberar recursos
cap.release()
cv2.destroyAllWindows()

print('Recolección de datos completada!')
