# Hand Gesture Recognition

## Project Description

This project is a hand gesture recognition system that uses a webcam to capture video input and a ML-based model to recognize the hand gestures. The system can be used for various applications such as sign language recognition, gesture-based control, and more. The project is built using Python and the libraries such as Pytorch, OpenCV, Scikit-learn and mediapipe. The project is part of the course *"Computer Vision"* at the Carlos III University of Madrid. 

## Project Structure

The project is structured as follows:

- `src/`: Source code for the project.
- `data/`: Data for the project.
- `Readme.md`: This readme file.

## Installation

1. Install the dependencies using the following command:

### Opción 1: Usando venv (Python Virtual Environment)

1. Crear y activar el entorno virtual:
```bash
# En Windows
python -m venv venv
.\venv\Scripts\activate

# En Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

### Opción 2: Usando Miniconda

1. Crear y activar el entorno conda:
```bash
# Crear el entorno
conda create -n gesture_env python=3.9

# Activar el entorno
conda activate gesture_env
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

2. Run the following command to train the model:

```bash
python src/train_classifier1.py
```

3. Run the following command to interact with the classifier:

```bash
python src/interact_classifier.py
```
4. Run the following command to collect new images and train the model:

```bash
python src/collect_images.py
```
## Project structure

```bash
proyecto/
├── data/
│   ├── 0/          # Imágenes del gesto A
│   ├── 1/          # Imágenes del gesto B
│   └── 2/          # Imágenes del gesto L
├── src/
│   ├── collect_images.py      # Script para capturar imágenes
│   ├── create_dataset.py      # Script para procesar imágenes y crear dataset
│   ├── interact_classifier.py  # Script para clasificación en tiempo real
│   ├── train_classifier.py    # Script para entrenar el modelo
│   └── data.pickle           # Dataset procesado
├── model.p                    # Modelo entrenado
├── .gitignore                # Configuración de Git
├── README.md                 # Documentación del proyecto
└── requirements.txt          # Dependencias del proyecto
```

## Team members

- [Doro]()
- [Fergus]()
- [Natalia]()
- [Sergio]()


