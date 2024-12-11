"""
EfficientNet Transfer Learning for Hand Gesture Recognition
=======================================================

This module implements a transfer learning approach using EfficientNet-B0 for hand gesture 
recognition. It fine-tunes a pre-trained EfficientNet model on ASL hand gesture data 
for improved classification performance.

Project Overview
--------------
Part of the Computer Vision Master's Project at UC3M (Universidad Carlos III de Madrid)
Date: 30/11/2024
Version: 1.0

Main Features
-----------
* Transfer learning implementation using EfficientNet-B0
* Custom dataset handling for hand gesture images
* Fine-tuning capabilities with configurable parameters
* Training progress monitoring and visualization
* Model performance evaluation
* Checkpoint saving and loading
* Learning rate scheduling

Technical Architecture
-------------------
1. Model Architecture:
   - Base: EfficientNet-B0 pre-trained on ImageNet
   - Modified classifier head for gesture classes
   - Frozen feature extraction layers
   - Fine-tuned top layers

2. Training Pipeline:
   - Custom dataset loading and preprocessing
   - Data augmentation
   - Transfer learning optimization
   - Learning rate scheduling
   - Model checkpointing

3. Evaluation Components:
   - Training/validation loss tracking
   - Accuracy metrics
   - Confusion matrix generation
   - Performance visualization

Dependencies
-----------
* PyTorch >= 1.9.0: Deep learning framework
* torchvision >= 0.10.0: Vision models and utilities
* EfficientNet-PyTorch: Pre-trained models
* NumPy >= 1.19.0: Numerical computations
* Matplotlib >= 3.3.0: Visualization
* PIL: Image processing
* tqdm: Progress tracking

Input Requirements
----------------
* Dataset Structure:
    - Root directory containing class subdirectories
    - Images organized by gesture classes
    - Supported formats: JPG, PNG  
    - Datasets:
        - Dataset 1: Custom dataset created from webcam data 
        - Dataset 2: [ASL Alphabet data](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data) from Kaggle
        - Dataset 3: Custom dataset based on the combination of datasets 1 and 2. 
* Training Configuration:
    - Batch size
    - Learning rate
    - Number of epochs
    - Device selection (CPU/GPU)

Output
------
* Trained Model:
    - Saved model checkpoints
    - Best model weights
    - Training state
* Performance Metrics:
    - Training/validation loss curves
    - Accuracy plots
    - Confusion matrix
    - Per-class performance metrics

Training Parameters
-----------------
* BATCH_SIZE: Mini-batch size for training
* LEARNING_RATE: Initial learning rate
* NUM_EPOCHS: Total training epochs
* WEIGHT_DECAY: L2 regularization factor
* NUM_CLASSES: Number of gesture classes
* CHECKPOINT_DIR: Directory for saving models

Model Architecture Details
------------------------
* Base Model: EfficientNet-B0
* Input Size: 224x224x3
* Feature Extraction: Pre-trained weights
* Classifier Head: Custom fully connected layers
* Output: Softmax probabilities for gestures

Training Process
--------------
1. Data Preparation:
   - Image resizing and normalization
   - Data augmentation (random transforms)
   - Batch creation

2. Training Loop:
   - Forward pass
   - Loss computation
   - Backpropagation
   - Optimizer step
   - Learning rate adjustment

3. Validation:
   - Model evaluation
   - Metric computation
   - Best model saving

Performance Considerations
------------------------
* GPU Requirements:
    - Recommended: NVIDIA GPU with 6GB+ VRAM
    - CUDA support required for GPU training
* Training Time:
    - Varies with dataset size and epochs
    - GPU training significantly faster
* Memory Usage:
    - Depends on batch size
    - Typical range: 4-8GB RAM

Notes
-----
* Pre-trained weights significantly reduce training time
* Data augmentation crucial for generalization
* Regular checkpointing prevents training loss
* Monitor validation metrics for overfitting

References
---------
1. EfficientNet Paper: https://arxiv.org/abs/1905.11946
2. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
3. Transfer Learning Guide: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
4. [Add relevant papers or resources]

"""

import thop
import torch
import torch.nn.functional as F
import torchmetrics #conda install -c conda-forge torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import time
import json
import torchviz
import graphviz
from torchsummary import summary

DATA_TYPE = 'data2'

if DATA_TYPE == 'data2':
    DATA_PATH = os.path.join('..', '..', 'data', 'ASL','asl_alphabet_train')
elif DATA_TYPE == 'data3':
    DATA_PATH = os.path.join('..', '..', 'data', 'unified_data','unified_data') 
else:
    raise ValueError(f"Data {DATA_TYPE} not found.")

# elif DATA_TYPE == 'data1': # No considerado
#     DATA_PATH = os.path.join('..', '..', 'data', 'webcam_data','unified_data')

# Function to create the DataFrame from the dataset
# Uncomment to use. The output is a dataframe stored in asl_dataset_info.csv

# def create_dataframe(data_path):
#     """
#     Crea un DataFrame con las rutas de las imágenes y sus etiquetas.

#     Args:
#         data_path (str): Ruta al directorio que contiene las carpetas de clases (0-29)

#     Returns:
#         pd.DataFrame: DataFrame con columnas ['Filepaths', 'Labels', 'Label_idx']
#     """
#     # Convertir a Path object para mejor manejo de rutas
#     data_path = Path(data_path)

#     if not data_path.exists():
#         raise ValueError(f"El directorio {data_path} no existe")

#     # Listas para almacenar datos
#     filepaths = []
#     labels = []
#     label_indices = []
#     img_sizes = []

#     # Obtener todas las carpetas y ordenarlas numéricamente
#     folders = sorted([f for f in data_path.iterdir() if f.is_dir()],
#                     key=lambda x: int(x.name))

#     print("Creando DataFrame...")
#     # Usar tqdm para mostrar progreso
#     for folder in tqdm(folders, desc="Procesando carpetas"):
#         label_idx = int(folder.name)

#         # Obtener todas las imágenes en la carpeta
#         valid_extensions = {'.jpg', '.jpeg', '.png'}
#         images = [f for f in folder.iterdir()
#                  if f.suffix.lower() in valid_extensions]

#         for img_path in images:
#             # Verificar que la imagen se puede leer
#             try:
#                 img = cv2.imread(str(img_path))
#                 if img is None:
#                     print(f"Advertencia: No se pudo leer {img_path}")
#                     continue

#                 height, width = img.shape[:2]

#                 filepaths.append(str(img_path))
#                 labels.append(folder.name)
#                 label_indices.append(label_idx)
#                 img_sizes.append((width, height))

#             except Exception as e:
#                 print(f"Error procesando {img_path}: {str(e)}")

#     # Crear DataFrame
#     df = pd.DataFrame({
#         'Filepaths': filepaths,
#         'Labels': labels,
#         'Label_idx': label_indices,
#         'Image_size': img_sizes
#     })

#     # Mostrar información del dataset
#     print("\nResumen del Dataset:")
#     print(f"Total de imágenes: {len(df)}")
#     print(f"Número de clases: {len(df['Labels'].unique())}")
#     print("\nDistribución de clases:")
#     print(df['Labels'].value_counts().sort_index())

#     # Verificar balance de clases
#     min_samples = df['Labels'].value_counts().min()
#     max_samples = df['Labels'].value_counts().max()
#     print(f"\nMínimo de muestras por clase: {min_samples}")
#     print(f"Máximo de muestras por clase: {max_samples}")

#     # Verificar tamaños de imagen
#     sizes = pd.DataFrame(df['Image_size'].tolist(), columns=['width', 'height'])
#     print("\nTamaños de imagen:")
#     print(f"Mínimo: {sizes.min().values}")
#     print(f"Máximo: {sizes.max().values}")
#     print(f"Moda: {sizes.mode().iloc[0].values}")

#     return df

# try:
#     # Images in 'data/...'  
#     df = create_dataframe(DATA_PATH)
    
#     # Save dataframe of images paths and labels
#     if DATA_TYPE == 'data2':
#         df.to_csv('asl_dataset_info.csv', index=False)
#     elif DATA_TYPE == 'data3':
#         df.to_csv('unified_data_dataset_info.csv', index=False)

#     print("\nPrimeras filas del DataFrame:")
#     print(df.head())
    
# except Exception as e:
#     print(f"Error: {str(e)}")

# Load DataFrame of images previously created
# Load DataFrame of images previously created
if DATA_TYPE == 'data2':
    df = pd.read_csv('src/efficientNet/asl_dataset_info.csv')
elif DATA_TYPE == 'data3':
    df = pd.read_csv('src/efficientNet/unified_data_dataset_info.csv')

print(df.head())

# Configure the device for training
def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True  # Optimiza el rendimiento
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    return device

DEVICE = setup_device()

class ASLDataset(Dataset):
    """
    Custom Dataset for loading ASL (American Sign Language) images.

    This dataset class handles loading and preprocessing of ASL hand gesture images.
    It supports on-the-fly data augmentation and preprocessing for model training.

    Attributes:
        df (pd.DataFrame): DataFrame containing image paths and labels
        transform (callable): Torchvision transforms for image preprocessing
        is_training (bool): Flag to enable/disable data augmentation
    """

    def __init__(self, dataframe, transform=None):
        """
        Initialize the ASL Dataset.

        Args:
            df (pd.DataFrame): DataFrame with columns ['Filepaths', 'Labels']
            transform (callable, optional): Transform to be applied to images
            is_training (bool): If True, enables data augmentation
        """
        self.dataframe = dataframe
        self.transform = transform
        self.labels = pd.Categorical(dataframe['Labels']).codes

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Fetch and preprocess a single image item from the dataset.

        Args:
            idx (int): Index of the image to fetch

        Returns:
            tuple: (image, label) where image is the preprocessed tensor
                  and label is the corresponding class index
        """
        img_path = self.dataframe.iloc[idx]['Filepaths']
        label = self.dataframe.iloc[idx]['Label_idx']  # Asegúrate de que esto sea un número entero

        try:
            # Read and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                image = self.transform(image)

            # Defining data type for labels
            label = torch.tensor(int(label), dtype=torch.long)

            return image, label

        except Exception as e:
            print(f"Error loading imagen {img_path}: {str(e)}")
            # Retorning a black image
            if self.transform:
                dummy_image = torch.zeros((3, 384, 384))
            else:
                dummy_image = np.zeros((384, 384, 3))
            return dummy_image, label

class ASLModel(nn.Module):
    """
    Custom Neural Network model for ASL gesture recognition using transfer learning.

    This model uses EfficientNetV2-S as the backbone with custom classification layers.
    The architecture is designed to balance accuracy and computational efficiency.

    Architecture Overview:
    ---------------------
    1. EfficientNetV2-S backbone (pretrained on ImageNet)
    2. Custom dense layers with batch normalization
    3. Dropout for regularization
    4. Softmax output layer

    Attributes:
        base_model (nn.Module): Pretrained EfficientNetV2-S model
        classifier (nn.Sequential): Custom classification layers
        num_classes (int): Number of output classes
    """

    def __init__(self, num_classes=29, base_model_name='efficientnet_v2_s', dense_units=256, dropout_rate=0.5):
        """
        Initialize the ASL Model.

        Args:
            num_classes (int): Number of output classes (ASL gestures)
            base_model_name (str): Name of the pretrained model to use
            dense_units (int):
            droput_rate (int):
        """
        super().__init__()

        # Load pretrained model
        if base_model_name.lower() == 'efficientnet_v2_s':
            self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
            num_features = 1280
        else:
            raise ValueError(f"Model {base_model_name} not supported")

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Own classifier
        self.dense_block1 = nn.Sequential(
            nn.Linear(num_features, dense_units*2, bias=False),
            nn.BatchNorm1d(dense_units*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.dense_block2 = nn.Sequential(
            nn.Linear(dense_units*2, dense_units, bias=False),
            nn.BatchNorm1d(dense_units),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Modify classifier
        self.classifier = nn.Linear(dense_units, num_classes)

        # Initialize weights
        self._initialize_weights()

        # Freeze pretrained newtork
        self.freeze_base_model()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_base_model(self):
        # Freeze early layers
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, num_layers=30):
        trainable_layers = list(self.base_model.parameters())[-num_layers:]
        for param in trainable_layers:
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, num_classes)
        """

        # Base model features
        x = self.base_model(x)

        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Dense blocks
        x = self.dense_block1(x)
        x = self.dense_block2(x)

        # Output with softmax
        x = self.classifier(x)
        # We don't apply a classification here, we do it later
        #out = F.softmax(x, dim=1)

        return x

def create_data_loaders(df, transform, batch_size=32, train_split=0.8, val_split=0.1):
    """
    Create train, validation, and test data loaders.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels
        transform (callable): Torchvision transforms for image preprocessing
        batch_size (int): Batch size for data loaders
        train_split (float): Proportion of data used for training (default: 0.8)
        val_split (float): Proportion of data used for validation (default: 0.1)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    dataset = ASLDataset(df, transform=transform)

    # Calculate sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # Create splits
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Adjust workers according to your CPU cores (generally num_cores - 1)
    # num_workers = min(4, os.cpu_count() - 1) if os.cpu_count() > 1 else 0

    # Configure a common DataLoader for training, validation, and testing dataloaders
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 0,#num_workers
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': False#if num_workers > 0 else False
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

    return train_loader, val_loader, test_loader

def generate_evaluation_metrics(model, test_loader, history_phase1, history_phase2, evaluation_path):
    """
    Generate and save comprehensive evaluation metrics for the model.

    This function creates various visualizations and metrics including:
    - Training/validation loss curves
    - Accuracy plots
    - Confusion matrix
    - Classification report
    - Per-class performance metrics

    Evaluation Components:
    ---------------------
    1. Model Performance Metrics:
        - Test Loss (Cross-Entropy)
        - Test Accuracy
        - Per-class Precision, Recall, and F1-score

    2. Visualizations:
        - Confusion Matrix: Shows prediction patterns across all classes
        - Training History Plots:
            * Loss curves (training and validation)
            * Accuracy curves (training and validation)

    3. Saved Outputs:
        - classification_metrics.csv: Detailed per-class metrics
        - training_history.json: Complete training history
        - confusion_matrix.png: Visual representation of model predictions
        - training_curves.png: Learning curves from both training phases

    Args:
        model (nn.Module): Trained model to evaluate
        test_loader (DataLoader): DataLoader for test data
        history_phase1 (dict): Training history from phase 1
        history_phase2 (dict): Training history from phase 2
        evaluation_path (str): Directory to save evaluation results

    Returns:
        dict: A dictionary containing all evaluation metrics and history:
            {
                'training_history': {
                    'phase1': {train_losses, train_accuracies, val_losses, val_accuracies},
                    'phase2': {train_losses, train_accuracies, val_losses, val_accuracies}
                },
                'final_metrics': {
                    'test_loss': float,
                    'test_accuracy': float,
                    'classification_report': dict
                }
            }
    """

    # Evaluate on test set
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluando'):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE, dtype=torch.long)

            outputs = model(inputs)
            outputs = outputs.float()

            # Loss function using cross entropy
            loss = F.cross_entropy(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate final metrics
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    # Load a label mapping
    label_mapping = load_label_mapping('../data/class_lookup_v2.json')

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    classification_rep = classification_report(all_labels, all_preds,
                                            target_names=list(label_mapping.values()),
                                            output_dict=True)

    # Save visualizations
    _save_confusion_matrix(cm, label_mapping, evaluation_path)
    _save_training_history(history_phase1, history_phase2, evaluation_path)

    # Save metrics in csv
    metrics_df = pd.DataFrame(classification_rep).transpose()
    metrics_df.to_csv(os.path.join(evaluation_path, 'classification_metrics.csv'))

    # Prepare complete history
    history_data = {
        'training_history': {
            'phase1': {
                'train_losses': [float(x) for x in history_phase1['train_losses']],
                'train_accuracies': [float(x) for x in history_phase1['train_accuracies']],
                'val_losses': [float(x) for x in history_phase1['val_losses']],
                'val_accuracies': [float(x) for x in history_phase1['val_accuracies']]
            },
            'phase2': {
                'train_losses': [float(x) for x in history_phase2['train_losses']],
                'train_accuracies': [float(x) for x in history_phase2['train_accuracies']],
                'val_losses': [float(x) for x in history_phase2['val_losses']],
                'val_accuracies': [float(x) for x in history_phase2['val_accuracies']]
            }
        },
        'final_metrics': {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'classification_report': classification_rep
        }
    }

    # Save history in a JSON
    with open(os.path.join(evaluation_path, 'training_history.json'), 'w') as f:
        json.dump(history_data, f, indent=4)

    # Show summary
    print("\nResumen Final del Entrenamiento:")
    print(f"Precisión en test: {test_accuracy:.2f}%")
    print(f"Pérdida en test: {test_loss:.4f}")
    print("\nMétricas por clase:")
    print(metrics_df)

    return history_data

def _save_confusion_matrix(cm, label_mapping, evaluation_path):
    """Save confusion matrix of model evaluation"""

    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(label_mapping.values()),
                yticklabels=list(label_mapping.values()))
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_path, 'confusion_matrix.png'))
    plt.close()

def _save_training_history(history_phase1, history_phase2, evaluation_path):
    """Save plots of training history"""
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history_phase1['train_losses'], label='Phase 1 Train')
    plt.plot(history_phase1['val_losses'], label='Phase 1 Val')
    plt.plot([len(history_phase1['train_losses']) + i for i in range(len(history_phase2['train_losses']))],
             history_phase2['train_losses'], label='Phase 2 Train')
    plt.plot([len(history_phase1['val_losses']) + i for i in range(len(history_phase2['val_losses']))],
             history_phase2['val_losses'], label='Phase 2 Val')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history_phase1['train_accuracies'], label='Phase 1 Train')
    plt.plot(history_phase1['val_accuracies'], label='Phase 1 Val')
    plt.plot([len(history_phase1['train_accuracies']) + i for i in range(len(history_phase2['train_accuracies']))],
             history_phase2['train_accuracies'], label='Phase 2 Train')
    plt.plot([len(history_phase1['val_accuracies']) + i for i in range(len(history_phase2['val_accuracies']))],
             history_phase2['val_accuracies'], label='Phase 2 Val')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_path, 'training_history.png'))
    plt.close()

def train_one_phase(model, train_loader, val_loader, optimizer, num_epochs, phase_name, early_stopping_patience, output_dir):
    """
    Train the model for a single phase.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        num_epochs (int): Number of training epochs.
        phase_name (str): Name of the training phase.
        early_stopping_patience (int): Number of epochs to wait before early stopping.
        output_dir (str): Directory to save model checkpoints and logs.

    Returns:
        dict: A dictionary containing training history.
    """

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Create output directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    scaler = torch.amp.GradScaler('cuda') # For precised training

    for epoch in range(num_epochs):

        # Training mode
        model.train()
        total_train_loss = 0
        train_steps = 0
        train_correct = 0
        train_total = 0

        # Progress bar during training
        train_pbar = tqdm(train_loader, desc=f'{phase_name} Epoch {epoch+1}/{num_epochs} [Train]')

        for inputs, labels in train_pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE,dtype=torch.long)

            optimizer.zero_grad()

            # Training with a mixed precision
            with torch.cuda.amp.autocast(): #torch.amp.autocast('cuda')
                outputs = model(inputs)
                outputs = outputs.float()
                loss = F.cross_entropy(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Calcute loss
            total_train_loss += loss.item()
            train_steps += 1

             # Calcute current accuracy
            current_train_acc = 100 * train_correct / train_total

            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_train_acc:.2f}%'
            })

        # Calcute training metrics
        avg_train_loss = total_train_loss / train_steps
        train_accuracy = 100 * train_correct / train_total

        # Save metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluation mode
        model.eval()
        total_val_loss = 0
        val_steps = 0
        correct = 0
        total = 0

        # Progress bar during validation
        val_pbar = tqdm(val_loader, desc=f'{phase_name} Epoch {epoch+1}/{num_epochs} [Val]')

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE,dtype=torch.long)

                outputs = model(inputs)
                outputs = outputs.float()
                loss = F.cross_entropy(outputs, labels)

                total_val_loss += loss.item()
                val_steps += 1

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = total_val_loss / val_steps
        val_accuracy = 100 * correct / total

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        }

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save best model
            best_model_path  = os.path.join(checkpoint_dir, f'{phase_name}_best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"\nGuardando el mejor modelo en {best_model_path}")
        else:
            patience_counter += 1

        # Show metrics
        print(f'\n{phase_name} Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        if patience_counter >= early_stopping_patience:
            print(f'\nEarly stopping triggered after {patience_counter} epochs without improvement')
            break

    # Load the best model before returing
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

def train_model_complete(df, transform, num_classes=29, batch_size=32, num_epochs1=5, num_epochs2=5,
                        learning_rate1=1e-3, learning_rate2=1e-5, early_stopping_patience=3, output_dir='results/'):
    """
    Train the model in two phases: Transfer Learning and Fine-tuning.

    Args:
        df (pd.DataFrame): DataFrame containing the training data.
        transform (torchvision.transforms.Compose): Transformations to apply to the input data.
        num_classes (int): Number of classes in the dataset.
        batch_size (int): Batch size for training.
        num_epochs1 (int): Number of epochs for the first phase of training.
        num_epochs2 (int): Number of epochs for the second phase of training.
        learning_rate1 (float): Learning rate for the first phase of training.
        learning_rate2 (float): Learning rate for the second phase of training.
        early_stopping_patience (int): Number of epochs to wait before early stopping.
        output_dir (str): Directory to save model checkpoints and logs.
    Returns:
        tuple: (trained_model, training_history)
    """

    print("Starting the complete training...")

    # Crear data loaders
    print("Creating  data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        df=df,
        transform=transform,
        batch_size=batch_size
    )

    # Create the model and move it to device
    print("Initializing the model...")
    model = ASLModel(num_classes=num_classes)
    model = model.to(DEVICE)

    # Fase 1: Transfer Learning
    print("\n Phase 1:  Transfer Learning - Only new layers")
    optimizer_phase1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate1,
        weight_decay=1e-4
    )

    history_phase1 = train_one_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_phase1,
        num_epochs=num_epochs1,
        phase_name='Transfer_Learning',
        early_stopping_patience=early_stopping_patience,
        output_dir=output_dir
    )

    # Fase 2: Fine-tuning
    print("\Phase 2: Fine-tuning - Complete model")
    model.unfreeze_layers(num_layers=30)

    optimizer_phase2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate2,
        weight_decay=1e-4
    )

    history_phase2 = train_one_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_phase2,
        num_epochs=num_epochs2,
        phase_name='Fine_Tuning',
        early_stopping_patience=early_stopping_patience,
        output_dir=output_dir
    )

    # Create a new directory to save evaluation
    evaluation_path = os.path.join(output_dir, 'evaluation')
    os.makedirs(evaluation_path, exist_ok=True)

    # Evaluate model
    print("\nRealizando evaluación final...")
    history_data = generate_evaluation_metrics(
        model=model,
        test_loader=test_loader,
        history_phase1=history_phase1,
        history_phase2=history_phase2,
        evaluation_path=evaluation_path
    )

    print(f"\nResultados guardados en: {evaluation_path}")

    return model, history_data

def load_label_mapping(json_path):
    """
    Load label mapping from a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary mapping class indices to class labels.
    """
    try:
        with open(json_path, 'r') as f:
            class_mapping = json.load(f)

        # Convertir las claves de string a int y los valores a mayúsculas
        label_mapping = {int(k): v.upper() for k, v in class_mapping.items()}
        return label_mapping

    except Exception as e:
        print(f"Error by loading label mapping: {e}")
        print("Using default label mapping...")

        # Mapeo por defecto
        default_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                         'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
        label_mapping = {idx: label for idx, label in enumerate(default_labels)}
        return label_mapping

def save_model_architecture(model, output_dir, input_size=(3, 384, 384)):
    """
    Save a visual representation of the model architecture.

    Creates a simplified visualization of the model's architecture using graphviz,
    showing the main components and their connections.

    Args:
        model (nn.Module): Model to visualize
        output_dir (str): Directory to save the visualization
        input_size (tuple): Input tensor dimensions (channels, height, width)

    Returns:
        str: Path to the saved visualization and summary files
    """
    import io
    from contextlib import redirect_stdout
    from torchinfo import summary
    from graphviz import Digraph

    # Create output directory
    architecture_dir = os.path.join(output_dir, 'model_architecture')
    os.makedirs(architecture_dir, exist_ok=True)

    try:
        # Create graph
        dot = Digraph(comment='Model Architecture')
        dot.attr(rankdir='TB')
        dot.attr('node', shape='box', style='rounded')

        # Define style to nodes
        dot.attr('node', fontname='Arial')

        # Input
        channels, height, width = input_size
        input_label = f'Input\n({height}×{width}×{channels})'
        dot.node('input', input_label, shape='oval')

        # EfficientNetV2-S (Pre-trained)
        dot.node('backbone', 'EfficientNetV2-S\n(Pre-trained)', style='filled', fillcolor='lightgray')

        # Global Pooling
        dot.node('pool', 'Global Average Pooling')

        # Dense blocks
        dot.node('dense1', 'Dense Block 1\n512 units\nBatchNorm + ReLU\nDropout (0.5)')
        dot.node('dense2', 'Dense Block 2\n256 units\nBatchNorm + ReLU\nDropout (0.3)')

        # Output
        dot.node('output', 'Output Layer\n29 classes\nSoftmax', shape='oval')

        # Add conexions
        dot.edge('input', 'backbone')
        dot.edge('backbone', 'pool')
        dot.edge('pool', 'dense1')
        dot.edge('dense1', 'dense2')
        dot.edge('dense2', 'output')

        # Save the graph
        dot.render(os.path.join(architecture_dir, 'model_architecture_simplified'),
                  format='png', cleanup=True)
        dot.render(os.path.join(architecture_dir, 'model_architecture_simplified'),
                  format='pdf', cleanup=True)

        # Save basic summary
        summary_file = os.path.join(architecture_dir, 'model_summary_simplified.txt')
        with open(summary_file, 'w') as f:
            f.write("ASL Hand Gesture Classification Model\n")
            f.write("====================================\n\n")
            f.write("Architecture Overview:\n")
            f.write("1. Input Layer: 384×384×3\n")
            f.write("2. Backbone: EfficientNetV2-S (pre-trained)\n")
            f.write("3. Global Average Pooling\n")
            f.write("4. Dense Block 1:\n")
            f.write("   - 512 units\n")
            f.write("   - Batch Normalization\n")
            f.write("   - ReLU Activation\n")
            f.write("   - Dropout (0.5)\n")
            f.write("5. Dense Block 2:\n")
            f.write("   - 256 units\n")
            f.write("   - Batch Normalization\n")
            f.write("   - ReLU Activation\n")
            f.write("   - Dropout (0.3)\n")
            f.write("6. Output Layer:\n")
            f.write("   - 29 units (classes)\n")
            f.write("   - Softmax activation\n\n")
            f.write("Training Strategy:\n")
            f.write("- Phase 1: Transfer Learning (frozen backbone)\n")
            f.write("- Phase 2: Fine-tuning (last 30 layers unfrozen)\n")

        print(f"Visualization saved in  {architecture_dir}")
        print(f"Summary saved in {summary_file}")

    except Exception as e:
        print(f"Error creating the visualization: {str(e)}")
        print("Make sure to have graphviz installed:")
        print("1. Install graphviz on your system:")
        print("   - Windows: https://graphviz.org/download/")
        print("   - Linux: sudo apt-get install graphviz")
        print("   - macOS: brew install graphviz")
        print("2. Install the Python package: pip install graphviz")

if __name__ == '__main__':
    # Needed for multiprocessing on Windows
    from multiprocessing import freeze_support
    freeze_support()
    
    TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    # i.e. 'results/efficientnet_v2_s_data2/evaluation_20241129_114615'
    if DATA_TYPE == 'data2':
        OUTPUT_PATH =  os.path.join('..', '..', 'results', 'efficientnet_v2_s_data2', f'evaluation_{TIME_STAMP}')
    elif DATA_TYPE == 'data3':
        OUTPUT_PATH =  os.path.join('..', '..', 'results', 'efficientnet_v2_s_data3', f'evaluation_{TIME_STAMP}')
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Define hyperparameters
    PARAMS = {
            'input_size': (384, 384, 3),
            'num_classes': 29,
            'batch_size': 32,
            'dense_units': 256,
            'dropout_rate': 0.5,
            'learning_rate1': 1e-3,
            'learning_rate2': 1e-5,
            'epochs_phase1': 1,
            'epochs_phase2': 1,
            'early_stopping_patience': 3
        }

    # Save simplified model architecture    
    # Model architecture
    model = ASLModel(num_classes=29, base_model_name='efficientnet_v2_s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, input_size=(PARAMS['input_size'][2], PARAMS['input_size'][0], PARAMS['input_size'][1]), device='cuda' if torch.cuda.is_available() else 'cpu')

    save_model_architecture(
        model=model,
        output_dir=OUTPUT_PATH,
        input_size=PARAMS['input_size']
    )

    # Image preprocessing
    # Using Imagenet mean and std
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((PARAMS['input_size'][0], PARAMS['input_size'][1])),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Create and train model
    model, history_data = train_model_complete(
        df=df,
        transform=transform,
        num_classes=PARAMS['num_classes'],
        batch_size=PARAMS['batch_size'],
        num_epochs1=PARAMS['epochs_phase1'],
        num_epochs2=PARAMS['epochs_phase2'],
        learning_rate1=PARAMS['learning_rate1'],
        learning_rate2=PARAMS['learning_rate2'],
        early_stopping_patience=PARAMS['early_stopping_patience'],
        output_dir=OUTPUT_PATH
    )
