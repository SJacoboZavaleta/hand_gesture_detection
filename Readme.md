# Real-Time Hand Gesture Recognition System

## Overview

A comprehensive computer vision system for real-time hand gesture recognition, developed as part of the Computer Vision course at Universidad Carlos III de Madrid. This project implements multiple machine learning approaches, including Random Forest for landmark-based classification and EfficientNet for deep learning-based image classification, to achieve robust hand gesture recognition through webcam input.

## Key Features

- **Multi-Model Architecture**: Implements both traditional ML (Random Forest) and deep learning (EfficientNet) approaches
- **Real-Time Processing**: Provides immediate gesture recognition through webcam feed
- **Extensible Framework**: Supports easy integration of new gestures and model architectures
- **Comprehensive Evaluation**: Includes detailed performance metrics and visualizations
- **Interactive Interface**: Real-time visualization of gesture recognition results

## Technical Architecture

The system comprises three main components:
1. **Data Processing Pipeline**: 
   - Hand landmark extraction using MediaPipe
   - Image preprocessing and augmentation
   - Dataset creation and management

2. **Model Implementation**:
   - Random Forest classifier using landmark features
   - EfficientNet with transfer learning and fine-tuning
   - Model evaluation and performance analysis

3. **Interactive Interface**:
   - Real-time webcam integration
   - Visual feedback system
   - Performance metrics display

## Installation

### Option 1: Using Python Virtual Environment (venv)

```bash
# Windows
python -m venv hand_gesture_env
.\hand_gesture_env\Scripts\activate

# Linux/macOS
python3 -m venv hand_gesture_env
source hand_gesture_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda Environment

```bash
# Create and configure environment
conda env create -f environment.yml -n hand_gesture_env

# Activate environment
conda activate hand_gesture_env
```

> **Note**: The environment name can be customized by modifying the `name` parameter in `environment.yml`.

## Usage

### Real-Time Classification

1. **Random Forest Model (Webcam-Collected Dataset)**
```bash
python src/interact_classifier.py --model_type ml --model_path results/randomforest_data1/evaluation_20241130_132749/checkpoints/random_forest_model.pkl
```

2. **EfficientNet Model (ASL Dataset)**
```bash
python src/interact_classifier.py --model_type efficientnet --model_path results/efficientnet_v2_s_data2/evaluation_20241129_222629/checkpoints/Fine_Tuning_best_model.pth
```

3. **EfficientNet Model (Unified Dataset)**
```bash
python src/interact_classifier.py --model_type efficientnet --model_path results/efficientnet_v2_s_data3/evaluation_20241129_153938/checkpoints/Fine_Tuning_best_model.pth
```

### Training Pipeline

#### Random Forest Model
1. **Dataset Creation**
```bash
python src/randomForest/create_dataset.py
```
This processes images from `data/webcam_data` and generates `unified_webcam_dataset.pickle`.

2. **Model Training**
```bash
python src/randomForest/train_classifier.py
```
Generates model artifacts and evaluation metrics in `results/randomforest_data1/evaluation_YYMMDD_HHMMSS/`.

#### EfficientNet Models

1. **Configuration**
   - Open `src/efficientNet/efficientnet_transferlearning_finetunning.py`
   - Set `DATA_TYPE = "data2"` or `"data3"` as needed

2. **Training**
```bash
python src/efficientNet/efficientnet_transferlearning_finetunning.py
```

> **Note**: An interactive Jupyter notebook version is available at `src/efficientNet/efficientnet_transferlearning_finetunning.ipynb`.

### Data Collection

To collect new training data:

1. Configure data directory in `src/collect_images.py`:
```python
DATA_DIR = str(Path(__file__).parent.parent / 'data' / 'new_webcam_data')
```

2. Run collection script:
```bash
python src/collect_images.py
```


## Project Structure

```bash
hand_gesture_detection/              # Hand Gesture Recognition Project Root
├── complementary/                   # Additional project resources and documentation
│   └── document/                   # Project documentation and research papers
│   └── clone_gc_repo.ipynb         # Notebook for cloning Google Colab repository
├── data/                           # Dataset storage and organization
│   ├── ASL/                       # American Sign Language dataset from Kaggle
│   │   ├───asl_alphabet_test      # Test set for ASL images
│   │   └───asl_alphabet_train     # Training set for ASL images
│   ├── unified_data/              # Combined and preprocessed dataset
│   │   └───unified_data           # Standardized image data for all gestures
│   ├── webcam_data/               # Custom dataset from webcam captures
│   │   ├───test                   # Test set from webcam data
│   │   └───unified_data           # Processed webcam images
│   └── class_lookup.json          # Mapping between gesture labels and numeric classes
├── results/                        # Training results and model evaluations
│   ├── efficientnet_v2_s_data2/   # EfficientNet results using ASL dataset
│   │   └── evaluation_20241129_222629/
│   │       ├── checkpoints/       # Model weight snapshots
│   │       │   ├── Fine_Tuning_best_model.pth      # Best model after fine-tuning
│   │       │   └── Transfer_Learning_best_model.pth # Best model after transfer learning
│   │       ├── evaluation/        # Performance evaluation results
│   │       │   ├── classification_metrics.csv       # Per-class performance metrics
│   │       │   ├── confusion_matrix.png            # Class prediction analysis
│   │       │   ├── training_history.json           # Training metrics over time
│   │       │   └── training_history.png            # Training progress visualization
│   │       └── model_architecture/                 # Network architecture details
│   │           ├── model_architecture_simplified.pdf  # PDF visualization
│   │           ├── model_architecture_simplified.png  # PNG visualization
│   │           └── model_summary_simplified.txt      # Layer-by-layer description
│   └── randomforest_data1/        # Random Forest model results
│       └── evaluation_20241130_132749/
│           ├── checkpoints/
│           │   └── random_forest_model.pkl         # Trained Random Forest model
│           └── evaluation/
│               ├── confusion_matrix.png            # Classification results matrix
│               ├── cross_validation_metrics.json   # K-fold validation results
│               ├── detailed_metrics.json           # Comprehensive metrics
│               ├── feature_importance.png          # Landmark importance analysis
│               ├── learning_curves.png            # Model learning progression
│               └── metrics.json                   # Summary performance metrics
├── src/                           # Source code for all implementations
│   ├── efficientNet/             # Deep learning implementation
│   │   ├── asl_dataset_info.csv     # ASL dataset metadata
│   │   ├── unified_data_dataset_info.csv      # Unified dataset metadata
│   │   ├── efficientnet_transferlearning_finetunning.ipynb  # Interactive training notebook
│   │   └── efficientnet_transferlearning_finetunning.py     # Training script
│   ├── randomForest/             # Traditional ML implementation
│   │   ├── unified_webcam_dataset_info.pickle  # Processed landmark features
│   │   ├── create_dataset.py     # Landmark extraction and preprocessing
│   │   └── train_classifier.py   # Random Forest training pipeline
│   ├── collect_images.py         # Webcam data collection utility
│   ├── hand_landmarker.task      # MediaPipe configuration for hand detection
│   └── interact_classifier.py     # Real-time gesture recognition interface
├── .gitignore                     # Version control exclusions
├── Readme.md                      # Project overview and setup guide
├── requirements.txt               # Pip dependencies
└── requirements.yml              # Conda environment specification
```

## Dependencies

- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Hand landmark detection
- **Scikit-learn**: Machine learning algorithms
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation

## Contributors

- [Doro](link)
- [Fergus](link)
- [Natalia](link)
- [Sergio](link)

## License

MIT License with Academic Citation Requirement.

Copyright (c) 2024 Fergus, Doro, Natalia and Sergio

## Acknowledgments

This project was developed as part of the Computer Vision course at Universidad Carlos III de Madrid under the supervision of [Professor Name].