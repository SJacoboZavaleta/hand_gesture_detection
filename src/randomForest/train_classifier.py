""""
Entrenamiento de clasificador de gestos de mano con Random Forest
Inputs:
    - Datos preparados previamente con create_dataset.py (unified_webcam_dataset_info.pickle)

Outputs:
    - Resultados de evaluación del clasificador en el conjunto de prueba (accuracy, precision, recall, f1, roc_auc, confusion matrix, classification report)
    - Gráficas de curvas de aprendizaje
    - Gráficas de importancia de características
    - Métricas en formato JSON
    - modelo entrenado (random_forest_model.pkl)
"""

import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)

def create_directories():
    """Crear estructura de directorios para resultados."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(__file__).parent.parent.parent / 'results' / 'randomforest_data1'
    eval_dir = base_dir / f'evaluation_{timestamp}'
    output_dir = eval_dir / 'evaluation'
    checkpoint_dir = eval_dir / 'checkpoints'
    
    for dir_ in [eval_dir, output_dir, checkpoint_dir]:
        dir_.mkdir(parents=True, exist_ok=True)
    
    return eval_dir, output_dir, checkpoint_dir

def plot_confusion_matrix(cm, classes, save_path):
    """Generar y guardar matriz de confusión."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_learning_curves(train_sizes, train_scores, val_scores, metric_name, save_path):
    """Generar y guardar gráficas de curvas de aprendizaje."""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, val_mean, label='Cross-validation', color='green', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='green')
    plt.xlabel('Training Samples')
    plt.ylabel(metric_name)
    plt.title(f'Learning Curves - {metric_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, save_path):
    """Generar y guardar gráfica de importancia de características."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics(metrics, save_path):
    """Guardar métricas en formato JSON."""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    # Crear directorios
    eval_dir, output_dir, checkpoint_dir = create_directories()
    
    try:
        # Cargar datos preprocesados
        print("Cargando datos...")
        with open(Path(__file__).parent / 'unified_webcam_dataset_info.pickle', 'rb') as f:
            data_dict = pickle.load(f)
        
        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])
        
        print(f"Forma de los datos: {data.shape}")
        print(f"Número de clases únicas: {len(np.unique(labels))}")
        
        # División de datos
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, shuffle=True,
            stratify=labels, random_state=42
        )
        
        # Configuración del modelo
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        print("Realizando validación cruzada...")
        cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
        cv_metrics = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std())
        }
        save_metrics(cv_metrics, output_dir / 'cross_validation_metrics.json')
        
        # Generar curvas de aprendizaje
        print("Generando curvas de aprendizaje...")
        train_sizes, train_scores, val_scores = learning_curve(
            model, x_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )
        
        # Graficar curvas de aprendizaje
        plot_learning_curves(
            train_sizes, train_scores, val_scores,
            'Accuracy',
            output_dir / 'learning_curves.png'
        )

        # Entrenamiento final
        print("Entrenando modelo final...")
        model.fit(x_train, y_train)
        
        # Graficar importancia de características
        plot_feature_importance(model, output_dir / 'feature_importance.png')

        # Evaluación
        print("Evaluando modelo...")
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)
        
        # Calcular métricas
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision_macro': float(precision_score(y_test, y_pred, average='macro')),
            'recall_macro': float(recall_score(y_test, y_pred, average='macro')),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
            'roc_auc_ovr': float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
        }
        
        # Guardar métricas detalladas
        classification_dict = classification_report(y_test, y_pred, output_dict=True)
        save_metrics(metrics, output_dir / 'metrics.json')
        save_metrics(classification_dict, output_dir / 'detailed_metrics.json')
        
        # Generar y guardar matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, np.unique(labels), output_dir / 'confusion_matrix.png')
        
        # Guardar modelo
        print("Guardando modelo...")
        model_path = checkpoint_dir / 'random_forest_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model}, f)
        
        print(f"\nEntrenamiento completado exitosamente!")
        print(f"Resultados guardados en: {output_dir}")
        print(f"Modelo guardado en: {model_path}")
        print(f"\nMétricas principales:")
        print(f"Cross-validation accuracy: {cv_metrics['cv_mean']:.4f} ± {cv_metrics['cv_std']:.4f}")
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
        print(f"ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
        
    except FileNotFoundError:
        print("Error: No se encontró el archivo de datos")
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        raise

if __name__ == "__main__":
    main()