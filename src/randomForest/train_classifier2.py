"""
Clasificador de Gestos de Mano mediante Random Forest

Descripción General:
- Entrena un modelo de Random Forest para clasificación de gestos de mano
- Utiliza técnicas avanzadas de selección de características y optimización de hiperparámetros
- Genera múltiples visualizaciones y métricas de rendimiento

Características Principales:
- Búsqueda exhaustiva de hiperparámetros con GridSearchCV
- Selección automática de características más relevantes
- Validación cruzada para evaluar la generalización del modelo
- Generación de gráficas de:
  * Curvas de aprendizaje
  * Importancia de características
  * Matriz de confusión

Inputs:
    - Conjunto de datos de gestos de mano preprocesados (unified_webcam_dataset_info.pickle)

Outputs:
    - Métricas de evaluación detalladas (JSON)
    - Modelo entrenado (random_forest_model.pkl)
    - Visualizaciones de rendimiento

Dependencias Clave:
    - scikit-learn para modelado y evaluación
    - matplotlib y seaborn para visualizaciones
    - numpy para manipulación de datos
"""

# Imports
import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, LearningCurveDisplay, ShuffleSplit, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report) 
from sklearn.base import clone

def create_directories():
    """
    Gestiona la creación de una estructura de directorios organizada para resultados.
    
    Características:
    - Genera directorios con marca de tiempo para evitar sobreescrituras
    - Crea subdirectorios para:
      * Evaluación general
      * Checkpoints del modelo
    
    Returns:
        Rutas a los directorios creados para almacenar resultados y checkpoints
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(__file__).parent.parent.parent / 'results' / 'randomforest_data1'
    eval_dir = base_dir / f'evaluation_{timestamp}'
    output_dir = eval_dir / 'evaluation'
    checkpoint_dir = eval_dir / 'checkpoints'
    
    for dir_ in [eval_dir, output_dir, checkpoint_dir]:
        dir_.mkdir(parents=True, exist_ok=True)
    
    return eval_dir, output_dir, checkpoint_dir

def plot_confusion_matrix(cm, classes, save_path):
    """
    Genera curvas de aprendizaje para analizar el comportamiento del modelo.
    
    Características Clave:
    - Múltiples iteraciones de validación cruzada (cv=5)
    - Visualiza:
      * Rendimiento en entrenamiento
      * Rendimiento en validación
    - Intervalos de confianza mediante desviación estándar
    
    Insights:
    - Detecta overfitting o underfitting
    - Ayuda a determinar si se necesitan más datos
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_learning_curves(model, X, y, save_path):
    """
    Explora la contribución de características al modelo.
    
    Beneficios:
    - Ranking de características según su importancia
    - Permite interpretabilidad del modelo
    - Guía para selección y reducción de características
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # print("train_scores: ", train_scores)
    # print("val_scores: ", val_scores)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training scores', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, val_mean, label='Validation scores', color='orange', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='orange')
    plt.xlabel('Number of samples in the training set')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves for Random Forest (cv=5)')
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
    # Preparación del Entorno de Experimentación
    # - Crea estructura de directorios
    # - Configura semilla aleatoria para reproducibilidad
    eval_dir, output_dir, checkpoint_dir = create_directories()

    # Carga y Preparación de Datos
    # - Lectura de dataset preprocesado
    # - División estratificada train-test
    # - Balanceo de clases
    
    # Optimización de Hiperparámetros
    # - Búsqueda exhaustiva con GridSearchCV
    # - Métricas de evaluación balanceadas
    # - Optimización por F1-score
    
    # Selección de Características
    # - Análisis de importancia
    # - Reducción de dimensionalidad (umbral 85% varianza)
    
    # Entrenamiento y Evaluación
    # - Validación cruzada
    # - Generación de métricas detalladas
    # - Visualizaciones de rendimiento
        
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
        
        # Definir el espacio de búsqueda de hiperparámetros balanceado
        # param_grid = {
        #     'n_estimators': [250, 300],          # Número moderado de árboles
        #     'max_depth': [7,8,9],              # Profundidad intermedia
        #     'min_samples_split': [3, 8, 10],     # Valores intermedios
        #     'min_samples_leaf': [3, 6, 7],       # Valores intermedios
        #     'max_features': ['sqrt'],            # Mantener sqrt para mejor generalización
        #     'max_samples': [0.75, 0.8]           # Valores moderados para bootstrap
        # }
        
        # param_grid = {
        #     'n_estimators': [200, 300],
        #     'max_depth': [6, 8],
        #     'min_samples_split': [8, 10],
        #     'min_samples_leaf': [6, 8],
        #     'max_features': ['sqrt'],
        #     'max_samples': [0.7, 0.8]  # Bootstrap sample size (bagging)
        # }

        # Definir el espacio de búsqueda de hiperparámetros balanceado
        param_grid = {
            'n_estimators': [100, 250, 300],          # Número moderado de árboles
            'max_depth': [8, 10, 15],              # Profundidad intermedia
            'min_samples_split': [3, 8, 10],     # Valores intermedios
            'min_samples_leaf': [3, 6, 7],       # Valores intermedios
            'max_features': ['sqrt'],            # Mantener sqrt para mejor generalización
            'max_samples': [0.75, 0.8]           # Valores moderados para bootstrap
        }
        
        # Crear el modelo base con balance entre rendimiento y generalización
        base_model = RandomForestClassifier(
            bootstrap=True,
            class_weight='balanced',             # Volver a balanced simple
            random_state=42,
            n_jobs=-1,
            oob_score=True,
            criterion='gini'                     # Volver a gini que suele ser más estable
        )

        # Configurar GridSearchCV manteniendo métricas balanceadas
        scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'balanced_accuracy': 'balanced_accuracy'
        }

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring=scoring,
            refit='f1_weighted'  # Optimizar para F1-score en lugar de accuracy
        )

        print("Realizando búsqueda de hiperparámetros...")
        grid_search.fit(x_train, y_train)

        # Usar los mejores parámetros encontrados
        print("\nMejores parámetros encontrados:")
        print(grid_search.best_params_)
        
        # Usar el mejor modelo encontrado
        model = grid_search.best_estimator_

        # Analizar importancia de características
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Utilizar todas las características disponibles (42 por defecto)
        n_features_to_keep = 42
        
        print(f"\nUtilizando todas las {n_features_to_keep} características disponibles...")
        selected_features = indices[:n_features_to_keep]
        
        # Reentrenar el modelo con las características seleccionadas
        x_train_selected = x_train[:, selected_features]
        x_test_selected = x_test[:, selected_features]
        
        final_model = clone(model)
        final_model.fit(x_train_selected, y_train)
        model = final_model

        # Cross-validation con las características seleccionadas
        print("Realizando validación cruzada...")
        cv_scores = cross_val_score(model, x_train_selected, y_train, cv=5, scoring='accuracy')
        cv_metrics = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'n_selected_features': int(n_features_to_keep)
        }
        save_metrics(cv_metrics, output_dir / 'cross_validation_metrics.json')
        
        # Entrenamiento final
        print("Entrenando modelo final...")
        model.fit(x_train_selected, y_train)
        
        print("Generando curvas de aprendizaje...")
        plot_learning_curves(model, x_train_selected, y_train, output_dir / 'learning_curves.png')

        # Graficar importancia de características seleccionadas
        print("Generando características de importancia...")
        plot_feature_importance(model, output_dir / 'feature_importance.png')

        # Evaluación con características seleccionadas
        print("Evaluando modelo...")
        y_pred = model.predict(x_test_selected)
        y_pred_proba = model.predict_proba(x_test_selected)
        
        # Calcular métricas más detalladas
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision_macro': float(precision_score(y_test, y_pred, average='macro')),
            'precision_weighted': float(precision_score(y_test, y_pred, average='weighted')),
            'recall_macro': float(recall_score(y_test, y_pred, average='macro')),
            'recall_weighted': float(recall_score(y_test, y_pred, average='weighted')),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
            'roc_auc_ovr': float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr')),
            'n_selected_features': int(n_features_to_keep),
            'oob_score': float(model.oob_score_) if hasattr(model, 'oob_score_') else None
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
    # Ejecutar el entrenamiento
    main()