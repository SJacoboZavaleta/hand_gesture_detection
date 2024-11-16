# Importación de bibliotecas necesarias
import pickle
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Cargar datos
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = torch.tensor(data_dict['data'], dtype=torch.float32)

# Convert string labels to numerical labels using LabelEncoder
label_encoder = LabelEncoder()
numerical_labels = label_encoder.fit_transform(data_dict['labels']) 
# fit_transform will convert string labels to numerical and remember the mapping

labels = torch.tensor(numerical_labels, dtype=torch.long) # Now this should work


# Convertir a tensores PyTorch
X = data.numpy()  # Convertimos a numpy porque RandomForest lo necesita
y = labels.numpy()

# Split datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# Crear y entrenar el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_predict = model.predict(X_test)
accuracy = (y_predict == y_test).mean()

print(f'{accuracy * 100}% de muestras fueron clasificadas correctamente!')

# Guardar el modelo
torch.save({
    'model': model,
    'accuracy': accuracy,
    'label_encoder': label_encoder # Save the encoder to use during inference
}, 'model.pth')

