# Crear estructura de carpetas y archivos .gitkeep
import os

# Lista de carpetas que necesitas mantener
carpetas = [
    'src/google_colab/data/doro',
    'src/google_colab/data/fergus',
    'src/google_colab/data/natalia',
    'src/google_colab/data/sergio'
]

# Crear carpetas y agregar .gitkeep
for carpeta in carpetas:
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, '.gitkeep'), 'w') as f:
        pass

print("Carpetas creadas con .gitkeep")

# Crear/actualizar .gitignore
gitignore_content = """
# Pero mantener las carpetas
!src/google_colab/data/doro/.gitkeep
!src/google_colab/data/fergus/.gitkeep
!src/google_colab/data/natalia/.gitkeep
!src/google_colab/data/sergio/.gitkeep
"""

with open('.gitignore', 'a') as f:
    f.write(gitignore_content)

print("\nArchivo .gitignore actualizado")