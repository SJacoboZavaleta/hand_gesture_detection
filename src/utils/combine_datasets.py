import os
import shutil
import json

# Paths to your 5 original folders
folder_paths = ['/Users/fergusproctor/dev/vision_data/c1', '/Users/fergusproctor/dev/vision_data/asl_alphabet_train'
                , '/Users/fergusproctor/dev/vision_data/doro_sign_language', '/Users/fergusproctor/dev/vision_data/fergus'
                , '/Users/fergusproctor/dev/vision_data/natalia']


# Function to get the parent directory name
def get_parent_dir_name(path):
    # Split the path and get the last directory name
    return os.path.basename(path)

# Rename files in all folders
#for folder_path in folder_paths:
    parent_name = get_parent_dir_name(folder_path)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Get full file path
            file_path = os.path.join(root, filename)
            
            # Create new filename with parent directory
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{parent_name}{ext}"
            
            # Create new file path
            new_file_path = os.path.join(root, new_filename)
            
            # Rename the file
            try:
                os.rename(file_path, new_file_path)
            except OSError as e:
                print(f"Error renaming {file_path}: {e}")



# Path to the new unified dataset
unified_dataset_path = 'unified_dataset'

# Create the unified dataset folder
os.makedirs(unified_dataset_path, exist_ok=True)
# Load the class lookup dictionary
with open('data/class_lookup.json', 'r') as f:
    class_lookup = json.load(f)

print(class_lookup)


i = 0
for label, class_num in class_lookup.items():
    

 # Assuming 29 classes (class_1 to class_29)
    # Create a new folder for each class in the unified dataset
    class_folder = os.path.join(unified_dataset_path, f'{class_num}')
    os.makedirs(class_folder, exist_ok=True)
    
    # Loop over the 5 original folders
    for folder in folder_paths:
        print(f'unifying {folder}')
        # Path to the class folder in the current folder
        class_folder_in_original = os.path.join(folder, f'{class_num}')
        
        # Check if the class folder exists in the original folder
        if os.path.exists(class_folder_in_original):
            # Move or copy all images from the current class folder into the new unified class folder
            for file_name in os.listdir(class_folder_in_original):
                file_path = os.path.join(class_folder_in_original, file_name)
                if os.path.isfile(file_path):
                    # Get the file extension from the original file
                    _, ext = os.path.splitext(file_name)
                    # Create new filename using counter i
                    new_filename = f"{i}{ext}"
                    # Update the destination path with the new filename
                    dest_path = os.path.join(class_folder, new_filename)
                    shutil.copy(file_path, class_folder)
                    i += 1

print("Data combined into the unified dataset.")
