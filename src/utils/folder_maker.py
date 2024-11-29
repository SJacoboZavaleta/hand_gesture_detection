# quick script to format asi test data folder format into structure required by pytorch

import os
import shutil


test_data_path = "/Users/fergusproctor/Library/CloudStorage/OneDrive-Personal/Documents/Robotics Masters/Vision por Computadora/Proyecto Vision/Proyecto Vision-Fergus’s MacBook Air/asl_alphabet_test/asl_alphabet_test"

# Source directory containing your test images
source_dir = test_data_path
# Destination directory where folders will be created
dest_base_dir = test_data_path

# Create the base destination directory if it doesn't exist
os.makedirs(dest_base_dir, exist_ok=True)

# Loop through all files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.jpg'):
        # Get the class name (first letter before _test.jpg)
        class_name = filename.split('_')[0].upper()
        
        # Handle special case for 'nothing' class
        if filename.startswith('nothing'):
            class_name = 'nothing'
        elif filename.startswith('space'):
            class_name = 'space'
            
        # Create class directory if it doesn't exist
        class_dir = os.path.join(dest_base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy the file to its class directory
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(class_dir, filename)
        shutil.copy2(source_file, dest_file)

print("Organization complete!")