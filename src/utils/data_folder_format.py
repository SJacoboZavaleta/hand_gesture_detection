

import os
import json
import re

def load_class_lookup(file_path):
    """Load the class lookup JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def replace_number_with_string(folder_name, lookup_dict):
    """Replace numbers in the folder name with their corresponding strings."""
    # Sort numbers in descending order to replace multi-digit numbers first (e.g., "22" before "2")
    for number in sorted(lookup_dict.keys(), key=lambda x: len(x), reverse=True):
        # Use regular expression to replace whole numbers
        folder_name = re.sub(rf'\b{number}\b', lookup_dict[number], folder_name)
    return folder_name

def rename_folders_in_directory(root_dir, lookup_dict):
    """Rename all folders inside 'vision_data' to lowercase and replace numbers based on class_lookup.json."""
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        
        # Check if it is a directory
        if os.path.isdir(folder_path):
            # Convert folder name to lowercase
            new_folder_name = folder_name.lower()
            
            # Replace numbers with corresponding strings using lookup_dict
            new_folder_name = replace_number_with_string(new_folder_name, lookup_dict)
            
            # Check if the name changed
            if new_folder_name != folder_name:
                # Rename the folder
                new_folder_path = os.path.join(root_dir, new_folder_name)
                os.rename(folder_path, new_folder_path)
                print(f"Renamed: {folder_name} -> {new_folder_name}")
            else:
                print(f"No change: {folder_name}")

# Main function to run the script
def main():
    root_dir = '/Users/fergusproctor/dev/vision_data/c1' # Path to the main directory containing subfolders
    lookup_file = 'data/class_lookup.json'   # Path to the lookup file
    
    # Load the lookup dictionary from JSON
    lookup_dict = load_class_lookup(lookup_file)
    
    # Rename the folders
    rename_folders_in_directory(root_dir, lookup_dict)

if __name__ == "__main__":
    main()
