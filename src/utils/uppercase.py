import os

# Simple script to turn all folders in a directory to uppercase


folder_path = '/Users/fergusproctor/dev/unified_dataset'

for folder in os.listdir(folder_path):
    if os.path.isdir(os.path.join(folder_path, folder)):
        os.rename(os.path.join(folder_path, folder), os.path.join(folder_path, folder.upper()))
        print(f"Renamed {folder} to {folder.upper()}")

print("Done")


