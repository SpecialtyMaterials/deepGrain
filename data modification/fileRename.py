import os
import re

def rename_files(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Pattern to match 'augmented_edge_[x]' where [x] is a number
    pattern = re.compile(r'augmented_image_(\d+)')
    
    for file_name in files:
        match = pattern.match(file_name)
        if match:
            # Extract the number part
            number = match.group(1)
            # Construct the new file name
            new_file_name = f'augmented_image_{number}.png'
            # Construct full paths for the old and new file names
            old_file_path = os.path.join(directory, file_name)
            new_file_path = os.path.join(directory, new_file_name)
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {file_name} -> {new_file_name}')

# Example usage:
directory = "C:\\Users\\nickb\\Desktop\\projects\\cob detection\\DiffusionEdge-main\\Data\\207\\trainingRoot\\edge\\raw"  # Replace with your directory path
rename_files(directory)
