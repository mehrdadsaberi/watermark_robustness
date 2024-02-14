import os
# import random
import numpy as np
import shutil
import argparse

def split_data_folderA_folderC(source_folder, test_partition_ids, dest_folder_test, dest_folder_train):
    # Create destination folders if they don't exist
    os.makedirs(dest_folder_test, exist_ok=True)
    os.makedirs(dest_folder_train, exist_ok=True)
    
    # Get a list of files in the source folder
    files = os.listdir(source_folder)
    
    # Iterate over each file in the source folder
    for file in files:
        filename = os.path.splitext(file)[0]  # Remove file extension
        
        # Extract x and y from the filename
        x, y, _, _ = filename.split('_')
        
        # Check if x or y is in the test partition ids
        if x in test_partition_ids or y in test_partition_ids:
            # Copy the file to the test folder
            shutil.copy2(os.path.join(source_folder, file), dest_folder_test)
        else:
            # Copy the file to the train folder
            shutil.copy2(os.path.join(source_folder, file), dest_folder_train)

def split_data_folderB_folderD(source_folder, test_partition_ids, dest_folder_test, dest_folder_train):
    # Create destination folders if they don't exist
    os.makedirs(dest_folder_test, exist_ok=True)
    os.makedirs(dest_folder_train, exist_ok=True)
    
    # Get a list of files in the source folder
    files = os.listdir(source_folder)
    
    # Iterate over each file in the source folder
    for file in files:
        filename = os.path.splitext(file)[0]  # Remove file extension
        
        # Extract x from the filename
        x, _, _ = filename.split('_')
        
        # Check if x is in the test partition ids
        if x in test_partition_ids:
            # Copy the file to the test folder
            shutil.copy2(os.path.join(source_folder, file), dest_folder_test)
        else:
            # Copy the file to the train folder
            shutil.copy2(os.path.join(source_folder, file), dest_folder_train)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("A")
    parser.add_argument("A1")  
    args = parser.parse_args()
    
    # Set the source folders and destination folders
    folder_A = args.A 
    folder_C_test = os.path.join(args.A1, "test")
    folder_C_train = os.path.join(args.A1, "train")
    
    # Set the number of IDs for the test partition
    num_test_ids = 200
    
    # Generate a list of all possible IDs (000 to 999)
    all_ids = [str(i).zfill(3) for i in range(1000)]
    
    # Randomly select IDs for the test partition
    # test_partition_ids = random.sample(all_ids, num_test_ids)
    np.random.seed(0)
    idx = np.arange(len(all_ids))
    np.random.shuffle(idx)
    idx = idx[: num_test_ids]
    test_partition_ids = np.stack(all_ids)[idx]
    
    if "original" not in folder_A:
        # Split data from folder A to folder C
        split_data_folderA_folderC(folder_A, test_partition_ids, folder_C_test, folder_C_train)
    else:
        # Split data from folder B to folder D
        split_data_folderB_folderD(folder_A, test_partition_ids, folder_C_test, folder_C_train)
