import os
import random
import numpy as np
from PIL import Image
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--A", default="images/noise/dwtDct", type=str) # wm noise
parser.add_argument("--B", default="images/imagenet/clean", type=str) # clean
parser.add_argument("--C", default="images/spoofed", type=str)
parser.add_argument("--beta", default=0.7, type=float)
args = parser.parse_args()

# Paths to folders
folder_A = args.A
folder_B = args.B
folder_C = os.path.join(args.C, folder_A.split('/')[-1])

if not os.path.exists(folder_C):
    os.makedirs(folder_C)

# Get the list of image files in folders A and B
files_A = os.listdir(folder_A)
files_B = os.listdir(folder_B)

beta = args.beta

# Process each image in folder B
for file_B in files_B:
    
    image_B = Image.open(os.path.join(folder_B, file_B))
    
    file_A = random.choice(files_A)
    while not file_A.endswith('.png'):
        file_A = random.choice(files_A)
    image_A = Image.open(os.path.join(folder_A, file_A))
    
    if image_B.size != image_A.size:
        image_B = image_B.resize(image_A.size)

    # # Convert the images to NumPy arrays
    np_image_A = np.array(image_A).astype('float32') 
    np_image_B = np.array(image_B).astype('float32') * (beta)
    M = np.ceil(255 - np.max(np_image_B))
    np_image_A = np_image_A / np_image_A.max() * M

    # Perform element-wise addition of the pixel values
    np_merged_image = np_image_A + np_image_B
    np_merged_image = np.clip(np_merged_image, 0, 255)

    # Convert the merged image back to PIL Image
    merged_image = Image.fromarray(np_merged_image.astype(np.uint8))
    new_image_path = os.path.join(folder_C, f'{file_B}')
    merged_image.save(new_image_path)