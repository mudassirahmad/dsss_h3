import os
from PIL import Image
import json
import random

import matplotlib.pyplot as plt
import numpy as np

dataset_folder = "./Mini_BAGLS_dataset/"

data_dict={"images": [], "segmentation": [],"metadata":[]}

# List all metadata files in the dataset folder
metadata_files = [file_name for file_name in os.listdir(dataset_folder) if file_name.endswith(".meta")]

# Choose four random metadata files
selected_files = random.sample(metadata_files, 4)

for file_name in selected_files:
    
    # Load segmentation mask
    if (file_name.endswith(".meta")):
        
        # Load metadata in sub-dictionary format
        metadata_path = os.path.join(dataset_folder, file_name)

        with open(metadata_path, 'r') as json_file:
            meta = json.load(json_file)

        data_dict["metadata"].append(meta)
    
        segmentation_file_name= file_name.replace(".meta","_seg.png")
        mask_path = os.path.join(dataset_folder, segmentation_file_name)
        mask = Image.open(mask_path)
        data_dict["segmentation"].append(mask)

        # Load image
        image_file_name= file_name.replace(".meta",".png")
        img_path = os.path.join(dataset_folder, image_file_name)
        img = Image.open(img_path)
        data_dict["images"].append(img)

# Plotting images with segmentation masks overlaid using subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i in range(len(data_dict["segmentation"])):
    # Plot image
    axs[i // 2, i % 2].imshow(np.array(data_dict["segmentation"][i]))
    axs[i // 2, i % 2].set_title(data_dict["metadata"][i]["Subject disorder status"])

    # Overlay segmentation mask
    axs[i // 2, i % 2].imshow(np.array(data_dict["segmentation"][i]), cmap='jet', alpha=0.5)


plt.show()
