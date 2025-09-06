import os
import shutil
from sklearn.model_selection import train_test_split

INPUT_DIR = "image_data"   
OUTPUT_DIR = "dataset" 

def split_dataset(input_dir = INPUT_DIR, output_dir = OUTPUT_DIR, train_percentage = 0.75, seed = 73):
    """
    Takes an input dataset from a folder (default: "image_data") containing labelled class folders where each image in 
    the class folder is of that class. This function outputs a new folder (default: "dataset") containing 2 directories: train_set and
    test_set. train_set contains, by default, 75% of the data while test_set contains 25%. Both folders contain the original labelled class folders.
    """
    # If output_dir already exists, remove it completely before recreating
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok = True)

    # Get class names from folders inside of 'image_data'
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    #print("Classes found:", classes)

    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpeg'))]

        train_files, test_files = train_test_split(images, train_size = train_percentage, random_state = seed)

        # Copy images into new directories
        def copy_files(file_list, split_name):
            split_path = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_path, exist_ok = True)
            for f in file_list:
                shutil.copy(os.path.join(cls_dir, f), os.path.join(split_path, f))

        # Copy to folders
        copy_files(train_files, "train_set")
        copy_files(test_files, "test_set")