import os, shutil
from sklearn.model_selection import train_test_split

# Paths
input_dir = "image_data"
output_dir = "dataset"

os.makedirs(output_dir, exist_ok = True)

train_percentage = 0.75

# Get class names from folders inside of 'image_data'
classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
#print("Classes found:", classes)

for cls in classes:
    cls_dir = os.path.join(input_dir, cls)
    images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpeg'))]

    train_files, test_files = train_test_split(images, train_size = train_percentage, random_state = 73)

    # Copy images into new directories
    def copy_files(file_list, split_name):
        split_path = os.path.join(output_dir, split_name, cls)
        os.makedirs(split_path, exist_ok = True)
        for f in file_list:
            shutil.copy(os.path.join(cls_dir, f), os.path.join(split_path, f))

    # Copy to folders
    copy_files(train_files, "train_set")
    copy_files(test_files, "test_set")