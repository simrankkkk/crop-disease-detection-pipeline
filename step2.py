
import os
import zipfile
import shutil
from clearml import Task, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import random

def main():
    # Connect to ClearML Task
    task = Task.init(project_name='PlantPipeline', task_name='step2-preprocess-data', task_type=Task.TaskTypes.data_processing)

    # Get dataset uploaded to ClearML
    dataset = Dataset.get(dataset_id='105163c10d0a4bbaa06055807084ec71')
    local_dataset_path = dataset.get_local_copy()

    # Define output directory
    output_dir = os.path.join(os.getcwd(), 'processed_data')
    os.makedirs(output_dir, exist_ok=True)

    # Define train/valid folders
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Preprocess logic (80-20 split maintaining subfolder structure)
    np.random.seed(42)
    random.seed(42)
    all_classes = os.listdir(local_dataset_path)

    for class_name in all_classes:
        class_path = os.path.join(local_dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        images = os.listdir(class_path)
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)

        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in val_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(valid_dir, class_name, img))

    # Return path for training step
    return output_dir

if __name__ == "__main__":
    main()
