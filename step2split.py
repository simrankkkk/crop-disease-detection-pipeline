from clearml import Task, Dataset
import os
import shutil
import random

# Start a ClearML task
task = Task.init(project_name="VisiblePipeline", task_name="step_preprocess", task_type=Task.TaskTypes.data_processing)

# Get the ClearML dataset
dataset = Dataset.get(dataset_name="New Augmented Plant Disease Dataset", dataset_project="VisiblePipeline", only_completed=True)
local_path = dataset.get_local_copy()

# Define output dirs
output_base = "./output_dataset"
train_dir = os.path.join(output_base, "train")
val_dir = os.path.join(output_base, "val")
test_dir = os.path.join(output_base, "test")

# Create dirs
for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

# Gather all image paths
image_paths = []
for root, _, files in os.walk(local_path):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, file))

# Shuffle and split
random.shuffle(image_paths)
n_total = len(image_paths)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_images = image_paths[:n_train]
val_images = image_paths[n_train:n_train + n_val]
test_images = image_paths[n_train + n_val:]

# Copy files
def copy_images(img_list, target_dir):
    for path in img_list:
        rel_path = os.path.relpath(path, local_path)
        target_path = os.path.join(target_dir, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(path, target_path)

copy_images(train_images, train_dir)
copy_images(val_images, val_dir)
copy_images(test_images, test_dir)

# Log file counts
print(f"[INFO] Total images: {n_total}")
print(f"[INFO] Train images: {len(train_images)}")
print(f"[INFO] Validation images: {len(val_images)}")
print(f"[INFO] Test images: {len(test_images)}")

# Upload to ClearML as a new dataset
new_dataset = Dataset.create(dataset_name="dataset_split", dataset_project="VisiblePipeline")
new_dataset.add_files(output_base)
new_dataset.upload()
new_dataset.finalize()
