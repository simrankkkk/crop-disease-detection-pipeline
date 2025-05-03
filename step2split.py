from clearml import Task, Dataset
import os
import shutil
import random
from collections import defaultdict

# Connect to ClearML
task = Task.init(project_name="VisiblePipeline", task_name="step_preprocess", task_type=Task.TaskTypes.data_processing)

# Set dataset ID (your final version)
DATASET_ID = "105163c10d0a4bbaa06055807084ec71"

# Download dataset
dataset = Dataset.get(dataset_id=DATASET_ID)
dataset_path = dataset.get_local_copy()

# Output path
output_dir = "./split_dataset"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

# Clean previous runs
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create directories
for split in [train_dir, val_dir, test_dir]:
    os.makedirs(split, exist_ok=True)

# Group images by class from `train/` folder inside dataset
class_to_images = defaultdict(list)
source_train_dir = os.path.join(dataset_path, "train")

for root, _, files in os.walk(source_train_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            label = os.path.basename(root)
            class_to_images[label].append(os.path.join(root, file))

# Split and copy
split_counts = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}
for label, images in class_to_images.items():
    random.shuffle(images)
    n = len(images)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    split_map = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, img_list in split_map.items():
        split_label_dir = os.path.join(output_dir, split, label)
        os.makedirs(split_label_dir, exist_ok=True)
        for img_path in img_list:
            shutil.copy(img_path, os.path.join(split_label_dir, os.path.basename(img_path)))
            split_counts[split][label] += 1

# ✅ Print class counts
print("\nImage count per class per split:")
for split in split_counts:
    print(f"\n{split.upper()}:")
    for label, count in split_counts[split].items():
        print(f"  {label}: {count} images")

# Upload split dataset
new_dataset = Dataset.create(
    dataset_name="plant_processed_data_split",
    dataset_project="VisiblePipeline",
    parent_datasets=[dataset.id]
)
new_dataset.add_files(path=output_dir)
new_dataset.finalize()
print("✅ Dataset successfully split and uploaded.")
