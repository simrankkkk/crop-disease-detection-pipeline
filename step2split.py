from clearml import Task, Dataset
import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Connect to ClearML task
task = Task.init(project_name="VisiblePipeline", task_name="step_preprocess", task_type=Task.TaskTypes.data_processing)
params = Task.current_task().get_parameters_as_dict()

# Robust parsing of dataset_task_id
dataset_id = (
    params.get("General", {}).get("dataset_task_id")
    or params.get("dataset_task_id")
)

if not dataset_id:
    raise ValueError("❌ 'dataset_task_id' not found in task parameters.")

# Load dataset
raw_dataset = Dataset.get(dataset_id=dataset_id)
raw_dataset_path = raw_dataset.get_local_copy()

# Output directory for split dataset
output_dir = "./processed_split_dataset"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Split ratios
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
random_seed = 42

# Get class folders
class_dirs = [d for d in os.listdir(raw_dataset_path) if os.path.isdir(os.path.join(raw_dataset_path, d))]

for class_name in class_dirs:
    class_path = os.path.join(raw_dataset_path, class_name)
    image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Train/valid/test split
    train_val, test = train_test_split(image_files, test_size=TEST_SPLIT, random_state=random_seed)
    train, val = train_test_split(train_val, test_size=VAL_SPLIT / (1 - TEST_SPLIT), random_state=random_seed)

    for subset_name, subset in zip(['train', 'val', 'test'], [train, val, test]):
        subset_dir = os.path.join(output_dir, subset_name, class_name)
        os.makedirs(subset_dir, exist_ok=True)
        for file_path in subset:
            shutil.copy(file_path, os.path.join(subset_dir, os.path.basename(file_path)))

# Upload to ClearML
output_dataset = Dataset.create(
    dataset_name="dataset_split",
    dataset_project="VisiblePipeline",
    parent_datasets=[raw_dataset.id]
)
output_dataset.add_files(output_dir)
output_dataset.upload()
output_dataset.finalize()
print("✅ Dataset split and uploaded successfully.")
