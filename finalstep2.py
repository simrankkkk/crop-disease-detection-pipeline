# finalstep2.py

from clearml import Dataset, Task
import os
import shutil
import random
from collections import defaultdict

# ✅ Initialize ClearML Task
task = Task.init(
    project_name="FinalProject",
    task_name="final_step_preprocess",
    task_type=Task.TaskTypes.data_processing
)

# ✅ Extract dataset_id from artifact (passed from previous step)
artifact = task.get_parameters().get("Args/dataset_id")
if isinstance(artifact, dict) and "preview" in artifact:
    dataset_id = artifact["preview"]
else:
    raise ValueError("Invalid dataset_id artifact format")

dataset = Dataset.get(dataset_id=dataset_id)
local_path = dataset.get_local_copy()
print(f"📂 Dataset downloaded to: {local_path}")

# ✅ Define split ratios
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15

# ✅ Create output directories
output_base = "./split_dataset"
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

# ✅ Organize images by class
class_images = defaultdict(list)
for root, _, files in os.walk(local_path):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            class_name = os.path.basename(root)
            class_images[class_name].append(os.path.join(root, file))

# ✅ Split and move files
for class_name, file_list in class_images.items():
    random.shuffle(file_list)
    total = len(file_list)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    subsets = {
        "train": file_list[:train_end],
        "valid": file_list[train_end:valid_end],
        "test": file_list[valid_end:]
    }

    for split_name, paths in subsets.items():
        split_class_dir = os.path.join(output_base, split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for path in paths:
            shutil.copy(path, split_class_dir)

# ✅ Log file counts
for split in ["train", "valid", "test"]:
    total_files = 0
    print(f"\n📦 {split.upper()} split:")
    split_dir = os.path.join(output_base, split)
    for class_dir in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_dir)
        count = len(os.listdir(class_path))
        total_files += count
        print(f"  • {class_dir}: {count} files")
    print(f"  Total {split} files: {total_files}")

# ✅ Upload the processed dataset to ClearML
print("\n🚀 Uploading split dataset to ClearML...")
new_dataset = Dataset.create(
    dataset_name="final_processed_data_split",
    dataset_project="FinalProject",
    parent_datasets=[dataset.id]
)
new_dataset.add_files(path=output_base)
new_dataset.upload()
new_dataset.finalize()
print("✅ Dataset successfully split and uploaded.")

task.close()
