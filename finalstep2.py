from clearml import Dataset, Task
import os
import shutil
import random
from collections import defaultdict

# ✅ Initialize task
task = Task.init(project_name="FinalProject", task_name="final_step_preprocess")

# ✅ Get dataset ID from previous step
params = task.get_parameters()
DATASET_ID = params.get("Args/dataset_id") or os.environ.get("DATASET_ID")

# Optional hardcoded fallback for debugging outside pipeline
if not DATASET_ID:
    DATASET_ID = "105163c10d0a4bbaa06055807084ec71"  # fallback raw dataset ID
    print("⚠️ Warning: Using fallback dataset_id!")

if not DATASET_ID:
    raise ValueError("❌ No dataset_id provided via Args or environment.")

dataset = Dataset.get(dataset_id=DATASET_ID)

local_path = dataset.get_local_copy()
print(f"📂 Dataset downloaded to: {local_path}")

# ✅ Define split ratios
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15

# ✅ Prepare split output directories
output_base = "./split_dataset"
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

# ✅ Group images by class
class_images = defaultdict(list)
for root, _, files in os.walk(local_path):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            class_name = os.path.basename(root)
            class_images[class_name].append(os.path.join(root, file))

# ✅ Split and copy
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

# ✅ Log stats
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

# ✅ Upload new dataset
print("\n🚀 Uploading split dataset to ClearML...")
new_dataset = Dataset.create(
    dataset_name="final_preprocessing_split",
    dataset_project="FinalProject",
    parent_datasets=[dataset.id]
)
new_dataset.add_files(path=output_base)
new_dataset.upload()
new_dataset.finalize()
print("✅ Dataset successfully split and uploaded.")

# ✅ Register new dataset_id for next step
task.set_parameter("dataset_id", new_dataset.id)
task.get_logger().report_text(f"📌 Preprocessed dataset_id for training: {new_dataset.id}")

task.close()
