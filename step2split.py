import os
import shutil
import random
from clearml import Task, Dataset

# ✅ ClearML Task Init
task = Task.init(project_name="VisiblePipeline", task_name="step_preprocess", task_type=Task.TaskTypes.data_processing)

# ✅ Dataset Retrieval
DATASET_ID = "105163c10d0a4bbaa06055807084ec71"
dataset = Dataset.get(dataset_id=DATASET_ID)
local_dataset_path = dataset.get_local_copy()

# ✅ Output Directory
output_dir = os.path.join(local_dataset_path, "..", "plant_processed_split")
os.makedirs(output_dir, exist_ok=True)

# ✅ Splitting Parameters
train_ratio, val_ratio, test_ratio = 0.8, 0.10, 0.10
random.seed(42)
log_lines = []

# ✅ Perform Split Per Class
for class_name in os.listdir(local_dataset_path):
    class_path = os.path.join(local_dataset_path, class_name)
    if not os.path.isdir(class_path):
        continue

    files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    total = len(files)
    if total == 0:
        continue

    random.shuffle(files)
    train_count = max(1, int(total * train_ratio))
    val_count = max(1, int(total * val_ratio)) if total >= 7 else 0
    test_count = total - train_count - val_count

    if train_count + val_count + test_count > total:
        train_count -= (train_count + val_count + test_count - total)

    splits = {
        "train": files[:train_count],
        "val": files[train_count:train_count + val_count],
        "test": files[train_count + val_count:]
    }

    for split, split_files in splits.items():
        split_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for f in split_files:
            shutil.copy(os.path.join(class_path, f), os.path.join(split_dir, f))
        log_lines.append(f"{class_name} - {split}: {len(split_files)} files")

    log_lines.append(f"{class_name} - Total: {total} | Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

# ✅ Upload New Dataset
output_dataset = Dataset.create(
    dataset_name="plant_processed_data_split",
    dataset_project="VisiblePipeline",
    parent_datasets=[dataset.id],
)
output_dataset.add_files(output_dir)
output_dataset.upload()
output_dataset.finalize()

# ✅ Log file counts
for line in log_lines:
    print(line)

task.close()
