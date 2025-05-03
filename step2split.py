import os
import shutil
import random
from clearml import Task, Dataset

# ✅ ClearML Task Init
task = Task.init(project_name="VisiblePipeline", task_name="step_preprocess", task_type=Task.TaskTypes.data_processing)
logger = task.get_logger()

# ✅ Dataset Retrieval
DATASET_ID = "105163c10d0a4bbaa06055807084ec71"
dataset = Dataset.get(dataset_id=DATASET_ID)
local_dataset_path = dataset.get_local_copy()

# ✅ Output Directory
output_dir = os.path.join(local_dataset_path, "..", "plant_processed_split")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# ✅ Split Ratios
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
random.seed(42)

# ✅ Loop over class folders
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

    # Adjust if too much
    if train_count + val_count + test_count > total:
        train_count -= (train_count + val_count + test_count - total)

    splits = {
        "train": files[:train_count],
        "val": files[train_count:train_count + val_count],
        "test": files[train_count + val_count:]
    }

    for split_name, split_files in splits.items():
        split_dir = os.path.join(output_dir, split_name, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for f in split_files:
            shutil.copy(os.path.join(class_path, f), os.path.join(split_dir, f))

        # ✅ Log per split
        log_line = f"{split_name.upper()} - {class_name}: {len(split_files)} images"
        print(log_line)
        logger.report_text(log_line)

    # ✅ Log total per class
    summary_line = f"{class_name} TOTAL: {total} → Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}"
    print(summary_line)
    logger.report_text(summary_line)

# ✅ Upload to ClearML
output_dataset = Dataset.create(
    dataset_name="plant_processed_data_split",
    dataset_project="VisiblePipeline",
    parent_datasets=[dataset.id],
)
output_dataset.add_files(output_dir)
output_dataset.upload()
output_dataset.finalize()

logger.report_text("✅ Dataset successfully split and uploaded.")
print("✅ Dataset successfully split and uploaded.")

task.close()
