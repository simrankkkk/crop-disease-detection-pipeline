import os
import shutil
import random
from clearml import Task, Dataset

# Task init
task = Task.init(project_name="VisiblePipeline", task_name="step2split")
dataset = Dataset.get(dataset_name="New Augmented Plant Disease Dataset", dataset_project=".", only_completed=True)
local_path = dataset.get_local_copy()

# Output folder
processed_output_folder = os.path.join(local_path, "split_dataset")
if os.path.exists(processed_output_folder):
    shutil.rmtree(processed_output_folder)
os.makedirs(processed_output_folder)

# Folder setup
splits = ['train', 'valid', 'test']
class_folders = [d for d in os.listdir(local_path) if d.startswith("train")]

# Shuffle and split
for class_folder in class_folders:
    full_path = os.path.join(local_path, class_folder)
    files = os.listdir(full_path)
    random.shuffle(files)

    n_total = len(files)
    n_train = int(0.7 * n_total)
    n_valid = int(0.15 * n_total)

    split_data = {
        'train': files[:n_train],
        'valid': files[n_train:n_train + n_valid],
        'test': files[n_train + n_valid:]
    }

    for split in splits:
        dest_dir = os.path.join(processed_output_folder, split, class_folder)
        os.makedirs(dest_dir, exist_ok=True)
        for file in split_data[split]:
            shutil.copy(os.path.join(full_path, file), os.path.join(dest_dir, file))

# ✅ Log class counts
def count_files(folder):
    count_dict = {}
    total = 0
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            num_files = len(os.listdir(class_path))
            count_dict[class_name] = num_files
            total += num_files
    return count_dict, total

logger = task.get_logger()
for split in splits:
    split_path = os.path.join(processed_output_folder, split)
    log_output = [f"\n📂 {split.upper()} SET"]
    class_counts, total = count_files(split_path)
    for cls, count in class_counts.items():
        log_output.append(f"  - {cls}: {count} files")
    log_output.append(f"✅ Total: {total} files")
    for line in log_output:
        print(line)
        logger.report_text(line)

# Upload to ClearML
output_dataset = Dataset.create(
    dataset_name="split_dataset",
    dataset_project="VisiblePipeline",
    parent_datasets=[dataset.id]
)
output_dataset.add_files(processed_output_folder)
output_dataset.finalize()
