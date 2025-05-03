import os
import shutil
import random
from clearml import Task, Dataset
from PIL import Image

# Initialize ClearML task
task = Task.init(project_name="VisiblePipeline", task_name="step2split")
dataset = Dataset.get(dataset_name="New Augmented Plant Disease Dataset", dataset_project=".", only_completed=True)
local_path = dataset.get_local_copy()

# Output folder
processed_output_folder = os.path.join(local_path, "split_dataset")
if os.path.exists(processed_output_folder):
    shutil.rmtree(processed_output_folder)
os.makedirs(processed_output_folder)

# Constants
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
MIN_VAL = 1
MIN_TEST = 1
random.seed(42)

# Identify class folders inside "train/"
train_root = os.path.join(local_path, "train")
class_folders = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]

# Split loop
for class_folder in class_folders:
    full_path = os.path.join(train_root, class_folder)
    files = [f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)

    n_total = len(files)
    n_test = max(MIN_TEST, int(TEST_SPLIT * n_total))
    n_valid = max(MIN_VAL, int(VAL_SPLIT * (n_total - n_test)))
    n_train = n_total - n_test - n_valid

    test_files = files[:n_test]
    valid_files = files[n_test:n_test + n_valid]
    train_files = files[n_test + n_valid:]

    split_map = {
        'train': train_files,
        'valid': valid_files,
        'test': test_files
    }

    for split_name, file_list in split_map.items():
        dest_dir = os.path.join(processed_output_folder, split_name, class_folder)
        os.makedirs(dest_dir, exist_ok=True)
        for file in file_list:
            shutil.copy(os.path.join(full_path, file), os.path.join(dest_dir, file))

# ✅ Log counts to ClearML console
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
for split in ['train', 'valid', 'test']:
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
output_dataset.upload()
output_dataset.finalize()
print("✅ New split dataset uploaded successfully.")
