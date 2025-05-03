# step2split.py — Split dataset and accept dataset ID as input
import sys, os, shutil, random
from collections import defaultdict
from clearml import Dataset, Task

dataset_id = sys.argv[1] if len(sys.argv) > 1 else None
assert dataset_id, "❌ Dataset ID must be provided as argument"

task = Task.init(project_name="VisiblePipeline", task_name="step_to_preprocess")
print(f"🔗 Connected to ClearML | Dataset ID: {dataset_id}")

dataset = Dataset.get(dataset_id=dataset_id)
local_path = dataset.get_local_copy()

# Split logic
output_base = "split_dataset"
ratios = {"train": 0.7, "valid": 0.15, "test": 0.15}
if os.path.exists(output_base): shutil.rmtree(output_base)
os.makedirs(output_base)

class_images = defaultdict(list)
for root, _, files in os.walk(local_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            class_name = os.path.basename(root)
            class_images[class_name].append(os.path.join(root, file))

for cls, imgs in class_images.items():
    random.shuffle(imgs)
    total = len(imgs)
    n_train = int(total * ratios["train"])
    n_valid = int(total * ratios["valid"])
    subsets = {
        "train": imgs[:n_train],
        "valid": imgs[n_train:n_train+n_valid],
        "test": imgs[n_train+n_valid:]
    }
    for split, paths in subsets.items():
        split_dir = os.path.join(output_base, split, cls)
        os.makedirs(split_dir, exist_ok=True)
        for p in paths:
            shutil.copy(p, split_dir)

# Register split dataset
ds = Dataset.create(dataset_name="plant_processed_data_split", dataset_project="VisiblePipeline", parent_datasets=[dataset_id])
ds.add_files(output_base)
ds.upload()
ds.finalize()
print(f"✅ OUTPUT_DATASET_ID={ds.id}")
