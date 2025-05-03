# step2split.py — Updated to avoid KeyError on 'General'
from clearml import Dataset, Task
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
import shutil
import os
import random

# ✅ Init ClearML Task
Task.init(
    project_name="VisiblePipeline",
    task_name="step_preprocess",
    task_type=Task.TaskTypes.data_processing
)

# ✅ Fetch raw dataset passed from step1
params = Task.current_task().get_parameters()
dataset_id = params.get("dataset_task_id")
dataset = Dataset.get(dataset_id=dataset_id)
input_path = Path(dataset.get_local_copy())
print("📂 Raw dataset located at:", input_path)

# ✅ Output path
output_dir = Path("processed_split")
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

# ✅ Prepare subfolders
splits = ['train', 'valid', 'test']
for split in splits:
    (output_dir / split).mkdir()

# ✅ Split images (assumes subfolder per class)
for class_dir in input_path.glob("*"):
    if not class_dir.is_dir():
        continue
    images = list(class_dir.glob("*.[jp][pn]g"))
    random.shuffle(images)
    total = len(images)
    train_split = int(0.8 * total)
    val_split = int(0.9 * total)

    split_map = {
        "train": images[:train_split],
        "valid": images[train_split:val_split],
        "test":  images[val_split:]
    }

    for split, imgs in split_map.items():
        class_out = output_dir / split / class_dir.name
        class_out.mkdir(parents=True, exist_ok=True)
        for img_path in imgs:
            try:
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                img.save(class_out / img_path.name)
            except Exception as e:
                print(f"⚠️ Skipped {img_path.name}: {e}")

# ✅ Upload split dataset to ClearML
ds = Dataset.create(
    dataset_name="dataset_split",                  # <--- renamed here
    dataset_project="VisiblePipeline"
)
ds.add_files(str(output_dir))
ds.upload()
ds.finalize()
print("✅ New dataset uploaded:", ds.id)
