# step2split.py — Preprocessing with ClearML (train/valid/test split)
from clearml import Task, Dataset
from pathlib import Path
from PIL import Image
import os, shutil
from sklearn.model_selection import train_test_split

# Initialize ClearML task
task = Task.init(project_name="PlantPipeline", task_name="Step 2 - Preprocess with Split", task_type=Task.TaskTypes.data_processing)

# Get input dataset
dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
raw_path = Path(dataset.get_local_copy())
print("✅ Raw dataset loaded from:", raw_path)

# Output directory
out = Path("processed_data")
temp = out / "temp"
if out.exists(): shutil.rmtree(out)
temp.mkdir(parents=True, exist_ok=True)

# Resize all to 224x224 into temp/class folders
for cls_dir in (raw_path / "train").iterdir():
    if not cls_dir.is_dir(): continue
    cls_out = temp / cls_dir.name
    cls_out.mkdir(parents=True, exist_ok=True)
    for img_path in cls_dir.glob("*.*"):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
        try:
            img = Image.open(img_path).convert("RGB").resize((224,224))
            img.save(cls_out / img_path.name)
        except Exception as e:
            print(f"⚠️ Skipped {img_path.name}: {e}")

# Create train/valid/test folders
for split in ("train", "valid", "test"):
    (out / split).mkdir(parents=True, exist_ok=True)

# Split and distribute
for cls_dir in temp.iterdir():
    images = list(cls_dir.glob("*.*"))
    train_imgs, testval_imgs = train_test_split(images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(testval_imgs, test_size=0.5, random_state=42)

    for split_name, split_imgs in zip(["train", "valid", "test"], [train_imgs, val_imgs, test_imgs]):
        dest = out / split_name / cls_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for img in split_imgs:
            shutil.copy(img, dest / img.name)

shutil.rmtree(temp)

# Upload as ClearML dataset
processed = Dataset.create(dataset_name="plant_processed_data_split", dataset_project="PlantPipeline")
processed.add_files(str(out))
processed.upload()
processed.finalize()
task.close()
print("✅ Processed dataset uploaded with train/valid/test folders.")
