from clearml import Task, Dataset
import os, shutil, random
from pathlib import Path
from PIL import Image

# Connect to ClearML
task = Task.init(project_name="VisiblePipeline", task_name="step_preprocess", task_type=Task.TaskTypes.data_processing)

# Hardcode your dataset ID here (from ClearML)
DATASET_ID = "105163c10d0a4bbaa06055807084ec71"  # Replace if needed

# Get dataset and download
raw_dataset = Dataset.get(dataset_id=DATASET_ID)
source_path = Path(raw_dataset.get_local_copy())
print("✅ Dataset downloaded to:", source_path)

# Folder structure expected: train/ClassName/xxx.jpg
original_train_folder = source_path / "train"
if not original_train_folder.exists():
    raise Exception("❌ Could not find 'train' folder in downloaded dataset")

# Output structure
output_dir = Path("processed_split")
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir()

# Split percentages
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
random.seed(42)

# Process each class
for class_dir in original_train_folder.iterdir():
    if not class_dir.is_dir():
        continue
    images = list(class_dir.glob("*.jpg"))
    random.shuffle(images)

    total = len(images)
    test_size = int(TEST_SPLIT * total)
    val_size = int(VAL_SPLIT * (total - test_size))

    test_imgs = images[:test_size]
    val_imgs = images[test_size:test_size + val_size]
    train_imgs = images[test_size + val_size:]

    for split, img_list in zip(["train", "valid", "test"], [train_imgs, val_imgs, test_imgs]):
        target_dir = output_dir / split / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for img_path in img_list:
            try:
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                img.save(target_dir / img_path.name)
            except Exception as e:
                print(f"⚠️ Skipped {img_path.name}: {e}")

# Upload new split dataset
split_dataset = Dataset.create(
    dataset_name="dataset_split_processed",
    dataset_project="VisiblePipeline",
    parent_datasets=[DATASET_ID]
)
split_dataset.add_files(str(output_dir))
split_dataset.upload()
split_dataset.finalize()
print("✅ Uploaded processed dataset:", split_dataset.id)
