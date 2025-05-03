from clearml import Task, Dataset
import shutil
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import random

def split_dataset(source_dir, output_dir, test_ratio=0.1, val_ratio=0.1):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ['train', 'valid', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    for cls in os.listdir(source_dir):
        cls_path = source_dir / cls
        if not cls_path.is_dir():
            continue
        images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.png"))
        random.shuffle(images)

        test_size = int(len(images) * test_ratio)
        val_size = int(len(images) * val_ratio)
        test_files = images[:test_size]
        val_files = images[test_size:test_size + val_size]
        train_files = images[test_size + val_size:]

        for split_name, files in zip(['train', 'valid', 'test'], [train_files, val_files, test_files]):
            dest_dir = output_dir / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for file in files:
                try:
                    img = Image.open(file).convert("RGB")
                    img.save(dest_dir / file.name)
                except Exception as e:
                    print(f"⚠️ Could not process {file}: {e}")

    # Debug print: file counts
    for split in ['train', 'valid', 'test']:
        count = len(list((output_dir / split).rglob("*.jpg")))
        print(f"✅ {split} contains {count} images")

if __name__ == "__main__":
    task = Task.init(project_name="VisiblePipeline", task_name="step_preprocess", task_type=Task.TaskTypes.data_processing)

    dataset_id = Task.current_task().get_parameters_as_dict().get("General/dataset_task_id")
    print("🔍 Getting dataset ID:", dataset_id)

    raw_dataset = Dataset.get(dataset_id=dataset_id)
    source_path = raw_dataset.get_local_copy()

    output_dir = "processed_split"
    split_dataset(source_path, output_dir)

    ds = Dataset.create(dataset_name="dataset_split", dataset_project="VisiblePipeline")
    ds.add_files(output_dir)
    ds.upload()
    ds.finalize()
    print("✅ Dataset split and uploaded successfully.")
