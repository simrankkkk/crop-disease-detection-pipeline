from clearml import Task, Dataset
from pathlib import Path
from PIL import Image
import shutil

def organize_and_preprocess_dataset(dataset_path: str, output_dir: str, image_size=(224, 224)) -> str:
    train_input = Path(dataset_path) / "train"
    valid_input = Path(dataset_path) / "valid"
    train_output = Path(output_dir) / "train"
    valid_output = Path(output_dir) / "valid"

    def process_images(src_dir, dst_dir):
        for class_dir in src_dir.iterdir():
            if not class_dir.is_dir():
                continue
            dst_class = dst_dir / class_dir.name
            dst_class.mkdir(parents=True, exist_ok=True)
            for image_file in class_dir.iterdir():
                if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                try:
                    with Image.open(image_file) as img:
                        img = img.convert("RGB")
                        img = img.resize(image_size)
                        img.save(dst_class / image_file.name)
                except Exception as e:
                    print(f"⚠️ Could not process {image_file}: {e}")

    train_output.mkdir(parents=True, exist_ok=True)
    valid_output.mkdir(parents=True, exist_ok=True)
    process_images(train_input, train_output)
    process_images(valid_input, valid_output)

    return str(Path(output_dir).resolve())

# 🚀 ClearML Task
task = Task.init(project_name="PlantPipeline", task_name="step2 - preprocessing")

# 🧠 Get dataset from ClearML
dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
dataset_path = dataset.get_local_copy()

# ⚙️ Run preprocessing
preprocessed_output = organize_and_preprocess_dataset(dataset_path, output_dir="processed_data")

# 📦 Upload artifact
task.upload_artifact(name="preprocessed_dataset", artifact_object=preprocessed_output)

print("✅ Step 2 completed. Preprocessed dataset saved and uploaded to ClearML.")
task.close()
