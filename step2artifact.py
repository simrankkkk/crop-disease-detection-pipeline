from clearml import Task, Dataset
from pathlib import Path
from PIL import Image
import shutil

def preprocess_images(input_dir: str, output_dir: str, image_size=(224, 224)) -> str:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    for class_dir in input_path.iterdir():
        if not class_dir.is_dir():
            continue
        dst_class_dir = output_path / class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            try:
                with Image.open(img_file) as img:
                    img = img.convert("RGB")
                    img = img.resize(image_size)
                    img.save(dst_class_dir / img_file.name)
            except Exception as e:
                print(f"⚠️ Could not process {img_file}: {e}")

    return str(output_path.resolve())

# 🚀 ClearML Task
task = Task.init(project_name="PlantPipeline", task_name="stage_preprocess")

# 🧠 Get uploaded dataset
dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")  # Your existing dataset
dataset_path = dataset.get_local_copy()

# ⚙️ Preprocess images
output_dir = "processed_data"
preprocessed_path = preprocess_images(dataset_path, output_dir)

# 📦 Upload preprocessed dataset
task.upload_artifact(name="preprocessed_dataset", artifact_object=preprocessed_path)

print("✅ Step 2 preprocessing completed.")
task.close()
