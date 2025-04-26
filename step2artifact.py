from clearml import Task, Dataset
from pathlib import Path
from PIL import Image

# 🚀 Initialize ClearML task
task = Task.init(project_name="PlantPipeline", task_name="stage_preprocess")

# 📦 Load your dataset
dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
dataset_path = dataset.get_local_copy(force=True)  # ✅ force=True ensures fresh download

print("✅ Dataset root contents:")
for item in Path(dataset_path).iterdir():
    print("-", item.name)  # ✅ SHOWS EXACT STRUCTURE

# 🛠 Preprocessing
output_dir = Path("processed_data")
output_dir.mkdir(parents=True, exist_ok=True)

for class_dir in Path(dataset_path).iterdir():
    if not class_dir.is_dir():
        continue
    output_class_dir = output_dir / class_dir.name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    for img_file in class_dir.glob("*"):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        try:
            img = Image.open(img_file).convert("RGB").resize((224, 224))
            img.save(output_class_dir / img_file.name)
        except Exception as e:
            print(f"⚠️ Error processing {img_file}: {e}")

# 📤 Upload preprocessed artifact
task.upload_artifact(name="preprocessed_dataset", artifact_object=str(output_dir))
print("✅ Stage preprocess completed successfully!")
task.close()
