# step2.py — Data Preprocessing (Torch-Free, Pipeline-Compatible)

import os
import shutil
from clearml import Task, Dataset
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# 🚀 Connect to ClearML Task
task = Task.init(project_name="PlantPipeline", task_name="step2 preprocessing")
dataset_path = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71").get_local_copy()

# ✅ Define output path
output_dir = os.path.join(os.getcwd(), "processed")
os.makedirs(output_dir, exist_ok=True)

# ✅ Prepare data
image_paths = []
labels = []

for class_dir in os.listdir(dataset_path + "/train"):
    class_path = os.path.join(dataset_path, "train", class_dir)
    if os.path.isdir(class_path):
        for fname in os.listdir(class_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(class_path, fname))
                labels.append(class_dir)

# 🔀 Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# 🧼 Rescale & Save
def preprocess_and_save(img_path, label, split):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))  # ✅ same as AIS notebook
    target_dir = os.path.join(output_dir, split, label)
    os.makedirs(target_dir, exist_ok=True)
    img.save(os.path.join(target_dir, os.path.basename(img_path)))

for img_path, label in zip(X_train, y_train):
    preprocess_and_save(img_path, label, "train")

for img_path, label in zip(X_val, y_val):
    preprocess_and_save(img_path, label, "valid")

# ✅ Log artifacts and finish
print(f"✅ Preprocessed data saved to: {output_dir}")
task.upload_artifact(name="processed_dataset", artifact_object=output_dir)
task.close()
