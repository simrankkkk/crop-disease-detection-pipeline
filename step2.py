# clearml_requirements: pillow, numpy, matplotlib, clearml

from clearml import Task, Dataset
from PIL import Image
import numpy as np
import os

# ✅ Start ClearML task
task = Task.init(
    project_name="plantdataset",
    task_name="Step 2 - Preprocessing Without Torch (AIS Personal)"
)

# ✅ Load dataset from ClearML
dataset = Dataset.get(dataset_name="New Augmented Plant Disease Dataset")
dataset_path = dataset.get_local_copy()

# Image resize parameters (AIS_Personal used 224x224)
resize_dim = (224, 224)

# Track image count and class summary
class_counts = {}
sample_shapes = []

# Process each image (preview-style)
for split in ['train', 'valid']:
    split_path = os.path.join(dataset_path, split)
    
    for class_folder in os.listdir(split_path):
        class_path = os.path.join(split_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        class_counts[class_folder] = class_counts.get(class_folder, 0)

        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")  # Ensure 3 channels
                        img = img.resize(resize_dim)
                        img_arr = np.array(img)
                        sample_shapes.append(img_arr.shape)
                        class_counts[class_folder] += 1
                except Exception as e:
                    print(f"⚠️ Skipping {img_file}: {e}")

print("✅ Preprocessing Summary:")
for cls, count in class_counts.items():
    print(f"Class: {cls} → {count} images")

if sample_shapes:
    print(f"Sample image shape: {sample_shapes[0]}")

print("✅ Step 2 complete – no torch, full AIS logic!")
