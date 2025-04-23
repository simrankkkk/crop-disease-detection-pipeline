# clearml_requirements: numpy, pillow, matplotlib

from clearml import Task, Dataset
import os
from PIL import Image
import numpy as np

# ✅ Start ClearML task
task = Task.init(
    project_name="plantdataset",
    task_name="Step 2 - Pillow Preprocessing (No Torch)"
)

# ✅ Load dataset from ClearML
dataset = Dataset.get(dataset_name="New Augmented Plant Disease Dataset")
dataset_path = dataset.get_local_copy()

# ✅ Just preview some images to simulate "processing"
train_dir = os.path.join(dataset_path, "train")

# List 3 sample images for confirmation
count = 0
for root, dirs, files in os.walk(train_dir):
    for file in files:
        if file.endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root, file)
            img = Image.open(path).resize((224, 224))
            img_array = np.array(img)
            print(f"✅ {file} → shape: {img_array.shape}")
            count += 1
        if count >= 3:
            break
    if count >= 3:
        break

print("✅ Step 2 complete — previewed sample images.")
