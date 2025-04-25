import os
import shutil
import numpy as np
from clearml import Task, Dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Step 1: Init ClearML task
task = Task.init(project_name="PlantPipeline", task_name="Step 2 - Data Preprocessing")
dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
dataset_path = dataset.get_local_copy()

# ✅ Step 2: Define output directory
output_dir = os.path.join(os.getcwd(), "processed_data")
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "valid")

# ✅ Step 3: Copy images from ClearML cache to output folder safely
for split in ["train", "valid"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    split_path = os.path.join(dataset_path, split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        dest_path = os.path.join(output_dir, split, class_name)
        os.makedirs(dest_path, exist_ok=True)
        for file in os.listdir(class_path):
            src_file = os.path.join(class_path, file)
            dst_file = os.path.join(dest_path, file)
            if os.path.isfile(src_file):
                shutil.copy(src_file, dst_file)

# ✅ Step 4: Rescale images using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

# ✅ Step 5: Save class index mapping
np.save(os.path.join(output_dir, "class_indices.npy"), train_gen.class_indices)

print("✅ Preprocessing completed. Processed data stored in:", output_dir)

task.close()
