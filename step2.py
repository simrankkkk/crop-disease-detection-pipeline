# clearml_requirements: torchvision, numpy, pillow, matplotlib

from clearml import Task, Dataset
import os
from torchvision import datasets, transforms

# ✅ Start ClearML task
task = Task.init(
    project_name="plantdataset",
    task_name="Step 2 - Data Preprocessing (No Torch)"
)

# ✅ Load dataset from ClearML
dataset = Dataset.get(
    dataset_name="New Augmented Plant Disease Dataset",
    dataset_project="plantdataset"
)
dataset_path = dataset.get_local_copy()

# ✅ Define preprocessing transforms
image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

# ✅ Load image datasets
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "valid")

train_dataset = datasets.ImageFolder(train_dir, transform=image_transforms["train"])
val_dataset = datasets.ImageFolder(val_dir, transform=image_transforms["val"])

print("✅ Preprocessing complete")
print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
print(f"Classes: {train_dataset.classes}")
