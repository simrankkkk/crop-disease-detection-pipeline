# step1.py – Load existing ClearML dataset (no upload required)

from clearml import Task, Dataset

# ✅ Start ClearML task
Task.init(
    project_name="plantdataset",
    task_name="Load Augmented Dataset from ClearML"
)

# ✅ Get dataset already uploaded to ClearML
# This name must exactly match the dataset name in your ClearML Web UI
dataset = Dataset.get(dataset_name="Upload New Augmented Plant Disease Dataset")
dataset_path = dataset.get_local_copy()

print("✅ Dataset retrieved to local path:", dataset_path)
