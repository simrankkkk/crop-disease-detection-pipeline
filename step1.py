from clearml import Task, Dataset

# ✅ Start ClearML task
Task.init(
    project_name="plantdataset",
    task_name="Load Augmented Dataset from ClearML"
)

# ✅ Get dataset from ClearML
dataset = Dataset.get(
    dataset_name="New Augmented Plant Disease Dataset",
    dataset_project="plantdataset"
)
dataset_path = dataset.get_local_copy()

print("✅ Dataset retrieved to local path:", dataset_path)
