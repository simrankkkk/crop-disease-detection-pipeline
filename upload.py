# step1.py — Upload original dataset and print dataset ID
from clearml import Dataset, Task

task = Task.init(project_name="VisiblePipeline", task_name="step_to_upload")
print("🔗 Connected to ClearML")

# Load dataset (already uploaded manually once, so just get it again)
dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
dataset_path = dataset.get_local_copy()
print("✅ Dataset path:", dataset_path)

# Pass dataset ID to next step
print(f"OUTPUT_DATASET_ID={dataset.id}")
