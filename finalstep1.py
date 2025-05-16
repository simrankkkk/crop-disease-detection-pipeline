# finalstep1.py

from clearml import Dataset, Task

# ✅ Initialize ClearML Task
task = Task.init(
    project_name="FinalProject",
    task_name="final_step_upload",
    task_type=Task.TaskTypes.data_processing
)

# ✅ Instead of uploading, pass the fixed dataset ID from GDrive-uploaded ClearML dataset
dataset_id = "105163c10d0a4bbaa06055807084ec71"
task.set_parameter("dataset_id", dataset_id)

# ✅ Optional: fetch locally (if needed for other debugging purposes)
dataset = Dataset.get(dataset_id=dataset_id)
local_path = dataset.get_local_copy()
print("✅ Dataset successfully fetched to local path:", local_path)

print(f"📌 Logged fixed dataset_id to pipeline: {dataset_id}")
task.close()
