# finalstep1.py

from clearml import Dataset, Task

# ✅ Initialize ClearML Task
task = Task.init(
    project_name="FinalProject",
    task_name="final_step_upload",
    task_type=Task.TaskTypes.data_processing
)

# ✅ Replace this with your actual dataset ID (or create new one if needed)
# If you've already uploaded the dataset manually, just reference it:
dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")

# ✅ Download dataset locally
local_path = dataset.get_local_copy()
print("✅ Dataset successfully fetched to local path:", local_path)

task.close()
