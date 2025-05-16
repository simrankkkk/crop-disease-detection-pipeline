# finalstep1.py

from clearml import Dataset, Task

# ✅ Initialize ClearML Task
task = Task.init(
    project_name="FinalProject",
    task_name="final_step_upload",
    task_type=Task.TaskTypes.data_processing
)

# ✅ Use fixed dataset ID (already uploaded manually)
params = {"dataset_id": "105163c10d0a4bbaa06055807084ec71"}
params = task.connect(params)  # <-- ✅ Makes it accessible via pipeline parameter injection

# ✅ Optional: confirm local access
dataset = Dataset.get(dataset_id=params["dataset_id"])
local_path = dataset.get_local_copy()
print("✅ Dataset fetched locally:", local_path)

print(f"📌 dataset_id connected to pipeline: {params['dataset_id']}")
task.close()
