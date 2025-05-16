# finalstep1.py

from clearml import Dataset, Task

# ✅ Initialize ClearML Task
task = Task.init(
    project_name="FinalProject",
    task_name="final_step_upload",
    task_type=Task.TaskTypes.data_processing
)

# ✅ Load or reference dataset that was manually uploaded or registered earlier
dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")

# ✅ Optionally download locally to trigger the caching
local_path = dataset.get_local_copy()
print("✅ Dataset successfully fetched to local path:", local_path)

# ✅ Pass dataset ID as a pipeline parameter
task.set_parameter("dataset_id", dataset.id)  # <- This makes it available downstream

# ✅ Done
print(f"📌 Logged dataset_id to pipeline: {dataset.id}")
task.close()
