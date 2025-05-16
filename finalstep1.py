from clearml import Dataset, Task

# ✅ Initialize ClearML Task
task = Task.init(
    project_name="FinalProject",
    task_name="final_step_upload",
    task_type=Task.TaskTypes.data_processing
)

# ✅ Replace this with your actual dataset ID (or create new one if needed)
dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")

local_path = dataset.get_local_copy()
print("✅ Dataset successfully fetched to local path:", local_path)

# ─── NEW ───────────────────────────────────────────────────────────────────────
# expose the ID so the pipeline can reference it
task.set_parameter("dataset_id", dataset.id)
task.get_logger().report_text(f"🔖 dataset_id = {dataset.id}")

# and close the task
task.close()
