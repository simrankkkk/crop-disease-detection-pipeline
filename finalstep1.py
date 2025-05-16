from clearml import Dataset, Task

# ✅ Initialize ClearML Task
task = Task.init(
    project_name="FinalProject",
    task_name="final_step_upload",
    task_type=Task.TaskTypes.data_processing
)

# ✅ Reference your uploaded raw dataset
dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")

# ✅ Optionally download locally (for debug or inspection)
local_path = dataset.get_local_copy()
print("✅ Dataset successfully fetched to local path:", local_path)

# ✅ Pass dataset_id to next step via pipeline
task.set_parameter("dataset_id", dataset.id)
task.get_logger().report_text(f"📌 dataset_id registered for pipeline: {dataset.id}")

task.close()
