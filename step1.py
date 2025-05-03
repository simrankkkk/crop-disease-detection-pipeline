from clearml import Dataset, Task

# ✅ Init task under new project
Task.init(
    project_name="VisiblePipeline",
    task_name="step_upload",
    task_type=Task.TaskTypes.data_processing
)

# ✅ Replace this with your actual dataset ID (or create new one if needed)
dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")

local_path = dataset.get_local_copy()
print("✅ Dataset successfully fetched to local path:", local_path)
