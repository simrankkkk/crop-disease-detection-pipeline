# finalstep1.py
from clearml import Dataset, Task

# ✅ Initialize ClearML Task
task = Task.init(
    project_name="FinalProject",
    task_name="final_step_upload",
    task_type=Task.TaskTypes.data_processing
)

# ✅ Reference existing dataset
dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")

# ✅ Download dataset locally
local_path = dataset.get_local_copy()
print("✅ Dataset successfully fetched to local path:", local_path)

# ✅ Pass dataset_id to next steps in pipeline
task.set_parameter("dataset_id", dataset.id)

task.close()
