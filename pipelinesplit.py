# pipeline_from_tasks.py — Task-based ClearML pipeline
from clearml.automation import PipelineController

pipe = PipelineController(
    name="Crop Disease Detection - Task Pipeline",
    project="PlantPipeline",
    version="1.0"
)

# Step 1: Dataset download (already uploaded raw dataset used)
pipe.add_step(
    name="stage_upload",
    base_task_project="PlantPipeline",
    base_task_name="Step 1 - Upload Raw Dataset"
)

# Step 2: Preprocess with split
pipe.add_step(
    name="stage_preprocess",
    base_task_project="PlantPipeline",
    base_task_name="Step 2 - Preprocess with Split",
    parents=["stage_upload"],
    parameter_override={
        "General/dataset_task_id": "${stage_upload.id}"
    }
)

# Step 3: Train hybrid model
pipe.add_step(
    name="stage_train",
    base_task_project="PlantPipeline",
    base_task_name="Step 3 - Train Hybrid Model",
    parents=["stage_preprocess"],
    parameter_override={
        "General/dataset_task_id": "${stage_preprocess.id}"
    }
)

if __name__ == '__main__':
    pipe.start(queue="pipeline")
