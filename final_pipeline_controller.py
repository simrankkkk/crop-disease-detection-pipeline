# final_pipeline_controller.py

from clearml import Task
from clearml.automation.controller import PipelineController

# ✅ Register this pipeline for visibility
Task.init(
    project_name="FinalProject",
    task_name="__pipeline_controller_entrypoint__",
    task_type=Task.TaskTypes.testing
).close()

# ✅ Define the controller pipeline
pipe = PipelineController(
    name="FinalPipeline",
    project="FinalProject",
    version="1.0"
)

# STEP 1: Upload dataset (dummy step for reference only)
pipe.add_step(
    name="final_step_upload",
    base_task_project="FinalProject",
    base_task_name="final_step_upload",
    parameter_override={},
    execution_queue="default"
)

# STEP 2: Preprocess (pass fixed dataset ID directly)
pipe.add_step(
    name="final_step_preprocess",
    base_task_project="FinalProject",
    base_task_name="final_step_preprocess",
    parents=["final_step_upload"],
    parameter_override={
        "Args/dataset_id": "105163c10d0a4bbaa06055807084ec71"
    },
    execution_queue="default"
)

# STEP 3: Baseline Training
pipe.add_step(
    name="final_step_baseline_train",
    base_task_project="FinalProject",
    base_task_name="final_step_baseline_train",
    parents=["final_step_preprocess"],
    parameter_override={
        "Args/dataset_id": "${final_step_preprocess.id}"
    },
    execution_queue="default"
)

# STEP 4: HPO
pipe.add_step(
    name="final_step_hpo",
    base_task_project="FinalProject",
    base_task_name="final_step_hpo",
    parents=["final_step_baseline_train"],
    parameter_override={
        "Args/dataset_id": "${final_step_preprocess.id}",
        "Args/baseline_task_id": "${final_step_baseline_train.id}"
    },
    execution_queue="default"
)

# STEP 5: Final Train
pipe.add_step(
    name="final_step_final_train",
    base_task_project="FinalProject",
    base_task_name="final_step_final_train",
    parents=["final_step_hpo"],
    parameter_override={
        "Args/dataset_id": "${final_step_preprocess.id}",
        "Args/hpo_task_id": "${final_step_hpo.id}"
    },
    execution_queue="default"
)

# ✅ Start the pipeline
pipe.start()
