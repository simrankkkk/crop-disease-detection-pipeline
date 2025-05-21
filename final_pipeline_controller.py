# final_pipeline_controller.py

from clearml import Task
from clearml.automation.controller import PipelineController

# Register a “head” task so ClearML knows this is a pipeline
Task.init(
    project_name="FinalProject",
    task_name="__pipeline_direct_preprocess__",
    task_type=Task.TaskTypes.testing
).close()

# Build the pipeline controller
pipe = PipelineController(
    name="FinalPipeline",
    project="FinalProject",
    version="1.0"
)

# STEP 2: Preprocess
pipe.add_step(
    name="final_step_preprocess",
    base_task_project="FinalProject",
    base_task_name="final_step_preprocess",
    parameter_override={
        "Args/dataset_id": "81e8c009a1f04dc583f7ec872ed76e5c"
    }
)

# STEP 3: Baseline Train
pipe.add_step(
    name="final_step_baseline_train",
    base_task_project="FinalProject",
    base_task_name="final_step_baseline_train",
    parents=["final_step_preprocess"],
    parameter_override={
        "Args/dataset_id": "81e8c009a1f04dc583f7ec872ed76e5c"
    }
)

# STEP 4: HPO
pipe.add_step(
    name="final_step_hpo",
    base_task_project="FinalProject",
    base_task_name="final_step_hpo",
    parents=["final_step_baseline_train"],
    parameter_override={
        "Args/dataset_id": "81e8c009a1f04dc583f7ec872ed76e5c",
        "Args/baseline_task_id": "${final_step_baseline_train.id}"
    }
)

# STEP 5: Final Training
pipe.add_step(
    name="final_step_final_train",
    base_task_project="FinalProject",
    base_task_name="final_step_final_train",
    parents=["final_step_hpo"],
    parameter_override={
        "Args/dataset_id": "81e8c009a1f04dc583f7ec872ed76e5c",
        "Args/hpo_task_id": "${final_step_hpo.id}"
    }
)

# Start the pipeline — this will execute each step right here, in-process
pipe.start()
