from clearml.automation.controller import PipelineController

# Initialize the pipeline
pipe = PipelineController(
    name="VisiblePipelineRun",
    project="VisiblePipeline",
    version="1.0"
)

# STEP 1: Upload Dataset
pipe.add_step(
    name="step_upload",
    base_task_project="VisiblePipeline",
    base_task_name="step_upload",
    execution_queue="default"
)

# STEP 2: Preprocess & Split Dataset
pipe.add_step(
    name="step_preprocess",
    parents=["step_upload"],
    base_task_project="VisiblePipeline",
    base_task_name="step_preprocess",
    execution_queue="default",
    parameter_override={
        "Args/dataset_id": "${step_upload.id}"
    }
)

# STEP 3: Baseline Training
pipe.add_step(
    name="step_train_baseline",
    parents=["step_preprocess"],
    base_task_project="VisiblePipeline",
    base_task_name="step_train_baseline",
    execution_queue="default",
    parameter_override={
        "Args/dataset_id": "${step_preprocess.id}"
    }
)

# STEP 4: Manual Grid HPO
pipe.add_step(
    name="step_hpo_manual_grid",
    parents=["step_preprocess"],
    base_task_project="VisiblePipeline",
    base_task_name="step_hpo_manual_grid",
    execution_queue="default",
    parameter_override={
        "Args/dataset_id": "${step_preprocess.id}"
    }
)

# STEP 5: Final Train using Best Params
pipe.add_step(
    name="step_train_final",
    parents=["step_hpo_manual_grid"],
    base_task_project="VisiblePipeline",
    base_task_name="step_train_final",
    execution_queue="default",
    parameter_override={
        "Args/dataset_id": "${step_preprocess.id}",
        "Args/best_result_json": "${step_hpo_manual_grid.artifacts.best_result_json.url}"
    }
)

# 🚀 Start the pipeline
pipe.start(queue="pipeline")
