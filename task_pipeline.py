from clearml import PipelineController

pipe = PipelineController(
    name="VisiblePipelineRun",
    project="VisiblePipeline",
    version="2.0",
    add_pipeline_tags=False
)
# Step 1: Upload
pipe.add_step(
    name="step_upload",
    base_task_project="VisiblePipeline",
    base_task_name="step_upload",
    execution_queue="default"
)


# Step 2: Preprocess
pipe.add_step(
    name="step_preprocess",
    base_task_project="VisiblePipeline",
    base_task_name="step_preprocess",
    execution_queue="default"
)

# Step 3: Baseline training
pipe.add_step(
    name="step_train_baseline",
    base_task_project="VisiblePipeline",
    base_task_name="step_train_baseline",
    parents=["step_preprocess"],
    execution_queue="default"
)

# Step 4: Manual HPO
pipe.add_step(
    name="step_hpo_manual_grid",
    base_task_project="VisiblePipeline",
    base_task_name="step_hpo_manual_grid",
    parents=["step_train_baseline"],
    execution_queue="default",
    parameter_override={
        "Args/base_task_id": "${step_train_baseline.id}"
    }
)

# Step 5: Final training with best params
pipe.add_step(
    name="step_train_final",
    base_task_project="VisiblePipeline",
    base_task_name="step_train_final",
    parents=["step_hpo_manual_grid"],
    execution_queue="default",
    parameter_override={
        "Args/hpo_task_id": "${step_hpo_manual_grid.id}"
    }
)

pipe.start()
