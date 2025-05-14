from clearml.automation.controller import PipelineController

# ✅ Initialize the pipeline
pipe = PipelineController(
    name="T3chOpsClearMLProject",
    project="T3chOpsClearMLProject",
    version="1.0"
)

# STEP 1: Upload raw dataset
pipe.add_step(
    name="step_upload",
    base_task_project="T3chOpsClearMLProject",
    base_task_name="step_upload",
    execution_queue="default"
)

# STEP 2: Preprocess & split dataset
pipe.add_step(
    name="step_preprocess",
    parents=["step_upload"],
    base_task_project="T3chOpsClearMLProject",
    base_task_name="step_preprocess",
    execution_queue="default",
    parameter_override={
        "Args/dataset_id": "${step_upload.id}"
    }
)

# STEP 3: Train baseline hybrid model
pipe.add_step(
    name="step_train_baseline",
    parents=["step_preprocess"],
    base_task_project="T3chOpsClearMLProject",
    base_task_name="step_train_baseline",
    execution_queue="default",
    parameter_override={
        "Args/dataset_id": "${step_preprocess.id}"
    }
)

# STEP 4: Manual grid HPO
pipe.add_step(
    name="step_hpo_manual_grid",
    parents=["step_preprocess"],
    base_task_project="T3chOpsClearMLProject",
    base_task_name="step_hpo_manual_grid",
    execution_queue="default",
    parameter_override={
        "Args/dataset_id": "${step_preprocess.id}"
    }
)

# STEP 5: Final training using best HPO params
pipe.add_step(
    name="step_train_final",
    parents=["step_hpo_manual_grid"],
    base_task_project="T3chOpsClearMLProject",
    base_task_name="step_train_final",
    execution_queue="default",
    parameter_override={
        "Args/dataset_id": "${step_preprocess.id}",
        "Args/hpo_task_id": "${step_hpo_manual_grid.id}",
        "Args/best_result": "${step_hpo_manual_grid.artifacts.best_result.url}"
    }
)

# ✅ Run the pipeline
pipe.start(queue="pipeline")
