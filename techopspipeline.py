from clearml.automation.controller import PipelineController
from clearml import Task

# ✅ Optional: Register project root explicitly
Task.init(project_name="T3chOpsClearMLProject", task_name="register_project_root").close()

pipe = PipelineController(
    name="T3chOpsClearMLProject",
    project="T3chOpsClearMLProject",
    version="1.0"
)

pipe.set_default_execution_queue("pipeline")

# STEP 1: Upload dataset
pipe.add_step(
    name="step_upload",
    base_task_project="T3chOpsClearMLProject",
    base_task_name="step_upload",
    execution_queue="default"
)

# STEP 2: Preprocess
pipe.add_step(
    name="step_preprocess",
    parents=["step_upload"],
    base_task_project="T3chOpsClearMLProject",
    base_task_name="step_preprocess",
    execution_queue="default",
    parameter_override={"Args/dataset_id": "${step_upload.id}"}
)

# STEP 3: Train baseline
pipe.add_step(
    name="step_train_baseline",
    parents=["step_preprocess"],
    base_task_project="T3chOpsClearMLProject",
    base_task_name="step_train_baseline",
    execution_queue="default",
    parameter_override={"Args/dataset_id": "${step_preprocess.id}"}
)

# STEP 4: Manual HPO
pipe.add_step(
    name="step_hpo_manual_grid",
    parents=["step_preprocess"],
    base_task_project="T3chOpsClearMLProject",
    base_task_name="step_hpo_manual_grid",
    execution_queue="default",
    parameter_override={
        "Args/dataset_id": "${step_preprocess.id}",
        "Args/baseline_task_id": "${step_train_baseline.id}"
    }
)

# STEP 5: Final model training
pipe.add_step(
    name="step_train_final",
    parents=["step_hpo_manual_grid"],
    base_task_project="T3chOpsClearMLProject",
    base_task_name="step_train_final",
    execution_queue="default",
    parameter_override={
        "Args/dataset_id": "${step_preprocess.id}",
        "Args/hpo_task_id": "${step_hpo_manual_grid.id}"
    }
)

pipe.start(queue="pipeline")
