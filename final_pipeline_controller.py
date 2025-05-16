# final_pipeline_controller.py

from clearml import Task
from clearml.automation.controller import PipelineController

# ─── Register the controller entrypoint ──────────────────────────────────────
Task.init(
    project_name="FinalProject",
    task_name="__pipeline_controller_entrypoint__",
    task_type=Task.TaskTypes.testing
).close()

# ─── Create (or recreate) the pipeline definition ───────────────────────────
pipe = PipelineController(
    name="FinalPipeline",
    project="FinalProject",
    version="1.1",                 # ← bumped from 1.0 to force new registration
    default_execution_queue="default",
    recreate_pipeline=True        # ← drop old definition and use this one
)

# ─── STEP 1: Upload Dataset ──────────────────────────────────────────────────
pipe.add_step(
    name="final_step_upload",
    base_task_project="FinalProject",
    base_task_name="final_step_upload",
    execution_queue="default",
    clone=True                    # ← always launch a fresh upload task
)

# ─── STEP 2: Preprocess ──────────────────────────────────────────────────────
pipe.add_step(
    name="final_step_preprocess",
    base_task_project="FinalProject",
    base_task_name="final_step_preprocess",
    parents=["final_step_upload"],
    parameter_override={
        "Args/dataset_id": "${final_step_upload.parameters.dataset_id}"
    },
    execution_queue="default"
)

# ─── STEP 3: Baseline Training ───────────────────────────────────────────────
pipe.add_step(
    name="final_step_baseline_train",
    base_task_project="FinalProject",
    base_task_name="final_step_baseline_train",
    parents=["final_step_preprocess"],
    parameter_override={
        "Args/dataset_id": "${final_step_preprocess.parameters.dataset_id}"
    },
    execution_queue="default"
)

# ─── STEP 4: Hyperparameter Optimization ─────────────────────────────────────
pipe.add_step(
    name="final_step_hpo",
    base_task_project="FinalProject",
    base_task_name="final_step_hpo",
    parents=["final_step_baseline_train"],
    parameter_override={
        "Args/dataset_id":        "${final_step_preprocess.parameters.dataset_id}",
        "Args/baseline_task_id":  "${final_step_baseline_train.id}"
    },
    execution_queue="default"
)

# ─── STEP 5: Final Training ─────────────────────────────────────────────────
pipe.add_step(
    name="final_step_final_train",
    base_task_project="FinalProject",
    base_task_name="final_step_final_train",
    parents=["final_step_hpo"],
    parameter_override={
        "Args/dataset_id":   "${final_step_preprocess.parameters.dataset_id}",
        "Args/hpo_task_id":  "${final_step_hpo.id}"
    },
    execution_queue="default"
)

# ─── Launch the pipeline ─────────────────────────────────────────────────────
pipe.start()
