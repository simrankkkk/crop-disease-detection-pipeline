# pipeline2.py

from clearml import Task
from clearml.automation.controller import PipelineDecorator

# ----------------------------------------------------------------------------
# 1) Force-register your project so it’s resolvable inside the decorator
# ----------------------------------------------------------------------------
Task.init(
    project_name="T3chOpsClearMLProject",
    task_name="__pipeline_entrypoint__",
    task_type=Task.TaskTypes.testing
).close()

# ----------------------------------------------------------------------------
# 2) Define your pipeline with the decorator
# ----------------------------------------------------------------------------
@PipelineDecorator.pipeline(
    name="T3chOpsClearMLProject",
    project="T3chOpsClearMLProject",
    version="1.0"
)
def run_pipeline():
    # STEP 1: Upload
    t1 = Task.add_task(
        project_name="T3chOpsClearMLProject",
        task_name="step_upload",
        task_type=Task.TaskTypes.data_processing,
        execution_queue="default",
    )
    ds_id = t1.id

    # STEP 2: Preprocess
    t2 = Task.add_task(
        project_name="T3chOpsClearMLProject",
        task_name="step_preprocess",
        task_type=Task.TaskTypes.data_processing,
        execution_queue="default",
        parameter_override={"Args/dataset_id": ds_id},
    )
    split_id = t2.id

    # STEP 3: Baseline train
    t3 = Task.add_task(
        project_name="T3chOpsClearMLProject",
        task_name="step_train_baseline",
        task_type=Task.TaskTypes.training,
        execution_queue="default",
        parameter_override={"Args/dataset_id": split_id},
    )
    base_id = t3.id

    # STEP 4: HPO
    t4 = Task.add_task(
        project_name="T3chOpsClearMLProject",
        task_name="step_hpo_manual_grid",
        task_type=Task.TaskTypes.controller,
        execution_queue="default",
        parameter_override={
            "Args/dataset_id": split_id,
            "Args/baseline_task_id": base_id,
        },
    )
    hpo_id = t4.id

    # STEP 5: Final train
    Task.add_task(
        project_name="T3chOpsClearMLProject",
        task_name="step_train_final",
        task_type=Task.TaskTypes.training,
        execution_queue="default",
        parameter_override={
            "Args/dataset_id": split_id,
            "Args/hpo_task_id": hpo_id,
        },
    )

# ----------------------------------------------------------------------------
# 3) Run the pipeline by calling the decorated function
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
