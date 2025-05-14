from clearml import Task
from clearml.automation import PipelineDecorator

@PipelineDecorator.pipeline(
    name="T3chOpsClearMLProject",
    project="T3chOpsClearMLProject",
    version="1.0"
)
def run_pipeline():
    # STEP 1: Upload
    step_upload = Task.add_task(
        project_name="T3chOpsClearMLProject",
        task_name="step_upload",
        task_type=Task.TaskTypes.data_processing,
        execution_queue="default"
    )
    dataset_id = step_upload.id

    # STEP 2: Preprocess
    step_preprocess = Task.add_task(
        project_name="T3chOpsClearMLProject",
        task_name="step_preprocess",
        task_type=Task.TaskTypes.data_processing,
        execution_queue="default",
        parameter_override={"Args/dataset_id": dataset_id}
    )
    split_dataset_id = step_preprocess.id

    # STEP 3: Train Baseline
    step_baseline = Task.add_task(
        project_name="T3chOpsClearMLProject",
        task_name="step_train_baseline",
        task_type=Task.TaskTypes.training,
        execution_queue="default",
        parameter_override={"Args/dataset_id": split_dataset_id}
    )
    baseline_task_id = step_baseline.id

    # STEP 4: HPO
    step_hpo = Task.add_task(
        project_name="T3chOpsClearMLProject",
        task_name="step_hpo_manual_grid",
        task_type=Task.TaskTypes.controller,
        execution_queue="default",
        parameter_override={
            "Args/dataset_id": split_dataset_id,
            "Args/baseline_task_id": baseline_task_id
        }
    )
    hpo_task_id = step_hpo.id

    # STEP 5: Final Train
    Task.add_task(
        project_name="T3chOpsClearMLProject",
        task_name="step_train_final",
        task_type=Task.TaskTypes.training,
        execution_queue="default",
        parameter_override={
            "Args/dataset_id": split_dataset_id,
            "Args/hpo_task_id": hpo_task_id
        }
    )

if __name__ == '__main__':
    run_pipeline()
