from clearml import Task
from clearml.automation import PipelineDecorator

# ✅ Automatically register project (solves missing project error)
Task.init(project_name="T3chOpsClearMLProject", task_name="pipeline_entrypoint").close()

@PipelineDecorator.pipeline(
    name="T3chOpsClearMLProject",
    project="T3chOpsClearMLProject",
    version="1.0"
)
def run_pipeline():
    # STEP 1: Upload
    step_upload_task = PipelineDecorator.task(
        name="step_upload",
        project="T3chOpsClearMLProject",
        task_name="step_upload",
        task_type="data_processing",
        execution_queue="default"
    )
    dataset_id = step_upload_task.id

    # STEP 2: Preprocess
    step_preprocess_task = PipelineDecorator.task(
        name="step_preprocess",
        project="T3chOpsClearMLProject",
        task_name="step_preprocess",
        task_type="data_processing",
        execution_queue="default",
        parameter_override={"Args/dataset_id": dataset_id}
    )
    split_dataset_id = step_preprocess_task.id

    # STEP 3: Train Baseline
    step_baseline_task = PipelineDecorator.task(
        name="step_train_baseline",
        project="T3chOpsClearMLProject",
        task_name="step_train_baseline",
        task_type="training",
        execution_queue="default",
        parameter_override={"Args/dataset_id": split_dataset_id}
    )
    baseline_task_id = step_baseline_task.id

    # STEP 4: HPO
    step_hpo_task = PipelineDecorator.task(
        name="step_hpo_manual_grid",
        project="T3chOpsClearMLProject",
        task_name="step_hpo_manual_grid",
        task_type="controller",
        execution_queue="default",
        parameter_override={
            "Args/dataset_id": split_dataset_id,
            "Args/baseline_task_id": baseline_task_id
        }
    )
    hpo_task_id = step_hpo_task.id

    # STEP 5: Final training
    step_final_task = PipelineDecorator.task(
        name="step_train_final",
        project="T3chOpsClearMLProject",
        task_name="step_train_final",
        task_type="training",
        execution_queue="default",
        parameter_override={
            "Args/dataset_id": split_dataset_id,
            "Args/hpo_task_id": hpo_task_id
        }
    )

if __name__ == "__main__":
    run_pipeline()
