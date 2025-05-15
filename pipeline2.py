# pipeline_controller.py

from clearml import Task
from clearml.automation.controller import PipelineController

if __name__ == "__main__":
    # 1. Instantiate the pipeline controller
    controller = PipelineController(
        project="T3chOpsClearMLProject",      # the ClearML project
        name="T3chOpsPipeline",               # a unique name for this pipeline
        version="1.0",                        # pipeline version
        default_queue="default"               # where steps will run :contentReference[oaicite:0]{index=0}
    )

    # 2. Step 1: Upload
    controller.add_step(
        name="step_upload",
        base_task_project="T3chOpsClearMLProject",
        base_task_name="step_upload",
        task_type=Task.TaskTypes.data_processing
    )

    # 3. Step 2: Preprocess (depends on upload)
    controller.add_step(
        name="step_preprocess",
        parents=["step_upload"],
        base_task_project="T3chOpsClearMLProject",
        base_task_name="step_preprocess",
        task_type=Task.TaskTypes.data_processing,
        parameter_override={
            "Args/dataset_id": "${step_upload.id}"
        }
    )

    # 4. Step 3: Baseline training (depends on preprocess)
    controller.add_step(
        name="step_train_baseline",
        parents=["step_preprocess"],
        base_task_project="T3chOpsClearMLProject",
        base_task_name="step_train_baseline",
        task_type=Task.TaskTypes.training,
        parameter_override={
            "Args/dataset_id": "${step_preprocess.id}"
        }
    )

    # 5. Step 4: Manual HPO (depends on baseline train)
    controller.add_step(
        name="step_hpo_manual_grid",
        parents=["step_train_baseline"],
        base_task_project="T3chOpsClearMLProject",
        base_task_name="step_hpo_manual_grid",
        task_type=Task.TaskTypes.controller,
        parameter_override={
            "Args/baseline_task_id": "${step_train_baseline.id}",
            "Args/dataset_id":         "${step_preprocess.id}"
        }
    )

    # 6. Step 5: Final training (depends on HPO)
    controller.add_step(
        name="step_train_final",
        parents=["step_hpo_manual_grid"],
        base_task_project="T3chOpsClearMLProject",
        base_task_name="step_train_final",
        task_type=Task.TaskTypes.training,
        parameter_override={
            "Args/hpo_task_id": "${step_hpo_manual_grid.id}",
            "Args/dataset_id":  "${step_preprocess.id}"
        }
    )

    # 7. Launch the pipeline
    controller.start(queue="default")  # Or omit queue if you set default_queue above :contentReference[oaicite:1]{index=1}
