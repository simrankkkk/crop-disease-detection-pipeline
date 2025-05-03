from clearml.automation.controller import PipelineController

pipe = PipelineController(
    name="VisiblePipelineRun",
    project="VisiblePipeline",
    version="1.0"
)

pipe.add_step(
    name="step_upload",
    base_task_project="VisiblePipeline",
    base_task_name="step_upload"
)

pipe.add_step(
    name="step_preprocess",
    parents=["step_upload"],
    base_task_project="VisiblePipeline",
    base_task_name="step_preprocess",
    parameter_override={
        "Args/dataset_id": "${step_upload.id}"
    }
)

pipe.add_step(
    name="step_train",
    parents=["step_preprocess"],
    base_task_project="VisiblePipeline",
    base_task_name="step_train",
    parameter_override={
        "Args/dataset_id": "${step_preprocess.id}"
    }
)

pipe.start(queue="default")
