from clearml.automation.controller import PipelineController

pipe = PipelineController(
    name="VisiblePipelineRun",
    project="VisiblePipeline",
    version="1.0"
)

pipe.add_step(
    name="step_upload",
    base_task_project="VisiblePipeline",
    base_task_name="step_to_upload"
)

pipe.add_step(
    name="step_preprocess",
    parents=["step_to_upload"],
    base_task_project="VisiblePipeline",
    base_task_name="step_preprocess",
    parameter_override={
        "args.dataset_id": "${step_upload.OUTPUT_DATASET_ID}"
    }
)

pipe.add_step(
    name="step_train",
    parents=["step_to_preprocess"],
    base_task_project="VisiblePipeline",
    base_task_name="step_to_train",
    parameter_override={
        "args.dataset_id": "${step_preprocess.OUTPUT_DATASET_ID}"
    }
)

pipe.start(queue="default")
