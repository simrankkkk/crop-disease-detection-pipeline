from clearml import PipelineDecorator

@PipelineDecorator.pipeline(
    name="CropPipeline",
    project="PlantPipeline",
    version="1.0"
)
def crop_pipeline():
    upload_task = PipelineDecorator.add_existing_task(
        task_id="db38d2b346e34c4b921ea5e25106b3a9",
        name="stage_upload"
    )

    preprocess_task = PipelineDecorator.add_existing_task(
        task_id="32475789c3c24b8c9d4966ceefef130a",
        name="stage_preprocess",
        parents=["stage_upload"]
    )

    train_task = PipelineDecorator.add_existing_task(
        task_id="ee371ffb8f3441e1a527ae0fe14d8860",
        name="stage_train",
        parents=["stage_preprocess"]
    )

if __name__ == "__main__":
    pipe = crop_pipeline()
    pipe.set_default_execution_queue("default")  # ✅ set queue AFTER pipeline object is created
    pipe.start()
    pipe.wait()
    pipe.stop()
