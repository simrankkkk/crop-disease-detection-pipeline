from clearml import PipelineDecorator

@PipelineDecorator.pipeline(
    name="CropPipeline",                  # ✅ Clean final pipeline name
    project="PlantPipeline",               # ✅ Your project name
    version="1.0",                         # ✅ Version tag
    default_execution_queue="default",     # ✅ Default ClearML queue
)
def crop_pipeline():
    # Stage 1: Upload Augmented Dataset (already done, reuse task)
    upload_task_id = "db38d2b346e34c4b921ea5e25106b3a9"  # ✅ Your Step 1 task ID
    upload_task = PipelineDecorator.add_existing_task(
        task_id=upload_task_id,
        name="stage_upload"
    )

    # Stage 2: Preprocess the dataset
    preprocess_task_id = "32475789c3c24b8c9d4966ceefef130a"  # ✅ Your Step 2 task ID
    preprocess_task = PipelineDecorator.add_existing_task(
        task_id=preprocess_task_id,
        name="stage_preprocess",
        parents=["stage_upload"]
    )

    # Stage 3: Train Hybrid Model
    train_task_id = "ee371ffb8f3441e1a527ae0fe14d8860"  # ✅ Your Step 3 task ID
    train_task = PipelineDecorator.add_existing_task(
        task_id=train_task_id,
        name="stage_train",
        parents=["stage_preprocess"]
    )

if __name__ == "__main__":
    crop_pipeline()
