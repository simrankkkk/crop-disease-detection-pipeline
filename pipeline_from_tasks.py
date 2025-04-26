from clearml import Task
from clearml.automation.controller import PipelineDecorator

@PipelineDecorator.pipeline(
    name="Crop Pipeline",
    project="PlantPipeline",
    version="1.0"
)
def pipeline():
    # Stage 1: Upload
    upload_task = PipelineDecorator.add_existing_task(
        task_id="db38d2b346e34c4b921ea5e25106b3a9",  # ✅ your actual Step 1 task ID
        name="stage_upload"
    )

    # Stage 2: Preprocess
    preprocess_task = PipelineDecorator.add_existing_task(
        task_id="32475789c3c24b8c9d4966ceefef130a",  # ✅ your actual Step 2 task ID
        name="stage_preprocess"
    )

    # Stage 3: Train
    train_task = PipelineDecorator.add_existing_task(
        task_id="ee371ffb8f3441e1a527ae0fe14d8860",  # ✅ your actual Step 3 task ID
        name="stage_train"
    )

if __name__ == "__main__":
    PipelineDecorator.run_locally()
    pipeline()
