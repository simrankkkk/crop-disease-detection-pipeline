from clearml import PipelineController

# 🚀 Initialize pipeline
pipe = PipelineController(
    name="Plant Disease Detection Pipeline",
    project="PlantPipeline",
    version="1.0",
    add_pipeline_tags=True
)

# 🔗 Step 1: Upload Dataset
pipe.add_step(
    name="upload_dataset",
    base_task_id="db38d2b346e34c4b921ea5e25106b3a9",  # Step 1 Task ID
    execution_queue="default"
)

# 🔗 Step 2: Preprocess Dataset
pipe.add_step(
    name="preprocess_dataset",
    base_task_id="32475789c3c24b8c9d4966ceefef130a",  # Step 2 Task ID
    parents=["upload_dataset"],   # Step 2 depends on Step 1
    execution_queue="default"
)

# 🔗 Step 3: Train Model
pipe.add_step(
    name="train_model",
    base_task_id="ee371ffb8f3441e1a527ae0fe14d8860",  # Step 3 Task ID
    parents=["preprocess_dataset"],  # Step 3 depends on Step 2
    execution_queue="default"
)

# ✅ Start the pipeline
pipe.start()
pipe.wait()
pipe.stop()
