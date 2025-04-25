from clearml import PipelineController

# 🚀 Initialize pipeline
pipe = PipelineController(
    name="Plant Disease Detection Full Pipeline",
    project="PlantPipeline",
    version="1.0",
    add_pipeline_tags=True
)

# 🔗 Stage 1: Upload Dataset
pipe.add_step(
    name="stage_upload",
    base_task_id="db38d2b346e34c4b921ea5e25106b3a9",  # Step 1 task id
    execution_queue="default"
)

# 🔗 Stage 2: Preprocessing
pipe.add_step(
    name="stage_preprocess",
    base_task_id="32475789c3c24b8c9d4966ceefef130a",  # Step 2 task id
    parents=["stage_upload"],                         # Depends on Stage 1
    execution_queue="default"
)

# 🔗 Stage 3: Train Model
pipe.add_step(
    name="stage_train",
    base_task_id="ee371ffb8f3441e1a527ae0fe14d8860",  # Step 3 task id
    parents=["stage_preprocess"],                     # Depends on Stage 2
    execution_queue="default"
)

# ✅ Start pipeline
pipe.start()
pipe.wait()
pipe.stop()
