from clearml import PipelineController

# ✅ Create the pipeline controller
pipe = PipelineController(
    name="Plant Disease Detection Full Pipeline",
    project="plantdataset",
    version="1.0"
)

# ✅ Step 1 - Dataset Loading
pipe.add_step(
    name="step1_upload",
    base_task_project="plantdataset",
    base_task_name="Clone Of Upload New Augmented Plant Disease Dataset"
)

# ✅ Step 2 - Preprocessing (Torch-free)
pipe.add_step(
    name="step2_preprocess",
    base_task_project="plantdataset",
    base_task_name="step2 without torch",
    parents=["step1_upload"]
)

# ✅ Step 3 - Hybrid Training (TF, No Torch)
pipe.add_step(
    name="step3_train",
    base_task_project="plantdataset",
    base_task_name="Clone Of Clone Of step3notorch",
    parents=["step2_preprocess"]
)

# ✅ Launch the pipeline (use your agent queue)
pipe.start(queue="mansimran-gpu")
