from clearml.automation.controller import PipelineDecorator, PipelineController

# ─────────────────────────────
# STEP 1: Fetch dataset
# ─────────────────────────────
@PipelineDecorator.component(name="step1_fetch_dataset")
def step1():
    from clearml import Dataset
    dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
    path = dataset.get_local_copy()
    print("✅ Dataset fetched at:", path)
    return path

# ─────────────────────────────
# STEP 2: Preprocessing
# ─────────────────────────────
@PipelineDecorator.component(name="step2_preprocess_data")
def step2(dataset_path: str):
    import step2  # runs step2.py script logic
    return step2.step2()  # must return path to preprocessed folder

# ─────────────────────────────
# STEP 3: Train hybrid CNN model
# ─────────────────────────────
@PipelineDecorator.component(name="step3_train_model")
def step3(preprocessed_path: str):
    import step3  # runs step3.py script logic
    return step3.step3()  # must return path to trained model

# ─────────────────────────────
# Main pipeline controller
# ─────────────────────────────
if __name__ == "__main__":
    pipe = PipelineDecorator.run_locally(
        name="PlantPipeline",
        project="PlantPipeline",
        version="1.0",
        queue="default"  # or your ClearML agent queue name
    )

    pipe.set_default_execution_queue("default")

    # Chain steps
    step1_output = step1()
    step2_output = step2(dataset_path=step1_output)
    step3_output = step3(preprocessed_path=step2_output)

    print("✅ Pipeline completed. Model saved at:", step3_output)
