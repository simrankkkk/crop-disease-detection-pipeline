from clearml import PipelineDecorator

@PipelineDecorator.component(name="stage_upload", return_values=["uploaded_dataset_id"])
def stage_upload():
    from clearml import Dataset
    dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
    return dataset.id

@PipelineDecorator.component(name="stage_preprocess", return_values=["preprocessed_dataset_id"])
def stage_preprocess(uploaded_dataset_id):
    # Dummy preprocessing for now
    return uploaded_dataset_id

@PipelineDecorator.component(name="stage_train")
def stage_train(preprocessed_dataset_id):
    print(f"Training with preprocessed dataset: {preprocessed_dataset_id}")

@PipelineDecorator.pipeline(
    name="CropPipeline",
    project="PlantPipeline",
    version="1.0"
)
def crop_pipeline():
    uploaded_dataset_id = stage_upload()
    preprocessed_dataset_id = stage_preprocess(uploaded_dataset_id=uploaded_dataset_id)
    stage_train(preprocessed_dataset_id=preprocessed_dataset_id)

if __name__ == "__main__":
    pipe = crop_pipeline()
    pipe.set_default_execution_queue("default")   # ✅ critical!
    pipe.start()
    pipe.wait()
    pipe.stop()
