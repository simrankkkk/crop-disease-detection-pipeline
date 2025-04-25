# pipeline_from_tasks.py
from clearml import Task, PipelineDecorator

@PipelineDecorator.pipeline(
    name="Plant Disease Detection Pipeline",
    project="PlantPipeline",
    version="1.0"
)
def run_pipeline():
    # Step 1: Load dataset from ClearML
    step1 = PipelineDecorator.add_function_step(
        name="upload_dataset",
        function="step1.py",
        function_kwargs={},
        task_type=Task.TaskTypes.data_processing,
        task_name="Step 1 - Load Dataset"
    )

    # Step 2: Preprocess images
    step2 = PipelineDecorator.add_function_step(
        name="preprocess_data",
        function="step2.py",
        function_kwargs={"input_path": step1},  # Will pass dataset path from Step 1
        task_type=Task.TaskTypes.data_processing,
        task_name="Step 2 - Preprocess Data"
    )

    # Step 3: Train the hybrid model
    step3 = PipelineDecorator.add_function_step(
        name="train_model",
        function="step3.py",
        function_kwargs={"dataset_path": step2},
        task_type=Task.TaskTypes.training,
        task_name="Step 3 - Train Hybrid Model"
    )

if __name__ == "__main__":
    PipelineDecorator.run_locally()
    run_pipeline()
