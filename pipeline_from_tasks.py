# pipeline_from_tasks.py
from clearml.automation.controller import PipelineController

if __name__ == '__main__':
    pipe = PipelineController(
        project='PlantPipeline',
        name='Crop Disease Detection Pipeline',
        version='1.0'
    )

    # Step 1: fetch existing dataset
    pipe.add_step(
        name='fetch_dataset',
        function='step1_fetch_dataset.py',  # or function=step1_fetch_dataset if imported
        clone=False
    )

    # Step 2: preprocess
    pipe.add_step(
        name='preprocess_data',
        function='step2_data_preprocessing.py',
        function_kwargs={'dataset_id': '${fetch_dataset.output}'},
        parent='fetch_dataset',
        clone=False
    )

    # Step 3: train
    pipe.add_step(
        name='train_model',
        function='step3_train_model.py',
        function_kwargs={'processed_folder': '${preprocess_data.output}'},
        parent='preprocess_data',
        clone=False
    )

    # launch on your default queue
    pipe.start(queue='default', sleep_interval=5)
    print("✅ Pipeline started.")
