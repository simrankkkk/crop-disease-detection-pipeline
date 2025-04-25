#!/usr/bin/env python3
from clearml.automation.controller import PipelineController
from clearml import Dataset, Task
import os
import subprocess

# --- STEP FUNCTIONS --------------------------------------------------------

def fetch_dataset(dataset_id: str, output_dir: str = "data"):
    """
    Downloads the dataset version you already uploaded to ClearML.
    """
    ds = Dataset.get(dataset_id=dataset_id)
    ds_path = ds.get_local_copy()  # returns local folder
    print(f"Dataset downloaded to {ds_path}")
    # you can also copy or reorganize if your scripts expect a different layout:
    if os.path.exists(output_dir):
        print(f"output_dir {output_dir} already exists, skipping copy.")
    else:
        os.rename(ds_path, output_dir)
        print(f"Moved dataset to ./{output_dir}")

def preprocess(output_dir: str = "data", processed_dir: str = "processed"):
    """
    Launches your preprocessing script (step2.py) as a subprocess.
    Expects that script to read from `output_dir` and write to `processed_dir`.
    """
    cmd = [
        "python", "step2.py",
        "--input", output_dir,
        "--output", processed_dir
    ]
    print("Running preprocessing:", " ".join(cmd))
    subprocess.check_call(cmd)

def train(processed_dir: str = "processed", model_dir: str = "models"):
    """
    Launches your training script (step3.py) as a subprocess.
    Expects that script to read from `processed_dir` and write out into `model_dir`.
    """
    cmd = [
        "python", "step3.py",
        "--data", processed_dir,
        "--save", model_dir
    ]
    print("Running training:", " ".join(cmd))
    subprocess.check_call(cmd)


# --- PIPELINE DEFINITION --------------------------------------------------

def main():
    # Adjust these to match your ClearML project/queue names
    PIPELINE_PROJECT = "PlantPipeline"
    PIPELINE_NAME    = "CropDiseasePipeline"
    EXEC_QUEUE       = "default"

    # Replace with your dataset ID
    DATASET_ID = "105163c10d0a4bbaa06055807084ec71"

    pipe = PipelineController(
        project=PIPELINE_PROJECT,
        name=PIPELINE_NAME,
        version="0.1",
        add_pipeline_tags=True,
        description="3-step pipeline for plant disease detection"
    )

    # 1) fetch the dataset
    fetch_step = pipe.add_function_step(
        name="fetch_dataset",
        function=fetch_dataset,
        function_kwargs={
            "dataset_id": DATASET_ID,
            "output_dir": "data"
        },
        execution_queue=EXEC_QUEUE,
        cache_executed_step=True
    )

    # 2) preprocess
    preprocess_step = pipe.add_function_step(
        name="preprocess",
        function=preprocess,
        function_kwargs={
            "output_dir": "data",
            "processed_dir": "processed"
        },
        parents=[fetch_step],
        execution_queue=EXEC_QUEUE,
        cache_executed_step=True
    )

    # 3) train
    train_step = pipe.add_function_step(
        name="train",
        function=train,
        function_kwargs={
            "processed_dir": "processed",
            "model_dir": "models"
        },
        parents=[preprocess_step],
        execution_queue=EXEC_QUEUE,
        cache_executed_step=False
    )

    # kick it off
    pipe.start()

if __name__ == "__main__":
    main()
