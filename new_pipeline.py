# pipeline_from_tasks.py

from clearml.automation.controller import PipelineController

def run_pipeline():
    # 1) Create a new Controller (new tile)
    pipe = PipelineController(
        name="Crop Pipeline Clone",   # this becomes the tile title
        project="PlantPipeline", 
        version="1.0",
        add_pipeline_tags=False
    )
    pipe.set_default_execution_queue("default")

    # 2) Clone your exact stage_upload task
    pipe.add_step(
        name="stage_upload",
        base_task_id="e85b62f486564ff5a6f2f70d928e0481",
        parameter_override={}  # no parameters needed
    )

    # 3) Clone your exact stage_preprocess task, passing the new upload’s ID
    pipe.add_step(
        name="stage_preprocess",
        parents=["stage_upload"],
        base_task_id="5d834d4721ea4a3faffb1a1110914d94",
        parameter_override={
            # replace this key with whatever your preprocess step expects
            "General/uploaded_dataset_task_id": "${stage_upload.id}"
        }
    )

    # 4) Clone your exact stage_train task, passing the new preprocess’s ID
    pipe.add_step(
        name="stage_train",
        parents=["stage_preprocess"],
        base_task_id="3cc5c80b43e04d01b2b6ddc41d6860f9",
        parameter_override={
            # replace this key with whatever your train step expects
            "General/processed_dataset_task_id": "${stage_preprocess.id}"
        }
    )

    # 5) Launch the Controller
    pipe.start()

if __name__ == "__main__":
    run_pipeline()
