# step1_fetch_dataset.py
from clearml import Task, Dataset

if __name__ == '__main__':
    # 1) Init ClearML Task
    task = Task.init(
        project_name='PlantPipeline',
        task_name='Step1-FetchDataset',
        task_type=Task.TaskTypes.data_processing
    )

    # 2) Fetch the existing dataset by its ID
    ds = Dataset.get(dataset_id='105163c10d0a4bbaa06055807084ec71')
    ds_folder = ds.get_local_copy()
    task.connect({'dataset_id': ds.id, 'dataset_folder': ds_folder})
    print(f"✅ Dataset fetched locally at: {ds_folder}")

    # 3) Return the dataset ID for the pipeline
    #    (ClearML pipeline controller will capture this)
    # Note: pipeline will use this return value
    return ds.id
