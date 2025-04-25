# step2_data_preprocessing.py
import os, shutil
from clearml import Task, Dataset

if __name__ == '__main__':
    task = Task.init(
        project_name='PlantPipeline',
        task_name='Step2-PreprocessData',
        task_type=Task.TaskTypes.data_processing
    )

    # Get dataset_id passed in via pipeline (or hard-code for manual run)
    dataset_id = task.get_parameter('dataset_id', default='105163c10d0a4bbaa06055807084ec71')

    # 1) Pull raw images
    ds = Dataset.get(dataset_id=dataset_id)
    raw_dir = ds.get_local_copy()
    print(f"Raw data directory: {raw_dir}")

    # 2) Preprocess exactly as in your ais_personal notebook
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    IMG_SIZE = (160,160)
    BATCH    = 32
    datagen  = ImageDataGenerator(rescale=1./255)

    # ensure output folder is clean
    processed_dir = os.path.abspath('preprocessed')
    shutil.rmtree(processed_dir, ignore_errors=True)
    os.makedirs(processed_dir, exist_ok=True)

    # generate one batch to materialize files
    datagen.flow_from_directory(
        raw_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode='categorical',
        save_to_dir=processed_dir,
        save_format='png'
    ).next()

    # 3) (Optional) log this processed folder as a new Dataset
    # pre_ds = Dataset.create(
    #     dataset_name='PlantDiseasePreprocessed',
    #     dataset_project='PlantPipeline',
    # )
    # pre_ds.add_files(processed_dir)
    # pre_ds.upload()
    # pre_ds.finalize()
    # print(f"↑ Preprocessed dataset version: {pre_ds.id}")

    task.upload_artifact('preprocessed_data', processed_dir)
    print(f"✅ Preprocessed data logged from: {processed_dir}")

    return processed_dir
