from clearml import PipelineDecorator

# Stage 1: Upload Dataset
@PipelineDecorator.component(name="stage_upload", return_values=["uploaded_dataset_id"])
def stage_upload():
    from clearml import Dataset
    # Download existing uploaded dataset
    dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
    uploaded_dataset_id = dataset.id
    print(f"✅ Stage 1 complete. Dataset ID: {uploaded_dataset_id}")
    return uploaded_dataset_id

# Stage 2: Preprocessing
@PipelineDecorator.component(name="stage_preprocess", return_values=["preprocessed_dataset_id"])
def stage_preprocess(uploaded_dataset_id):
    from clearml import Dataset
    from pathlib import Path
    from PIL import Image
    import shutil

    dataset = Dataset.get(dataset_id=uploaded_dataset_id)
    dataset_path = dataset.get_local_copy()

    output_dir = Path("processed_data")
    train_input = Path(dataset_path) / "train"
    valid_input = Path(dataset_path) / "valid"
    train_output = output_dir / "train"
    valid_output = output_dir / "valid"

    def process_images(src_dir, dst_dir, image_size=(224, 224)):
        for class_dir in src_dir.iterdir():
            if not class_dir.is_dir():
                continue
            dst_class = dst_dir / class_dir.name
            dst_class.mkdir(parents=True, exist_ok=True)
            for image_file in class_dir.iterdir():
                if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                try:
                    with Image.open(image_file) as img:
                        img = img.convert("RGB")
                        img = img.resize(image_size)
                        img.save(dst_class / image_file.name)
                except Exception as e:
                    print(f"⚠️ Could not process {image_file}: {e}")

    train_output.mkdir(parents=True, exist_ok=True)
    valid_output.mkdir(parents=True, exist_ok=True)
    process_images(train_input, train_output)
    process_images(valid_input, valid_output)

    # Upload processed dataset
    preprocessed_dataset = Dataset.create(
        dataset_project="PlantPipeline",
        dataset_name="Preprocessed Crop Dataset",
        parent_datasets=[dataset.id]
    )
    preprocessed_dataset.add_files(path=output_dir.as_posix())
    preprocessed_dataset.upload()
    preprocessed_dataset.finalize()

    print(f"✅ Stage 2 complete. Preprocessed Dataset ID: {preprocessed_dataset.id}")
    return preprocessed_dataset.id

# Stage 3: Train Model
@PipelineDecorator.component(name="stage_train")
def stage_train(preprocessed_dataset_id):
    from clearml import Dataset
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2, DenseNet121
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate
    from tensorflow.keras.optimizers import Adam
    import os
    from pathlib import Path

    dataset = Dataset.get(dataset_id=preprocessed_dataset_id)
    data_dir = dataset.get_local_copy()

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")
    batch_size = 32
    img_size = (224, 224)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
    val_gen = datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

    input_tensor = Input(shape=(224, 224, 3))
    mobilenet = MobileNetV2(include_top=False, input_tensor=input_tensor, weights='imagenet')
    densenet = DenseNet121(include_top=False, input_tensor=input_tensor, weights='imagenet')

    for layer in mobilenet.layers:
        layer.trainable = False
    for layer in densenet.layers:
        layer.trainable = False

    avg_pool_1 = GlobalAveragePooling2D()(mobilenet.output)
    avg_pool_2 = GlobalAveragePooling2D()(densenet.output)

    merged = Concatenate()([avg_pool_1, avg_pool_2])
    fc = Dense(256, activation='relu')(merged)
    fc = Dropout(0.3)(fc)
    output = Dense(train_gen.num_classes, activation='softmax')(fc)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=5)

    Path("model_output").mkdir(exist_ok=True)
    model.save("model_output/hybrid_model.h5")
    print("✅ Stage 3 complete. Model trained and saved.")

# PIPELINE WRAPPER
@PipelineDecorator.pipeline(name="Crop Pipeline", project="PlantPipeline", version="1.0")
def crop_pipeline():
    uploaded_dataset_id = stage_upload()
    preprocessed_dataset_id = stage_preprocess(uploaded_dataset_id=uploaded_dataset_id)
    stage_train(preprocessed_dataset_id=preprocessed_dataset_id)

# Run the pipeline
if __name__ == "__main__":
    PipelineDecorator.run()
    crop_pipeline()
