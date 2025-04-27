from clearml import PipelineDecorator

# Stage 1: Upload Dataset
@PipelineDecorator.component(name="stage_upload", return_values=["uploaded_dataset_path"])
def stage_upload():
    from clearml import Dataset
    dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")  # Your uploaded dataset ID
    dataset_path = dataset.get_local_copy(force=True)
    return dataset_path

# Stage 2: Preprocess Dataset
@PipelineDecorator.component(name="stage_preprocess", return_values=["preprocessed_data_path"])
def stage_preprocess(uploaded_dataset_path):
    from pathlib import Path
    from PIL import Image
    import os

    output_dir = Path("processed_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in Path(uploaded_dataset_path).iterdir():
        if not class_dir.is_dir():
            continue
        output_class_dir = output_dir / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        for img_file in class_dir.glob("*"):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            try:
                img = Image.open(img_file).convert("RGB").resize((224, 224))
                img.save(output_class_dir / img_file.name)
            except Exception as e:
                print(f"⚠️ Error processing {img_file}: {e}")

    return str(output_dir)

# Stage 3: Train Model
@PipelineDecorator.component(name="stage_train")
def stage_train(preprocessed_data_path):
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2, DenseNet121
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate
    from tensorflow.keras.optimizers import Adam
    import os

    img_size = (224, 224)
    batch_size = 32

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2, rescale=1./255)

    train_gen = datagen.flow_from_directory(
        preprocessed_data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        preprocessed_data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

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

    os.makedirs("model_output", exist_ok=True)
    model.save("model_output/hybrid_model.h5")

# Define Full Pipeline
@PipelineDecorator.pipeline(
    name="Crop Disease Detection Pipeline",
    project="PlantPipeline",
    version="1.0",
)
def crop_pipeline():
    uploaded_dataset_path = stage_upload()
    preprocessed_data_path = stage_preprocess(uploaded_dataset_path=uploaded_dataset_path)
    stage_train(preprocessed_data_path=preprocessed_data_path)

if __name__ == "__main__":
    crop_pipeline()
