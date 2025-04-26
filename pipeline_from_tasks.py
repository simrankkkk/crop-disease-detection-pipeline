from clearml import Task, Dataset
from clearml.automation.controller import PipelineDecorator
import os

# STEP 1: Upload
@PipelineDecorator.component(return_values=["dataset_path"])
def stage_upload():
    dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
    return dataset.get_local_copy()

# STEP 2: Preprocess
@PipelineDecorator.component(return_values=["output_dir"])
def stage_preprocess(dataset_path):
    from PIL import Image
    from pathlib import Path

    def preprocess(src, dst, size=(224, 224)):
        for class_dir in Path(src).iterdir():
            if class_dir.is_dir():
                out_dir = Path(dst) / class_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                for image_file in class_dir.glob("*"):
                    try:
                        with Image.open(image_file) as img:
                            img = img.convert("RGB")
                            img = img.resize(size)
                            img.save(out_dir / image_file.name)
                    except:
                        continue

    output_dir = "preprocessed_data"
    train_src = os.path.join(dataset_path, "train")
    valid_src = os.path.join(dataset_path, "valid")
    train_dst = os.path.join(output_dir, "train")
    valid_dst = os.path.join(output_dir, "valid")

    preprocess(train_src, train_dst)
    preprocess(valid_src, valid_dst)

    return os.path.abspath(output_dir)

# STEP 3: Train
@PipelineDecorator.component()
def stage_train(output_dir):
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    img_size = (224, 224)
    batch_size = 32

    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(
        os.path.join(output_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )
    val_gen = datagen.flow_from_directory(
        os.path.join(output_dir, "valid"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    base = MobileNetV2(include_top=False, input_shape=img_size + (3,), weights="imagenet")
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(train_gen.num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=5)

# PIPELINE WRAPPER
@PipelineDecorator.pipeline(name="Crop Pipeline", project="PlantPipeline", version="1.0")
def pipeline_flow():
    dataset_path = stage_upload()
    preprocessed = stage_preprocess(dataset_path=dataset_path)
    stage_train(output_dir=preprocessed)

# EXECUTE LOCALLY
if __name__ == "__main__":
    PipelineDecorator.run_locally()
    pipeline_flow()
