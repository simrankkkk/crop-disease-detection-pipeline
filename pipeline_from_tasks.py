from clearml import PipelineDecorator, Task
from clearml import Dataset
from pathlib import Path
from PIL import Image
import shutil
import zipfile
import os
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

@PipelineDecorator.pipeline(
    name="CropPipeline",
    project="PlantPipeline",
    version="1.0"
)


# ✨ Step 2: Preprocessing
@PipelineDecorator.component(name="stage_preprocess")
def stage_preprocess(dataset_path):
    output_dir = "processed_data"
    img_size = (224, 224)

    train_input = Path(dataset_path) / "train"
    valid_input = Path(dataset_path) / "valid"
    train_output = Path(output_dir) / "train"
    valid_output = Path(output_dir) / "valid"

    def process_images(src_dir, dst_dir):
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
                        img = img.resize(img_size)
                        img.save(dst_class / image_file.name)
                except Exception as e:
                    print(f"⚠️ Could not process {image_file}: {e}")

    train_output.mkdir(parents=True, exist_ok=True)
    valid_output.mkdir(parents=True, exist_ok=True)
    process_images(train_input, train_output)
    process_images(valid_input, valid_output)

    zip_path = "preprocessed_data.zip"
    shutil.make_archive("preprocessed_data", 'zip', output_dir)
    return zip_path

# ✨ Step 3: Train hybrid model
@PipelineDecorator.component(name="stage_train")
def stage_train(preprocessed_zip):
    extract_dir = "unzipped_data"
    with zipfile.ZipFile(preprocessed_zip, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    train_dir = os.path.join(extract_dir, "train")
    val_dir = os.path.join(extract_dir, "valid")

    img_size = (224, 224)
    batch_size = 32

    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical")
    val_gen = datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical")

    input_tensor = Input(shape=(224, 224, 3))

    mobilenet = MobileNetV2(include_top=False, input_tensor=input_tensor, weights="imagenet")
    densenet = DenseNet121(include_top=False, input_tensor=input_tensor, weights="imagenet")

    for layer in mobilenet.layers:
        layer.trainable = False
    for layer in densenet.layers:
        layer.trainable = False

    avg_pool_1 = GlobalAveragePooling2D()(mobilenet.output)
    avg_pool_2 = GlobalAveragePooling2D()(densenet.output)

    merged = Concatenate()([avg_pool_1, avg_pool_2])
    fc = Dense(256, activation="relu")(merged)
    fc = Dropout(0.3)(fc)
    output = Dense(train_gen.num_classes, activation="softmax")(fc)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_gen, validation_data=val_gen, epochs=5)

    Path("model_output").mkdir(exist_ok=True)
    model.save("model_output/hybrid_model.h5")
    return "model_output/hybrid_model.h5"

# ✨ Full Pipeline
@PipelineDecorator.pipeline(
    name="Crop Disease Detection Pipeline",
    project="PlantPipeline",
)
def crop_pipeline():
    dataset_path = stage_upload()
    preprocessed_zip = stage_preprocess(dataset_path=dataset_path)
    model_path = stage_train(preprocessed_zip=preprocessed_zip)
    return model_path

if __name__ == "__main__":
    PipelineDecorator.run_locally()
    crop_pipeline()
