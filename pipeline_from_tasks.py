# pipeline_from_tasks.py

from clearml import Dataset, Task
from clearml.automation.controller import PipelineDecorator
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os, shutil

# ─── Stage 1: Download raw dataset ────────────────────────────────────────────
@PipelineDecorator.component(
    name="stage_upload",
    return_values=["uploaded_dataset_path"],
    execution_queue="default"
)
def stage_upload():
    dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
    path = dataset.get_local_copy()
    print("✅ Raw dataset downloaded to:", path)
    return path

# ─── Stage 2: Preprocess & publish as new Dataset ─────────────────────────────
@PipelineDecorator.component(
    name="stage_preprocess",
    return_values=["processed_dataset_id"],
    execution_queue="default"
)
def stage_preprocess(uploaded_dataset_path):
    inp = Path(uploaded_dataset_path)
    out = Path("processed_data")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("🔄 Resizing images to 224×224…")
    for cls_dir in inp.iterdir():
        if not cls_dir.is_dir(): continue
        dst = out / cls_dir.name
        dst.mkdir(exist_ok=True)
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                Image.open(img_path).convert("RGB").resize((224,224)).save(dst / img_path.name)
            except Exception as e:
                print(f"⚠️ Skipped {img_path.name}: {e}")

    # Create & upload new ClearML Dataset for processed images
    ds = Dataset.create(
        dataset_name="plant_processed_data",
        dataset_project="PlantPipeline"
    )
    ds.add_files(str(out))
    ds.upload()             # ensure files are uploaded
    ds.finalize()           # registers the dataset
    dataset_id = ds.id      # <— grab the string ID here
    print("✅ Created processed dataset ID:", dataset_id)
    return dataset_id

# ─── Stage 3: Train hybrid model using processed Dataset ─────────────────────
@PipelineDecorator.component(
    name="stage_train",
    execution_queue="default"
)
def stage_train(processed_dataset_id):
    # fetch the processed-data dataset by its string ID
    ds = Dataset.get(dataset_id=processed_dataset_id)
    base_dir = ds.get_local_copy()
    print("✅ Retrieved processed data from:", base_dir)

    # Data generators with 20% validation split
    img_size, batch = (224,224), 32
    gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
    train_gen = gen.flow_from_directory(base_dir, target_size=img_size,
                                        batch_size=batch, class_mode="categorical",
                                        subset="training")
    val_gen   = gen.flow_from_directory(base_dir, target_size=img_size,
                                        batch_size=batch, class_mode="categorical",
                                        subset="validation")

    # Build frozen dual‐backbone model
    inp = Input(shape=(*img_size,3))
    m1 = MobileNetV2(include_top=False, input_tensor=inp, weights="imagenet")
    m2 = DenseNet121(include_top=False, input_tensor=inp, weights="imagenet")
    for layer in m1.layers + m2.layers:
        layer.trainable = False

    merged = Concatenate()([
        GlobalAveragePooling2D()(m1.output),
        GlobalAveragePooling2D()(m2.output)
    ])
    x = Dense(256, activation="relu")(merged)
    x = Dropout(0.3)(x)
    out_layer = Dense(train_gen.num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out_layer)
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=5)

    # Save & upload trained model
    os.makedirs("model_output", exist_ok=True)
    model_path = os.path.join("model_output", "hybrid_model.h5")
    model.save(model_path)
    Task.current_task().upload_artifact(name="hybrid_model", artifact_object=model_path)
    print("✅ Training complete, model saved to:", model_path)

# ─── Define & register the Pipeline ─────────────────────────────────────────
@PipelineDecorator.pipeline(
    name="Crop Disease Detection Pipeline",
    project="PlantPipeline",
    version="1.0"
)
def crop_pipeline():
    raw_path      = stage_upload()
    processed_id  = stage_preprocess(uploaded_dataset_path=raw_path)
    stage_train(processed_dataset_id=processed_id)

if __name__ == "__main__":
    crop_pipeline()
