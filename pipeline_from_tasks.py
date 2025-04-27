# pipeline_from_tasks.py

from clearml.automation.controller import PipelineDecorator

# ─── Ensure all steps land on the “pipeline” queue ─────────────────────────────
PipelineDecorator.set_default_execution_queue("pipeline")

# ─── Stage 1: Upload ─────────────────────────────────────────────────────────
@PipelineDecorator.component(
    name="stage_upload",
    return_values=["uploaded_dataset_path"],
    execution_queue="pipeline",
    docker="tensorflow/tensorflow:2.11.0"
)
def stage_upload():
    from clearml import Dataset, Task
    dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
    path = dataset.get_local_copy(force=True)
    print("✅ Downloaded dataset to:", path)
    # auto‐upload original as artifact if you like:
    Task.current_task().upload_artifact("raw_dataset", path)
    return path

# ─── Stage 2: Preprocess ──────────────────────────────────────────────────────
@PipelineDecorator.component(
    name="stage_preprocess",
    return_values=["preprocessed_data_path"],
    execution_queue="pipeline",
    docker="tensorflow/tensorflow:2.11.0"
)
def stage_preprocess(uploaded_dataset_path):
    from pathlib import Path
    from PIL import Image
    from clearml import Task

    inp = Path(uploaded_dataset_path)
    out = Path("processed_data")
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("✅ Resizing images to 224×224 …")
    for cls in inp.iterdir():
        if not cls.is_dir(): continue
        dest = out / cls.name
        dest.mkdir(exist_ok=True)
        for img in cls.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                Image.open(img).convert("RGB").resize((224,224)).save(dest / img.name)
            except Exception as e:
                print(f"⚠️  Skipped {img.name}: {e}")

    # record the processed data
    Task.current_task().upload_artifact(name="preprocessed_dataset", artifact_object=str(out))
    print("✅ Preprocessing done.")
    return str(out)

# ─── Stage 3: Train with split ────────────────────────────────────────────────
@PipelineDecorator.component(
    name="stage_train",
    execution_queue="pipeline",
    docker="tensorflow/tensorflow:2.11.0"
)
def stage_train(preprocessed_data_path):
    from clearml import Task
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2, DenseNet121
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    import os

    base_dir = preprocessed_data_path
    img_size = (224,224)
    batch = 32
    val_split = 0.2

    gen = ImageDataGenerator(rescale=1./255, validation_split=val_split)
    train_gen = gen.flow_from_directory(
        base_dir, target_size=img_size, batch_size=batch,
        class_mode="categorical", subset="training"
    )
    val_gen = gen.flow_from_directory(
        base_dir, target_size=img_size, batch_size=batch,
        class_mode="categorical", subset="validation"
    )

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
    out = Dense(train_gen.num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=5)

    os.makedirs("model_output", exist_ok=True)
    save_path = os.path.join("model_output", "hybrid_model.h5")
    model.save(save_path)
    Task.current_task().upload_artifact(name="hybrid_model", artifact_object=save_path)
    print("✅ Training complete, model saved to:", save_path)

# ─── Controller: tie the steps together ───────────────────────────────────────
@PipelineDecorator.pipeline(
    name="Crop Disease Detection Pipeline",
    project="PlantPipeline",
    version="1.0"
)
def crop_pipeline():
    ds = stage_upload()
    pp = stage_preprocess(uploaded_dataset_path=ds)
    stage_train(preprocessed_data_path=pp)

if __name__ == "__main__":
    crop_pipeline()
