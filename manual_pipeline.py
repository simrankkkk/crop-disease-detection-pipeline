# manual_pipeline.py

from clearml import Dataset, Task
from clearml.automation.controller import PipelineController
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os, shutil

# ─── Optional callbacks ───────────────────────────────────────────────────────
def pre_execute_callback(pipeline, node, param_override):
    print(f"[CB] About to run step '{node.name}' with params: {param_override}")
    return True

def post_execute_callback(pipeline, node):
    print(f"[CB] Completed step '{node.name}', Task ID = {node.executed}")

# ─── Step 1: download raw dataset ──────────────────────────────────────────────
def stage_upload():
    ds = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
    path = ds.get_local_copy()
    print("✅ Raw dataset at:", path)
    return path

# ─── Step 2: preprocess & publish dataset ──────────────────────────────────────
def stage_preprocess(uploaded_dataset_path):
    inp = Path(uploaded_dataset_path)
    out = Path("processed_data")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("🔄 Resizing images to 224×224…")
    for split in ("train", "valid"):
        src, dst = inp / split, out / split
        dst.mkdir(exist_ok=True)
        for cls in src.iterdir():
            if not cls.is_dir(): continue
            (dst / cls.name).mkdir(exist_ok=True)
            for img in cls.iterdir():
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    try:
                        Image.open(img).convert("RGB").resize((224,224))\
                             .save(dst / cls.name / img.name)
                    except Exception as e:
                        print(f"⚠️ Skipped {img.name}: {e}")

    # publish as new ClearML Dataset
    new_ds = Dataset.create(
        dataset_name="plant_processed_data",
        dataset_project="PlantPipeline"
    )
    new_ds.add_files(str(out))
    new_ds.upload()
    new_ds.finalize()
    print("✅ Published processed dataset ID = ", new_ds.id)
    return new_ds.id

# ─── Step 3: train the hybrid model ────────────────────────────────────────────
def stage_train(processed_dataset_id):
    ds = Dataset.get(dataset_id=processed_dataset_id)
    base = ds.get_local_copy()
    print("✅ Retrieved processed data from:", base)

    train_dir, valid_dir = os.path.join(base,"train"), os.path.join(base,"valid")
    gen = ImageDataGenerator(rescale=1.0/255)
    train_gen = gen.flow_from_directory(train_dir, target_size=(224,224),
                                        batch_size=32, class_mode="categorical")
    val_gen   = gen.flow_from_directory(valid_dir, target_size=(224,224),
                                        batch_size=32, class_mode="categorical")

    inp = Input(shape=(224,224,3))
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

    os.makedirs("model_output", exist_ok=True)
    model_path = os.path.join("model_output", "hybrid_model.h5")
    model.save(model_path)
    Task.current_task().upload_artifact(name="hybrid_model", artifact_object=model_path)
    print("✅ Training complete, model saved to:", model_path)

# ─── Assemble & launch the pipeline ───────────────────────────────────────────
if __name__ == "__main__":
    pipe = PipelineController(
        name="New Crop Pipeline",      # this tile title is new/different
        project="PlantPipeline",   # same project folder
        version="1.0",
        add_pipeline_tags=False
    )
    pipe.set_default_execution_queue("default") 

    pipe.add_function_step("stage_upload",        stage_upload,       pre_execute_callback=pre_execute_callback)
    pipe.add_function_step("stage_preprocess",    stage_preprocess,   parents=["stage_upload"],    pre_execute_callback=pre_execute_callback)
    pipe.add_function_step("stage_train",         stage_train,        parents=["stage_preprocess"], pre_execute_callback=pre_execute_callback, post_execute_callback=post_execute_callback)

    pipe.start()
