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

# ─── Optional callbacks ────────────────────────────────────────────────────────
def pre_execute_callback(pipeline, node, param_override):
    print(f"[CB] About to run {node.name} with params {param_override}")
    return True

def post_execute_callback(pipeline, node):
    print(f"[CB] Finished {node.name}, Task ID={node.executed}")

# ─── Step 1: Download raw dataset ──────────────────────────────────────────────
def stage_upload():
    ds = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
    path = ds.get_local_copy()
    print("✅ Raw data at", path)
    return path  # returns a str

# ─── Step 2: Preprocess & publish ──────────────────────────────────────────────
def stage_preprocess(uploaded_dataset_path: str):
    inp = Path(uploaded_dataset_path)
    out = Path("processed_data")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("🔄 Resizing images…")
    for split in ("train","valid"):
        src, dst = inp/split, out/split
        dst.mkdir(exist_ok=True)
        for cls in src.iterdir():
            if not cls.is_dir(): continue
            (dst/cls.name).mkdir(exist_ok=True)
            for img in cls.iterdir():
                if img.suffix.lower() not in (".jpg",".jpeg",".png"): 
                    continue
                Image.open(img).convert("RGB")\
                     .resize((224,224))\
                     .save(dst/cls.name/img.name)

    new_ds = Dataset.create("plant_processed_data", "PlantPipeline")
    new_ds.add_files(str(out))
    new_ds.upload()
    new_ds.finalize()
    print("✅ Published processed dataset ID =", new_ds.id)
    return new_ds.id  # returns a str

# ─── Step 3: Train the hybrid model ────────────────────────────────────────────
def stage_train(processed_dataset_id: str):
    ds = Dataset.get(dataset_id=processed_dataset_id)
    base = ds.get_local_copy()
    print("✅ Retrieved processed data from", base)

    train_dir = os.path.join(base, "train")
    valid_dir = os.path.join(base, "valid")
    gen = ImageDataGenerator(rescale=1.0/255)
    train_gen = gen.flow_from_directory(train_dir, (224,224), batch_size=32, class_mode="categorical")
    val_gen   = gen.flow_from_directory(valid_dir, (224,224), batch_size=32, class_mode="categorical")

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
    print("✅ Model saved to", model_path)

# ─── Assemble & launch the pipeline ────────────────────────────────────────────
if __name__ == "__main__":
    pipe = PipelineController(
        name="Crop Pipeline",
        project="PlantPipeline",
        version="1.3",           # ← bumped to 1.3 for a fresh tile
        add_pipeline_tags=False
    )
    pipe.set_default_execution_queue("default")

    # Capture the raw-path return as "uploaded_dataset_path"
    pipe.add_function_step(
        name="stage_upload",
        function=stage_upload,
        function_return=["uploaded_dataset_path"],
        pre_execute_callback=pre_execute_callback
    )
    # Pass that into preprocess, capture its return as "processed_dataset_id"
    pipe.add_function_step(
        name="stage_preprocess",
        function=stage_preprocess,
        parents=["stage_upload"],
        function_kwargs={"uploaded_dataset_path": "${stage_upload.uploaded_dataset_path}"},
        function_return=["processed_dataset_id"],
        pre_execute_callback=pre_execute_callback
    )
    # Finally wire processed_dataset_id into training
    pipe.add_function_step(
        name="stage_train",
        function=stage_train,
        parents=["stage_preprocess"],
        function_kwargs={"processed_dataset_id": "${stage_preprocess.processed_dataset_id}"},
        pre_execute_callback=pre_execute_callback,
        post_execute_callback=post_execute_callback
    )

    pipe.start()
