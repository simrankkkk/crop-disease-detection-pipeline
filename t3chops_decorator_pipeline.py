# t3chops_decorator_pipeline.py

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

# ─── Stage 1: Upload raw dataset ──────────────────────────────────────────────
@PipelineDecorator.component(
    name="stage_upload",
    return_values=["raw_dataset_path"],
    execution_queue="default"
)
def stage_upload():
    ds = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
    path = ds.get_local_copy()
    print("✅ Raw dataset downloaded to:", path)
    return path

# ─── Stage 2: Split / preprocess ──────────────────────────────────────────────
@PipelineDecorator.component(
    name="stage_preprocess",
    return_values=["preprocessed_dataset_id"],
    execution_queue="default"
)
def stage_preprocess(raw_dataset_path):
    inp = Path(raw_dataset_path)
    out = Path("preprocessed_data")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print("🔄 Splitting and resizing to 224×224…")
    for split in ("train", "valid", "test"):
        src = inp / split
        dst = out / split
        dst.mkdir(exist_ok=True)
        for cls in src.iterdir():
            if not cls.is_dir(): continue
            tgt = dst / cls.name
            tgt.mkdir(exist_ok=True)
            for img in cls.iterdir():
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                try:
                    Image.open(img).convert("RGB")\
                         .resize((224,224))\
                         .save(tgt / img.name)
                except Exception as e:
                    print(f"⚠️ Skipped {img.name}: {e}")

    # publish as a new ClearML Dataset
    new_ds = Dataset.create(
        dataset_name="plant_preprocessed_data",
        dataset_project="T3chOpsDecoratorProject"
    )
    new_ds.add_files(str(out))
    new_ds.upload()
    new_ds.finalize()
    ds_id = new_ds.id
    print("✅ Published preprocessed dataset ID:", ds_id)
    return ds_id

# ─── Stage 3: Baseline training ───────────────────────────────────────────────
@PipelineDecorator.component(
    name="stage_train_baseline",
    return_values=["baseline_task_id"],
    execution_queue="default"
)
def stage_train_baseline(preprocessed_dataset_id):
    # You can either re-implement your training logic here,
    # or clone your existing step_train_baseline task:
    baseline_task = Task.clone(
        source_task=Task.get_task(project_name="T3chOpsClearMLProject",
                                  task_name="step_train_baseline"),
        name="baseline_clone"
    )
    baseline_task.execute_remotely(queue_name="default")
    baseline_task.wait()  # block until done
    print("✅ Baseline training finished:", baseline_task.id)
    return baseline_task.id

# ─── Stage 4: Manual HPO ──────────────────────────────────────────────────────
@PipelineDecorator.component(
    name="stage_hpo_manual_grid",
    return_values=["hpo_task_id","best_params"],
    execution_queue="default"
)
def stage_hpo_manual_grid(baseline_task_id, preprocessed_dataset_id):
    hpo_task = Task.clone(
        source_task=Task.get_task(project_name="T3chOpsClearMLProject",
                                  task_name="step_hpo_manual_grid"),
        name="hpo_clone"
    )
    # override args so it knows which baseline to tune
    hpo_task.set_parameter("Args/baseline_task_id", baseline_task_id)
    hpo_task.set_parameter("Args/dataset_id", preprocessed_dataset_id)
    hpo_task.execute_remotely(queue_name="default")
    hpo_task.wait()
    best = hpo_task.get_parameters()["best_params"]
    print("✅ HPO done:", hpo_task.id, "→", best)
    return hpo_task.id, best

# ─── Stage 5: Final training ──────────────────────────────────────────────────
@PipelineDecorator.component(
    name="stage_train_final",
    execution_queue="default"
)
def stage_train_final(hpo_task_id, preprocessed_dataset_id):
    final_task = Task.clone(
        source_task=Task.get_task(project_name="T3chOpsClearMLProject",
                                  task_name="step_train_final"),
        name="final_clone"
    )
    final_task.set_parameter("Args/hpo_task_id", hpo_task_id)
    final_task.set_parameter("Args/dataset_id", preprocessed_dataset_id)
    final_task.execute_remotely(queue_name="default")
    final_task.wait()
    print("✅ Final training complete:", final_task.id)

# ─── Pipeline definition ─────────────────────────────────────────────────────
@PipelineDecorator.pipeline(
    project="T3chOpsDecoratorProject",
    name="T3chOpsFullPipeline",
    version="1.0",
    queue="default"
)
def full_pipeline():
    raw_path      = stage_upload()
    preproc_id    = stage_preprocess(raw_dataset_path=raw_path)
    base_id       = stage_train_baseline(preprocessed_dataset_id=preproc_id)
    hpo_id, _     = stage_hpo_manual_grid(baseline_task_id=base_id,
                                          preprocessed_dataset_id=preproc_id)
    stage_train_final(hpo_task_id=hpo_id,
                      preprocessed_dataset_id=preproc_id)

if __name__ == "__main__":
    full_pipeline()
