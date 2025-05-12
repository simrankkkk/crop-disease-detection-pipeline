# pipeline_decorator.py  – ClearML pipeline (Fetch Processed ➜ Quick Train ➜ HPO ➜ Final Train)
"""End‑to‑end pipeline using your **already‑split dataset**
(ID =`8ca91c7e2c8e42568425a921f85e4d0e`).

Workflow
─────────
1. **stage_fetch_processed**   Downloads the processed/split dataset.
2. **stage_train_quick**       1‑epoch smoke test on 10 % of the data.
3. **stage_hpo**               Random‑search HPO (8 trials) cloning the quick step.
4. **stage_final_train**       30 ‑epoch full run on 100 % data with the best params.

All stages log models, confusion matrices, and training curves; the pipeline
appears under **VisiblePipeline ➜ Pipelines** as **Crop‑Disease‑Pipeline**.
"""

from clearml.automation.controller import PipelineDecorator
from clearml import Task, Dataset
from pathlib import Path
import os, json, subprocess, sys
from typing import Dict

IMAGE_SIZE = (224, 224)

# helper for lazy install inside ClearML agent
_install = lambda pkg: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ╭──────────────────────────────────────────╮
# │ 1️⃣  FETCH PROCESSED DATASET             │
# ╰──────────────────────────────────────────╯
@PipelineDecorator.component(cache=True, execution_queue="default")
def stage_fetch_processed(dataset_id: str = "8ca91c7e2c8e42568425a921f85e4d0e") -> str:
    ds = Dataset.get(dataset_id=dataset_id)
    path = ds.get_local_copy()
    print(f"✅ Processed dataset fetched to {path}")
    return ds.id  # pass the dataset ID downstream

# ╭──────────────────────────────────────────╮
# │ 2️⃣  QUICK TRAIN                         │
# ╰──────────────────────────────────────────╯
@PipelineDecorator.component(execution_queue="default")
def stage_train_quick(dataset_id: str,
                      lr: float = 1e-3,
                      batch_size: int = 32,
                      dropout: float = 0.4,
                      epochs: int = 1,
                      subset_ratio: float = 0.1) -> str:
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.applications import MobileNetV2, DenseNet121
        from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, Dense, Dropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
    except ModuleNotFoundError as e:
        _install(str(e).split("'")[1]); return stage_train_quick(dataset_id, lr, batch_size, dropout, epochs, subset_ratio)

    task = Task.current_task(); ds = Dataset.get(dataset_id=dataset_id)
    base = ds.get_local_copy(); paths = [os.path.join(base, s) for s in ("train","valid","test")]
    gen = ImageDataGenerator(rescale=1/255)
    train_gen = gen.flow_from_directory(paths[0], IMAGE_SIZE, batch_size=batch_size, class_mode="categorical")
    val_gen   = gen.flow_from_directory(paths[1], IMAGE_SIZE, batch_size=batch_size, class_mode="categorical")
    test_gen  = gen.flow_from_directory(paths[2], IMAGE_SIZE, batch_size=batch_size, class_mode="categorical", shuffle=False)

    if subset_ratio < 1.0:
        train_gen.samples = int(train_gen.samples * subset_ratio)
        val_gen.samples   = int(val_gen.samples * subset_ratio)
        test_gen.samples  = int(test_gen.samples * subset_ratio)

    inp = Input(shape=(*IMAGE_SIZE,3))
    m1 = MobileNetV2(include_top=False, input_tensor=inp, weights="imagenet")
    m2 = DenseNet121(include_top=False, input_tensor=inp, weights="imagenet")
    for l in (*m1.layers, *m2.layers): l.trainable = False

    merged = Concatenate()([GlobalAveragePooling2D()(m1.output), GlobalAveragePooling2D()(m2.output)])
    x = Dense(256, activation="relu")(merged); x = Dropout(dropout)(x)
    out = Dense(train_gen.num_classes, activation="softmax")(x)
    model = Model(inp, out); model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    loss, acc = model.evaluate(test_gen, verbose=0); task.get_logger().report_scalar("test_accuracy", "quick", acc, 0)

    # artifacts: confusion matrix + curves + model
    y_pred = model.predict(test_gen, verbose=0).argmax(axis=1)
    cm = confusion_matrix(test_gen.classes, y_pred)
    sns.heatmap(cm, annot=False, cmap="Blues").figure.savefig("cm_quick.png"); task.upload_artifact("cm_quick", "cm_quick.png")
    fig, (a1,a2) = plt.subplots(1,2, figsize=(10,4))
    a1.plot(hist.history['accuracy'], label='train'); a1.plot(hist.history['val_accuracy'], label='val'); a1.set_title('Acc'); a1.legend()
    a2.plot(hist.history['loss'], label='train'); a2.plot(hist.history['val_loss'], label='val'); a2.set_title('Loss'); a2.legend()
    fig.savefig('curves_quick.png'); task.upload_artifact('curves_quick','curves_quick.png')
    model.save('model_quick.h5'); task.upload_artifact('model_quick','model_quick.h5')
    return task.id  # baseline task id for HPO

# ╭──────────────────────────────────────────╮
# │ 3️⃣  HPO                                 │
# ╰──────────────────────────────────────────╯
@PipelineDecorator.component(execution_queue="default")
def stage_hpo(base_task_id: str,
              max_trials: int = 8,
              queue: str = "default") -> str:
    try:
        from clearml.automation.opt import HyperParameterOptimizer, UniformParameterRange, DiscreteParameterRange
    except ModuleNotFoundError:
        from clearml.automation import HyperParameterOptimizer, UniformParameterRange, DiscreteParameterRange

    task = Task.current_task()
    space = [
        UniformParameterRange("lr", 1e-4, 1e-2, log_scale=True),
        DiscreteParameterRange("batch_size", [16,32,64]),
        DiscreteParameterRange("dropout", [0.3,0.4,0.5])
    ]
    opt = HyperParameterOptimizer(
        base_task_id=base_task_id, hyper_parameters=space,
        objective_metric_title="val_accuracy", objective_metric_sign="max",
        max_total_number_of_configs=max_trials, max_number_of_concurrent_tasks=2,
        optimizer_class="random_search", execute_queue=queue)
    best = opt.start()
    Path("best_params.json").write_text(json.dumps(best, indent=2))
    task.upload_artifact("best_params","best_params.json")
    return json.dumps(best)

# ╭──────────────────────────────────────────╮
# │ 4️⃣  FINAL TRAIN                         │
# ╰──────────────────────────────────────────╯
@PipelineDecorator.component(execution_queue="default")
def stage_final_train(dataset_id: str,
                      best_params_json: str,
                      epochs: int = 30) -> None:
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.applications import MobileNetV2, DenseNet121
        from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, Dense, Dropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
    except ModuleNotFoundError as e:
        _install(str(e).split("'")[1]); return stage_final_train(dataset_id, best_params_json, epochs)

    best = json.loads(best_params_json)
    lr = best.get("lr", 1e-3); batch_size = best.get("batch_size", 32); dropout = best.get("dropout", 0.4)

    task = Task.current_task(); ds = Dataset.get(dataset_id=dataset_id)
    base = ds.get_local_copy(); paths = [os.path.join(base, s) for s in ("train","valid","test")]
    gen = ImageDataGenerator(rescale=1/255)
    train_gen = gen.flow_from
