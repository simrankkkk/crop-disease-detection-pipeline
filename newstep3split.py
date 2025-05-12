#!/usr/bin/env python
# step3split.py  – quick‑test trainer, now robust if some classes are absent

import argparse, os, pickle, numpy as np, matplotlib.pyplot as plt, sys, subprocess
from clearml import Task, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras import layers, models, Input, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

# ensure seaborn is available for the heatmap
try:
    import seaborn as sns
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns

# ─── 1. CLI flags ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Hybrid model trainer (quick test)")
parser.add_argument('--lr',            type=float, default=1e-3)
parser.add_argument('--batch_size',    type=int,   default=32)
parser.add_argument('--dropout',       type=float, default=0.4)
parser.add_argument('--epochs',        type=int,   default=1)
parser.add_argument('--subset_ratio',  type=float, default=0.1)
args = parser.parse_args()

# ─── 2. ClearML task ─────────────────────────────────────────────────────
task = Task.init(project_name="VisiblePipeline", task_name="step_train")
task.connect(args)
logger = task.get_logger()

# ─── 3. Dataset ──────────────────────────────────────────────────────────
ds = Dataset.get(dataset_name="plant_processed_data_split",
                 dataset_project="VisiblePipeline",
                 only_completed=True)
root  = ds.get_local_copy()
paths = {s: os.path.join(root, s) for s in ['train', 'valid', 'test']}

class_names, num_classes = None, None
def make_ds(split, shuffle):
    global class_names, num_classes
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        paths[split], image_size=(160,160), batch_size=args.batch_size, shuffle=shuffle)
    if class_names is None:
        class_names  = ds.class_names
        num_classes  = len(class_names)
    if 0 < args.subset_ratio < 1.0:
        n = tf.data.experimental.cardinality(ds).numpy()
        ds = ds.take(int(n * args.subset_ratio))
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = make_ds('train', shuffle=True)
val_ds   = make_ds('valid', shuffle=False)
test_ds  = make_ds('test',  shuffle=False)

# ─── 4. Model ────────────────────────────────────────────────────────────
inp  = Input(shape=(160,160,3))
mnet = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inp)
dnet = DenseNet121(include_top=False, weights='imagenet', input_tensor=inp)
mnet.trainable, dnet.trainable = True, False

x = layers.Concatenate()([
        layers.GlobalAveragePooling2D()(mnet.output),
        layers.GlobalAveragePooling2D()(dnet.output)
])
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(args.dropout)(x)
out = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=inp, outputs=out)
model.compile(optimizer=optimizers.Adam(learning_rate=args.lr),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ─── 5. Train ────────────────────────────────────────────────────────────
ckpt = ModelCheckpoint("best.h5", save_best_only=True,
                       monitor='val_accuracy', mode='max')
history = model.fit(train_ds, validation_data=val_ds,
                    epochs=args.epochs, callbacks=[ckpt])

# ─── 6. Artifacts & evaluation ──────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
model.save("outputs/final.h5")
task.upload_artifact("model_final", "outputs/final.h5")
task.upload_artifact("model_best",  "best.h5")

y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)

all_labels = list(range(num_classes))                     # <- fixed
report = classification_report(
    y_true, y_pred,
    labels=all_labels,
    target_names=class_names,
    zero_division=0
)
logger.report_text(report)

cm = confusion_matrix(y_true, y_pred, labels=all_labels)  # <- fixed
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix ({} classes)".format(num_classes))
plt.savefig("outputs/confusion_matrix.png")
task.upload_artifact("confusion_matrix", "outputs/confusion_matrix.png")

# training curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],     label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy'); plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'],     label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss'); plt.legend()
plt.savefig("outputs/train_curves.png")
task.upload_artifact("training_curves", "outputs/train_curves.png")

task.close()
print("✅ Quick‑test run finished without class‑mismatch errors.")
