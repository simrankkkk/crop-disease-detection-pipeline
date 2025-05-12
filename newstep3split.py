#!/usr/bin/env python
# step3split.py  – quick‑test version (10 % data, 1 epoch by default)

import argparse, os, pickle, numpy as np, matplotlib.pyplot as plt
from clearml import Task, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras import layers, models, Input, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import seaborn as sns

# ─── 1. Tunables & quick‑test flags ──────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--lr',            type=float, default=1e-3,  help='Learning rate')
parser.add_argument('--batch_size',    type=int,   default=32,    help='Batch size')
parser.add_argument('--dropout',       type=float, default=0.4,   help='Drop‑out rate')
parser.add_argument('--epochs',        type=int,   default=1,     help='Training epochs')
parser.add_argument('--subset_ratio',  type=float, default=0.10,  help='Fraction of data to use (0‑1)')
args = parser.parse_args()

# ─── 2. Register with ClearML ────────────────────────────────────────────
task = Task.init(project_name="VisiblePipeline", task_name="step_train")
task.connect(args)          # lr, batch_size, dropout, epochs, subset_ratio
logger = task.get_logger()

# ─── 3. Dataset paths ────────────────────────────────────────────────────
dataset = Dataset.get(
    dataset_name="plant_processed_data_split",
    dataset_project="VisiblePipeline",
    only_completed=True
)
root = dataset.get_local_copy()
dirs = {k: os.path.join(root, k) for k in ['train', 'valid', 'test']}

def make_ds(split):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs[split],
        image_size=(160, 160),
        batch_size=args.batch_size,
        shuffle=(split == 'train')
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    if 0 < args.subset_ratio < 1.0:
        n = tf.data.experimental.cardinality(ds).numpy()
        ds = ds.take(int(n * args.subset_ratio))
    return ds

train_ds = make_ds('train')
val_ds   = make_ds('valid')
test_ds  = make_ds('test')
class_names = train_ds.class_names
num_classes = len(class_names)

# ─── 4. Build hybrid model ───────────────────────────────────────────────
inp       = Input(shape=(160, 160, 3))
mnet      = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inp)
densenet  = DenseNet121(include_top=False, weights='imagenet', input_tensor=inp)
mnet.trainable = True
densenet.trainable = False

out = layers.Concatenate()([
        layers.GlobalAveragePooling2D()(mnet.output),
        layers.GlobalAveragePooling2D()(densenet.output)
      ])
out = layers.Dense(256, activation='relu')(out)
out = layers.BatchNormalization()(out)
out = layers.Dropout(args.dropout)(out)
out = layers.Dense(num_classes, activation='softmax')(out)
model = models.Model(inputs=inp, outputs=out)

model.compile(
    optimizer=optimizers.Adam(learning_rate=args.lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ─── 5. Training ────────────────────────────────────────────────────────
ckpt = ModelCheckpoint("best.h5", save_best_only=True, monitor='val_accuracy', mode='max')
history = model.fit(train_ds, validation_data=val_ds,
                    epochs=args.epochs, callbacks=[ckpt])

# ─── 6. Artifacts & quick metrics ────────────────────────────────────────
model.save("final.h5")
task.upload_artifact("model_final", "final.h5")
task.upload_artifact("model_best",  "best.h5")

y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)
report = classification_report(y_true, y_pred, target_names=class_names)
logger.report_text(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.savefig("confusion.png")
task.upload_artifact("confusion_matrix", "confusion.png")

task.close()
print("✅ quick‑test run complete.")
