# finalstep3.py

import argparse
import os, pickle
from clearml import Task, Dataset
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Parse CLI args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_id",     type=str,   required=True)
parser.add_argument("--learning_rate",  type=float, default=0.001)
parser.add_argument("--dropout",        type=float, default=0.4)
parser.add_argument("--epochs",         type=int,   default=1)
parser.add_argument("--image_size",     type=int,   default=160)
parser.add_argument("--train_ratio",    type=float, default=0.1)
parser.add_argument("--val_ratio",      type=float, default=0.5)
args = parser.parse_args()

# ─── Init ClearML task ───────────────────────────────────────────────────────────
task = Task.init(
    project_name="FinalProject",
    task_name="final_step_baseline_train",
    task_type=Task.TaskTypes.training
)

# ─── Unpack hyperparameters ──────────────────────────────────────────────────────
lr         = args.learning_rate
dropout    = args.dropout
epochs     = args.epochs
img_size   = args.image_size
train_ratio= args.train_ratio
val_ratio  = args.val_ratio

# ─── Load dataset ───────────────────────────────────────────────────────────────
dataset = Dataset.get(dataset_id=args.dataset_id)
local_path = dataset.get_local_copy()
train_dir = os.path.join(local_path, "train")
val_dir   = os.path.join(local_path, "valid")
test_dir  = os.path.join(local_path, "test")

def load_subset(path, ratio):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path, image_size=(img_size, img_size), batch_size=32, shuffle=True
    )
    total = tf.data.experimental.cardinality(ds).numpy()
    subset = max(1, int(total * ratio))
    return ds.take(subset).prefetch(tf.data.AUTOTUNE), ds.class_names

train_ds, class_names = load_subset(train_dir, train_ratio)
val_ds, _              = load_subset(val_dir, val_ratio)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=(img_size, img_size), batch_size=32, shuffle=False
).prefetch(tf.data.AUTOTUNE)

# ─── Build & compile model ───────────────────────────────────────────────────────
inp = Input(shape=(img_size, img_size, 3))
mnet = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inp); mnet.trainable=True
dnet = DenseNet121(include_top=False, weights="imagenet", input_tensor=inp); dnet.trainable=False

m_out = layers.GlobalAveragePooling2D()(mnet.output)
d_out = layers.GlobalAveragePooling2D()(dnet.output)
x = layers.Concatenate()([m_out, d_out])
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(dropout)(x)
out = layers.Dense(len(class_names), activation="softmax")(x)

model = Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ─── Train & checkpoint ─────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
checkpoint = ModelCheckpoint(
    "outputs/best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max"
)
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint])

# ─── Log scalars ────────────────────────────────────────────────────────────────
logger = task.get_logger()
for i in range(epochs):
    logger.report_scalar("accuracy",   "train_accuracy", iteration=i, value=history.history["accuracy"][i])
    logger.report_scalar("accuracy",   "val_accuracy",   iteration=i, value=history.history["val_accuracy"][i])
    logger.report_scalar("loss",       "train_loss",     iteration=i, value=history.history["loss"][i])
    logger.report_scalar("loss",       "val_loss",       iteration=i, value=history.history["val_loss"][i])

# ─── Save artifacts ──────────────────────────────────────────────────────────────
model.save("outputs/final_model.h5")
with open("outputs/train_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

task.upload_artifact("final_model",     artifact_object="outputs/final_model.h5")
task.upload_artifact("best_model",      artifact_object="outputs/best_model.h5")
task.upload_artifact("training_history", artifact_object="outputs/train_history.pkl")

# ─── Final reporting ─────────────────────────────────────────────────────────────
task.close()

# Print a marker so HPO can parse it
best_val = max(history.history["val_accuracy"])
print(f"__BEST_VAL__:{best_val:.4f}", flush=True)
