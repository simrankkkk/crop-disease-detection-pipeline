import argparse
from clearml import Task, Dataset
import os, pickle, numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ─── Parse & register args ────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--subset_ratio",  type=float, default=1.0)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--dropout",       type=float, default=0.4)
parser.add_argument("--dense_units",   type=int,   default=256)
parser.add_argument("--epochs",        type=int,   default=2)
args = parser.parse_args()

task = Task.init(project_name="VisiblePipeline", task_name="stage_train_hpo_3")
# this writes them under Args/... so HPO can override
task.connect(args)

# ─── Load & subset ──────────────────────────────────────────────────────────
ds = Dataset.get(dataset_name="plant_processed_data_split",
                 dataset_project="VisiblePipeline", only_completed=True)
base = ds.get_local_copy()
IMG_SIZE=(160,160); BATCH=32

def make_ds(split):
    return tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(base, split), image_size=IMG_SIZE, batch_size=BATCH
    )

train_ds = make_ds("train").take(int(tf.data.experimental.cardinality(make_ds("train")) * args.subset_ratio))
val_ds   = make_ds("valid").take(int(tf.data.experimental.cardinality(make_ds("valid")) * args.subset_ratio))
test_ds  = make_ds("test")

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)

# ─── Build model ────────────────────────────────────────────────────────────
inp = tf.keras.Input(shape=IMG_SIZE+(3,))
m = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_tensor=inp)
m.trainable = True
d = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_tensor=inp)
d.trainable = False

x = tf.keras.layers.Concatenate()([
    tf.keras.layers.GlobalAveragePooling2D()(m.output),
    tf.keras.layers.GlobalAveragePooling2D()(d.output)
])
x = tf.keras.layers.Dense(args.dense_units, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(args.dropout)(x)
out = tf.keras.layers.Dense(len(train_ds.class_names), activation="softmax")(x)

model = tf.keras.Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ─── Train & log val_accuracy ────────────────────────────────────────────────
history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2)
for epoch, val_acc in enumerate(history.history["val_accuracy"]):
    task.get_logger().report_scalar("accuracy", "val_accuracy", iteration=epoch, value=val_acc)

# ─── Save artifacts ──────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
model.save("outputs/final_model.h5")
with open("outputs/history.pkl","wb") as f:
    pickle.dump(history.history, f)
task.upload_artifact("final_model","outputs/final_model.h5")
task.upload_artifact("history","outputs/history.pkl")
task.close()
