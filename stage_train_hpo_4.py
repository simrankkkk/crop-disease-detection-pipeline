from clearml import Task, Dataset
import os, sys, subprocess, pickle, numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ─── 1) Initialize ClearML Task ────────────────────────────────────────────────
task = Task.init(project_name="VisiblePipeline", task_name="stage_train_hpo_3")
logger = task.get_logger()

# ─── 2) Fetch & Register Hyperparameters ────────────────────────────────────────
# Default values: subset 100%, 2 epochs (you can override via pipeline or HPO)
subset_ratio = float(task.get_parameter("General/subset_ratio", 1.0))
learning_rate = float(task.get_parameter("General/learning_rate", 0.001))
dropout_rate = float(task.get_parameter("General/dropout", 0.4))
dense_units = int(task.get_parameter("General/dense_units", 256))
epochs = int(task.get_parameter("General/epochs", 2))

# Register them so HPO & pipeline see them
task.connect({
    "General": {
        "subset_ratio": subset_ratio,
        "learning_rate": learning_rate,
        "dropout": dropout_rate,
        "dense_units": dense_units,
        "epochs": epochs
    }
})

# ─── 3) Load & Subset Dataset ───────────────────────────────────────────────────
dataset = Dataset.get(dataset_name="plant_processed_data_split",
                      dataset_project="VisiblePipeline", only_completed=True)
base_path = dataset.get_local_copy()

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(base_path, "train"), image_size=(160,160), batch_size=32)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(base_path, "valid"), image_size=(160,160), batch_size=32)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(base_path, "test"),  image_size=(160,160), batch_size=32)

# Grab class names before subsetting
class_names = train_ds.class_names
num_classes = len(class_names)

# Subset if desired
def subset(ds, ratio):
    count = tf.data.experimental.cardinality(ds).numpy()
    return ds.take(int(count * ratio))

train_ds = subset(train_ds, subset_ratio)
val_ds   = subset(val_ds,   subset_ratio)

# Prefetch for performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)

# ─── 4) Build & Compile Hybrid Model ────────────────────────────────────────────
inp = tf.keras.Input(shape=(160,160,3))
# MobileNetV2 branch
m = tf.keras.applications.MobileNetV2(
    include_top=False, weights="imagenet", input_tensor=inp)
m.trainable = True
# DenseNet121 branch
d = tf.keras.applications.DenseNet121(
    include_top=False, weights="imagenet", input_tensor=inp)
d.trainable = False

m_out = tf.keras.layers.GlobalAveragePooling2D()(m.output)
d_out = tf.keras.layers.GlobalAveragePooling2D()(d.output)
x = tf.keras.layers.Concatenate()([m_out, d_out])
x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(dropout_rate)(x)
out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs=inp, outputs=out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ─── 5) Train & Save Artifacts ─────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
ckpt = tf.keras.callbacks.ModelCheckpoint(
    "outputs/best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[ckpt])
model.save("outputs/final_model.h5")
with open("outputs/train_history.pkl","wb") as f:
    pickle.dump(history.history, f)

# Upload artifacts
task.upload_artifact("best_model", "outputs/best_model.h5")
task.upload_artifact("final_model","outputs/final_model.h5")
task.upload_artifact("history",    "outputs/train_history.pkl")

# ─── 6) Evaluate & Report ───────────────────────────────────────────────────────
y_true = np.concatenate([y.numpy() for _,y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names)
logger.report_text("Classification Report:\n" + report)

# Confusion matrix
import seaborn as sns
plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_true,y_pred), annot=True, fmt="d",
            xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.savefig("outputs/confusion_matrix.png")
task.upload_artifact("confusion_matrix","outputs/confusion_matrix.png")

# Training curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend(); plt.title("Accuracy")
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend(); plt.title("Loss")
plt.savefig("outputs/train_curves.png")
task.upload_artifact("train_curves","outputs/train_curves.png")

task.close()
print("✅ stage_train_hpo_3 complete.")
