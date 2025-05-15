# finalstep3.py (Baseline Training Script with Scalar Logging)

from clearml import Task, Dataset
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import os, numpy as np, pickle, matplotlib.pyplot as plt, seaborn as sns

# ✅ Init ClearML Task
task = Task.init(
    project_name="FinalProject",
    task_name="final_step_baseline_train",
    task_type=Task.TaskTypes.training
)

# ✅ Hyperparameters
def_params = {
    "learning_rate": 0.001,
    "dropout": 0.4,
    "epochs": 1,
    "image_size": 160,
    "train_ratio": 0.1,
    "val_ratio": 0.5
}
params = task.connect(def_params)

# ✅ Dataset loading
dataset_id = task.get_parameters().get("Args/dataset_id")
dataset = Dataset.get(dataset_id=dataset_id)
dataset_path = dataset.get_local_copy()
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "valid")
test_dir = os.path.join(dataset_path, "test")

# ✅ Image loading
def load_subset(ds_path, ratio):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        ds_path, image_size=(params["image_size"], params["image_size"]), batch_size=32, shuffle=True
    )
    total = tf.data.experimental.cardinality(ds).numpy()
    subset = max(1, int(total * ratio))
    return ds.take(subset).prefetch(tf.data.AUTOTUNE)

train_ds = load_subset(train_dir, params["train_ratio"])
val_ds = load_subset(val_dir, params["val_ratio"])
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=(params["image_size"], params["image_size"]), batch_size=32, shuffle=False
).prefetch(tf.data.AUTOTUNE)

class_names = tf.keras.preprocessing.image_dataset_from_directory(train_dir).class_names
num_classes = len(class_names)

# ✅ Build hybrid model
inp = Input(shape=(params["image_size"], params["image_size"], 3))
mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inp)
mobilenet.trainable = True
densenet = DenseNet121(include_top=False, weights='imagenet', input_tensor=inp)
densenet.trainable = False

m_out = layers.GlobalAveragePooling2D()(mobilenet.output)
d_out = layers.GlobalAveragePooling2D()(densenet.output)
merged = layers.Concatenate()([m_out, d_out])
x = layers.Dense(256, activation='relu')(merged)
x = layers.BatchNormalization()(x)
x = layers.Dropout(params["dropout"])(x)
out = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inp, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train model
os.makedirs("outputs", exist_ok=True)
checkpoint_cb = ModelCheckpoint("outputs/best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")
history = model.fit(train_ds, validation_data=val_ds, epochs=params["epochs"], callbacks=[checkpoint_cb])

# ✅ Log scalar metrics manually
logger = task.get_logger()
for i in range(params["epochs"]):
    if "accuracy" in history.history and "val_accuracy" in history.history:
        logger.report_scalar("accuracy", "train_accuracy", i, history.history["accuracy"][i])
        logger.report_scalar("accuracy", "val_accuracy", i, history.history["val_accuracy"][i])

# ✅ Save artifacts
model.save("outputs/final_model.h5")
with open("outputs/train_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

task.upload_artifact("final_model", artifact_object="outputs/final_model.h5")
task.upload_artifact("best_model", artifact_object="outputs/best_model.h5")
task.upload_artifact("training_history", artifact_object="outputs/train_history.pkl")

# ✅ Evaluation
y_true = np.concatenate([y.numpy() for x, y in test_ds])
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
logger.report_text("📊 Classification Report:\n" + report)

# ✅ Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
task.upload_artifact("confusion_matrix", artifact_object="outputs/confusion_matrix.png")

# ✅ Accuracy/loss plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss")
plt.legend()
plt.savefig("outputs/train_curves.png")
task.upload_artifact("training_curves", artifact_object="outputs/train_curves.png")

task.close()
print("✅ Baseline training complete.")
