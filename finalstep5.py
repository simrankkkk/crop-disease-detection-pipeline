# finalstep5.py

from clearml import Task, Dataset
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import os, numpy as np, pickle, matplotlib.pyplot as plt, seaborn as sns, json

# ✅ Start ClearML Task
task = Task.init(
    project_name="FinalProject",
    task_name="final_step_final_train",
    task_type=Task.TaskTypes.training
)

# ✅ Pull pipeline parameters
params = task.get_parameters()
dataset_id = params.get("Args/dataset_id")
hpo_task_id = params.get("Args/hpo_task_id")

if not hpo_task_id:
    raise ValueError("❌ hpo_task_id not passed from pipeline.")

# ✅ Load best params from HPO task
hpo_task = Task.get_task(task_id=hpo_task_id)
artifact = hpo_task.artifacts.get("best_result")
if artifact is None:
    raise ValueError("❌ best_result.json not found in HPO task.")

best_result = json.load(open(artifact.get_local_copy(), "r"))
best_params = best_result["best_params"]

lr = float(best_params["learning_rate"])
dropout = float(best_params["dropout"])
epochs = 2
img_size = 160
train_ratio = 0.1
val_ratio = 0.5

# ✅ Load dataset
if not dataset_id:
    raise ValueError("❌ dataset_id not provided.")
dataset = Dataset.get(dataset_id=dataset_id)
dataset_path = dataset.get_local_copy()

train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "valid")
test_dir = os.path.join(dataset_path, "test")

# ✅ Load subsets
def load_subset(ds_path, ratio):
    ds_full = tf.keras.preprocessing.image_dataset_from_directory(
        ds_path, image_size=(img_size, img_size), batch_size=32, shuffle=True
    )
    total = tf.data.experimental.cardinality(ds_full).numpy()
    subset = max(1, int(total * ratio))
    return ds_full.take(subset).prefetch(tf.data.AUTOTUNE)

train_ds = load_subset(train_dir, train_ratio)
val_ds = load_subset(val_dir, val_ratio)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=(img_size, img_size), batch_size=32, shuffle=False
).prefetch(tf.data.AUTOTUNE)

class_names = tf.keras.preprocessing.image_dataset_from_directory(train_dir).class_names
num_classes = len(class_names)

# ✅ Build hybrid model
inp = Input(shape=(img_size, img_size, 3))
mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inp)
mobilenet.trainable = True

densenet = DenseNet121(include_top=False, weights='imagenet', input_tensor=inp)
densenet.trainable = False

m_out = layers.GlobalAveragePooling2D()(mobilenet.output)
d_out = layers.GlobalAveragePooling2D()(densenet.output)
merged = layers.Concatenate()([m_out, d_out])
x = layers.Dense(256, activation='relu')(merged)
x = layers.BatchNormalization()(x)
x = layers.Dropout(dropout)(x)
out = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inp, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train model
os.makedirs("outputs", exist_ok=True)
checkpoint_cb = ModelCheckpoint("outputs/best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint_cb])

# ✅ Log scalars manually
logger = task.get_logger()
for i in range(epochs):
    logger.report_scalar("accuracy", "train_accuracy", i, history.history["accuracy"][i])
    logger.report_scalar("accuracy", "val_accuracy", i, history.history["val_accuracy"][i])
    logger.report_scalar("loss", "train_loss", i, history.history["loss"][i])
    logger.report_scalar("loss", "val_loss", i, history.history["val_loss"][i])

# ✅ Save outputs
model.save("outputs/final_model.h5")
with open("outputs/train_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

task.upload_artifact("final_model", artifact_object="outputs/final_model.h5")
task.upload_artifact("training_history", artifact_object="outputs/train_history.pkl")

# ✅ Evaluate
y_true = np.concatenate([y.numpy() for x, y in test_ds])
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
logger.report_text("📊 Final Classification Report:\n" + report)

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Final Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
task.upload_artifact("confusion_matrix", artifact_object="outputs/confusion_matrix.png")

# ✅ Training curves
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
print("✅ Final model training complete.")
