from clearml import Task, Dataset
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import os, numpy as np, pickle, matplotlib.pyplot as plt
import json, seaborn as sns

# ✅ Start ClearML task
task = Task.init(project_name="T3chOpsClearMLProject, task_name="step_train_final", task_type=Task.TaskTypes.training)

# ✅ Get HPO task and load best_result.json
params = task.get_parameters()
hpo_task_id = params.get("Args/hpo_task_id", "6995c0140b534b2e854ddf93590f2d3e")

print(f"📦 Using HPO Task ID: {hpo_task_id}")
hpo_task = Task.get_task(task_id=hpo_task_id)

artifact = hpo_task.artifacts.get("best_result")
if artifact is None:
    raise ValueError("❌ Could not find 'best_result' artifact in HPO task.")

best_result = json.load(open(artifact.get_local_copy(), "r"))

# ✅ Extract best parameters
best_params = best_result["best_params"]
lr = float(best_params["learning_rate"])
dropout = float(best_params["dropout"])
epochs = 3  # Adjust as needed
img_size = 160
train_ratio = 0.1  # ✅ Use 10% of training data
val_ratio = 0.5

logger = task.get_logger()
print(f"🔧 Training final model with: lr={lr}, dropout={dropout}, epochs={epochs}, train_ratio={train_ratio}")

# ✅ Load dataset
dataset_id = params.get("Args/dataset_id")
if dataset_id:
    print(f"📂 Loading dataset from pipeline: {dataset_id}")
    dataset = Dataset.get(dataset_id=dataset_id)
else:
    print("📂 No dataset_id passed — using fallback dataset manually.")
    dataset = Dataset.get(dataset_name="T3chOps_processed_data_split", dataset_project="T3chOpsClearMLProject", only_completed=True)
dataset_path = dataset.get_local_copy()
train_dir, val_dir, test_dir = [os.path.join(dataset_path, x) for x in ["train", "valid", "test"]]

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
    test_dir, image_size=(img_size, img_size), batch_size=32, shuffle=False).prefetch(tf.data.AUTOTUNE)

class_names = tf.keras.preprocessing.image_dataset_from_directory(train_dir).class_names
num_classes = len(class_names)

# ✅ Build model
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

# ✅ Train
os.makedirs("outputs", exist_ok=True)
checkpoint_cb = ModelCheckpoint("outputs/best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint_cb])

# ✅ Log scalars
for i in range(epochs):
    logger.report_scalar("accuracy", "train_accuracy", i, history.history["accuracy"][i])
    logger.report_scalar("accuracy", "val_accuracy", i, history.history["val_accuracy"][i])
    logger.report_scalar("loss", "train_loss", i, history.history["loss"][i])
    logger.report_scalar("loss", "val_loss", i, history.history["val_loss"][i])

# ✅ Save artifacts
model.save("outputs/final_model.h5")
with open("outputs/train_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

task.upload_artifact("final_model", artifact_object="outputs/final_model.h5")
task.upload_artifact("training_history", artifact_object="outputs/train_history.pkl")

# ✅ Evaluation
y_true = np.concatenate([y.numpy() for x, y in test_ds])
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

labels_present = sorted(unique_labels(y_true, y_pred))
filtered_class_names = [class_names[i] for i in labels_present]
report = classification_report(y_true, y_pred, target_names=filtered_class_names, zero_division=0)
logger.report_text("📊 Final Classification Report:\n" + report)
print(report)

# ✅ Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
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
print("✅ Final training complete and model artifacts saved.")
