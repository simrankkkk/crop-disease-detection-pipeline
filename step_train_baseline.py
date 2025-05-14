from clearml import Task, Dataset
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import os, numpy as np, pickle, matplotlib.pyplot as plt
import json
import seaborn as sns

# ✅ Create ClearML task
task = Task.init(project_name="T3chOpsClearMLProject", task_name="step_train_baseline", task_type=Task.TaskTypes.training)

# ✅ Force scalars to show up even if missing
val_acc_dummy = 0.0

default_args = {
    "learning_rate": 0.001,
    "dropout": 0.4,
    "epochs": 1,
    "train_split_ratio": 0.1,
    "val_split_ratio": 0.5,
    "image_size": 160
}
params = task.connect(default_args)

# Extract hyperparameters
lr = float(params["learning_rate"])
dropout = float(params["dropout"])
epochs = int(params["epochs"])
train_ratio = float(params["train_split_ratio"])
val_ratio = float(params["val_split_ratio"])
img_size = int(params["image_size"])

logger = task.get_logger()
print(f"📌 Using Args: lr={lr}, dropout={dropout}, epochs={epochs}, img={img_size}, train%={train_ratio}, val%={val_ratio}")

# ✅ Load dataset
dataset = Dataset.get(dataset_name="T3chOps_processed_data_split", dataset_project="T3chOpsClearMLProject", only_completed=True)
dataset_path = dataset.get_local_copy()
train_dir, val_dir, test_dir = [os.path.join(dataset_path, x) for x in ["train", "valid", "test"]]

# ✅ Load subset
def load_subset(ds_path, ratio):
    ds_full = tf.keras.preprocessing.image_dataset_from_directory(
        ds_path, image_size=(img_size, img_size), batch_size=32, shuffle=True
    )
    class_names = ds_full.class_names
    total = tf.data.experimental.cardinality(ds_full).numpy()
    subset = max(1, int(total * ratio))
    return ds_full.take(subset).prefetch(tf.data.AUTOTUNE), class_names

train_ds, class_names = load_subset(train_dir, train_ratio)
val_ds, _ = load_subset(val_dir, val_ratio)
test_ds, _ = load_subset(test_dir, 1.0)
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

# ✅ Train model
os.makedirs("outputs", exist_ok=True)
checkpoint_cb = ModelCheckpoint("outputs/best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint_cb], verbose=1)

# ✅ Log scalars manually for each epoch, even if missing
for i in range(epochs):
    train_acc = history.history.get("accuracy", [val_acc_dummy])[i]
    val_acc = history.history.get("val_accuracy", [val_acc_dummy])[i]
    train_loss = history.history.get("loss", [0.0])[i]
    val_loss = history.history.get("val_loss", [0.0])[i]

    logger.report_scalar("accuracy", "train_accuracy", iteration=i, value=train_acc)
    logger.report_scalar("accuracy", "val_accuracy", iteration=i, value=val_acc)
    logger.report_scalar("loss", "train_loss", iteration=i, value=train_loss)
    logger.report_scalar("loss", "val_loss", iteration=i, value=val_loss)

# ✅ Save model and training history
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

labels_present = sorted(unique_labels(y_true, y_pred))
filtered_class_names = [class_names[i] for i in labels_present]
report = classification_report(y_true, y_pred, target_names=filtered_class_names, zero_division=0)
print(report)
logger.report_text("📊 Classification Report:\n" + report)

# ✅ Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
task.upload_artifact("confusion_matrix", artifact_object="outputs/confusion_matrix.png")

# ✅ Training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history.get('accuracy', []), label="Train Acc")
plt.plot(history.history.get('val_accuracy', []), label="Val Acc")
plt.title("Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history.get('loss', []), label="Train Loss")
plt.plot(history.history.get('val_loss', []), label="Val Loss")
plt.title("Loss")
plt.legend()
plt.savefig("outputs/train_curves.png")
task.upload_artifact("training_curves", artifact_object="outputs/train_curves.png")

task.close()
print("✅ Baseline training complete — now fully HPO compatible.")
