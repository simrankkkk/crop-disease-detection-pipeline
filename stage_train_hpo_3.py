from clearml import Task, Dataset
import os, sys, subprocess, numpy as np, pickle
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ✅ Fix seaborn/matplotlib conflict (if needed)
try:
    import seaborn as sns
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn==0.13.2", "matplotlib==3.10.1"])
    import seaborn as sns

from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# ✅ Init ClearML task
task = Task.init(project_name="VisiblePipeline", task_name="stage_train_hpo_3")
logger = task.get_logger()

# ✅ Load dataset from ClearML
dataset = Dataset.get(dataset_name="plant_processed_data_split", dataset_project="VisiblePipeline", only_completed=True)
dataset_path = dataset.get_local_copy()

# ✅ Define paths
train_dir = os.path.join(dataset_path, "train")
val_dir   = os.path.join(dataset_path, "valid")
test_dir  = os.path.join(dataset_path, "test")

# ✅ Load data
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 2  # 👈 Faster HPO testing

train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
val_ds   = tf.keras.preprocessing.image_dataset_from_directory(val_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
test_ds  = tf.keras.preprocessing.image_dataset_from_directory(test_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

# ✅ Get class names before subsetting
class_names = train_ds.class_names
num_classes = len(class_names)

# ✅ Hardcoded: use 10% subset for HPO
subset_ratio = 0.1

def subset_dataset(dataset, ratio):
    total = tf.data.experimental.cardinality(dataset).numpy()
    return dataset.take(int(total * ratio))

train_ds = subset_dataset(train_ds, subset_ratio)
val_ds   = subset_dataset(val_ds, subset_ratio)

# ✅ Prefetch for speed
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)

# ✅ Hyperparameters from ClearML
lr = float(task.get_parameter("General/learning_rate", 0.001))
dropout = float(task.get_parameter("General/dropout", 0.4))
dense_units = int(task.get_parameter("General/dense_units", 256))

# ✅ Build hybrid model
inp = Input(shape=(160, 160, 3))
mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inp)
mobilenet.trainable = True

densenet = DenseNet121(include_top=False, weights='imagenet', input_tensor=inp)
densenet.trainable = False

m_out = layers.GlobalAveragePooling2D()(mobilenet.output)
d_out = layers.GlobalAveragePooling2D()(densenet.output)
merged = layers.Concatenate()([m_out, d_out])
x = layers.Dense(dense_units, activation='relu')(merged)
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
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[checkpoint_cb])
model.save("outputs/final_model.h5")

# ✅ Save training artifacts
with open("outputs/train_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
task.upload_artifact("final_model", artifact_object="outputs/final_model.h5")
task.upload_artifact("best_model", artifact_object="outputs/best_model.h5")
task.upload_artifact("training_history", artifact_object="outputs/train_history.pkl")

# ✅ Evaluate model
y_true = np.concatenate([y.numpy() for x, y in test_ds])
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(y_true, y_pred, target_names=class_names)
logger.report_text("Classification Report:\\n" + report)

# ✅ Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
task.upload_artifact("confusion_matrix", artifact_object="outputs/confusion_matrix.png")

# ✅ Accuracy/loss curves
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
print("✅ stage_train_hpo_3 complete.")
