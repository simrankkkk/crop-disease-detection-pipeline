
import subprocess
import sys

# Auto-install seaborn if missing
try:
    import seaborn as sns
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns

from clearml import Task, Dataset
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import os, numpy as np, pickle, matplotlib.pyplot as plt

# Connect to ClearML
task = Task.init(project_name="VisiblePipeline", task_name="step_train")
logger = task.get_logger()

# Load dataset
dataset = Dataset.get(dataset_name="plant_processed_data_split", dataset_project="VisiblePipeline", only_completed=True)
dataset_path = dataset.get_local_copy()

# Define paths
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "valid")
test_dir = os.path.join(dataset_path, "test")

# Load image datasets
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 5

train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True)
val_ds   = tf.keras.preprocessing.image_dataset_from_directory(val_dir,   image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False)
test_ds  = tf.keras.preprocessing.image_dataset_from_directory(test_dir,  image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_ds.class_names
num_classes = len(class_names)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)

# Build hybrid model
inp = Input(shape=(160, 160, 3))
mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inp)
mobilenet.trainable = True

densenet = DenseNet121(include_top=False, weights='imagenet', input_tensor=inp)
densenet.trainable = False

m_out = layers.GlobalAveragePooling2D()(mobilenet.output)
d_out = layers.GlobalAveragePooling2D()(densenet.output)
merged = layers.Concatenate()([m_out, d_out])
x = layers.Dense(256, activation='relu')(merged)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inp, outputs=out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
os.makedirs("outputs", exist_ok=True)
checkpoint_cb = ModelCheckpoint("outputs/best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[checkpoint_cb])

# Save models and history
model.save("outputs/final_model.h5")
with open("outputs/train_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

task.upload_artifact("final_model", artifact_object="outputs/final_model.h5")
task.upload_artifact("best_model", artifact_object="outputs/best_model.h5")
task.upload_artifact("training_history", artifact_object="outputs/train_history.pkl")

# Evaluation
y_true = np.concatenate([y.numpy() for x, y in test_ds])
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Report
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
logger.report_text("📊 Classification Report:\n" + report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
task.upload_artifact("confusion_matrix", artifact_object="outputs/confusion_matrix.png")

# Accuracy/loss plots
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
print("✅ Training complete and artifacts uploaded.")
