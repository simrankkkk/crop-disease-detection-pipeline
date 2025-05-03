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
import os, numpy as np, pickle, matplotlib.pyplot as plt, seaborn as sns

# Connect to ClearML
task = Task.init(project_name="VisiblePipeline", task_name="step_train_fast")
logger = task.get_logger()

# Load dataset
dataset = Dataset.get(dataset_name="plant_processed_data_split", dataset_project="VisiblePipeline", only_completed=True)
dataset_path = dataset.get_local_copy()

train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "valid")
test_dir = os.path.join(dataset_path, "test")

# Load only 2% of data using validation_split
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, validation_split=0.98, subset="validation", seed=123
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, validation_split=0.98, subset="validation", seed=123
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

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

# Callbacks
os.makedirs("outputs", exist_ok=True)
checkpoint_cb = ModelCheckpoint("outputs/best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")

# Train only for 1 epoch
history = model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=[checkpoint_cb])

# Save model
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

# Classification report
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

task.close()
print("✅ Fast training complete and artifacts uploaded.")
