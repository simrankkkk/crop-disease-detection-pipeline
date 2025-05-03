# step3split.py — Train model from dataset ID input
import sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from clearml import Dataset, Task
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# Get dataset ID
dataset_id = sys.argv[1] if len(sys.argv) > 1 else None
assert dataset_id, "❌ Dataset ID must be provided"

# Task init
task = Task.init(project_name="VisiblePipeline", task_name="step_to_train")
logger = task.get_logger()

dataset = Dataset.get(dataset_id=dataset_id)
base_path = dataset.get_local_copy()
train_dir, val_dir, test_dir = [os.path.join(base_path, x) for x in ["train", "valid", "test"]]

# Dataset loading
img_size, batch_size = (160, 160), 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, image_size=img_size, batch_size=batch_size, shuffle=True)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir, image_size=img_size, batch_size=batch_size)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, image_size=img_size, batch_size=batch_size)
class_names = train_ds.class_names

# Prefetch
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Model
inp = Input(shape=(160, 160, 3))
mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inp)
mobilenet.trainable = True
densenet = DenseNet121(include_top=False, weights='imagenet', input_tensor=inp)
densenet.trainable = False
x = layers.Concatenate()([layers.GlobalAveragePooling2D()(mobilenet.output), layers.GlobalAveragePooling2D()(densenet.output)])
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=inp, outputs=out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
os.makedirs("outputs", exist_ok=True)
checkpoint = ModelCheckpoint("outputs/best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max", verbose=1)

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[checkpoint])
model.save("outputs/final_model.h5")
with open("outputs/train_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Upload artifacts
task.upload_artifact("final_model", artifact_object="outputs/final_model.h5")
task.upload_artifact("best_model", artifact_object="outputs/best_model.h5")
task.upload_artifact("training_history", artifact_object="outputs/train_history.pkl")

# Evaluate
y_true = np.concatenate([y.numpy() for x, y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
logger.report_text(report)
cm = confusion_matrix(y_true, y_pred)
import seaborn as sns
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
task.upload_artifact("confusion_matrix", artifact_object="outputs/confu
