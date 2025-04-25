from clearml import Task
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import zipfile
import os
from pathlib import Path

# 🚀 ClearML task
task = Task.init(project_name="PlantPipeline", task_name="step3 - hybrid model training")

# 🧠 Load Step 2 artifact (zip containing preprocessed train/valid images)
step2_task = Task.get_task(task_id="32475789c3c24b8c9d4966ceefef130a")  # ✅ Replace with your Step 2 Task ID
dataset_path = step2_task.artifacts["preprocessed_dataset"].get_local_copy()

# 🗂️ Unzip the processed dataset if needed
if dataset_path.endswith(".zip"):
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall("unzipped_data")
    data_dir = "unzipped_data"
else:
    data_dir = dataset_path

train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "valid")

# 🔁 Data generators
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical")
val_gen = datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical")

# 🧠 Base models
input_tensor = Input(shape=(224, 224, 3))

mobilenet = MobileNetV2(include_top=False, input_tensor=input_tensor, weights="imagenet")
densenet = DenseNet121(include_top=False, input_tensor=input_tensor, weights="imagenet")

for layer in mobilenet.layers:
    layer.trainable = False
for layer in densenet.layers:
    layer.trainable = False

# 🔀 Concatenate features
avg_pool_1 = GlobalAveragePooling2D()(mobilenet.output)
avg_pool_2 = GlobalAveragePooling2D()(densenet.output)

merged = Concatenate()([avg_pool_1, avg_pool_2])
fc = Dense(256, activation="relu")(merged)
fc = Dropout(0.3)(fc)
output = Dense(train_gen.num_classes, activation="softmax")(fc)

# 🏗️ Final model
model = Model(inputs=input_tensor, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# 🎯 Train
model.fit(train_gen, validation_data=val_gen, epochs=5)

# 💾 Save model
Path("model_output").mkdir(exist_ok=True)
model_path = os.path.join("model_output", "hybrid_model.h5")
model.save(model_path)

# 📦 Upload model to ClearML
task.upload_artifact(name="hybrid_model", artifact_object=model_path)

print("✅ Step 3 completed. Model saved and uploaded to ClearML.")
task.close()
