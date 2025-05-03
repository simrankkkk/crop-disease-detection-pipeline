# step3split.py — Training with ClearML using train/valid/test
from clearml import Task, Dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from pathlib import Path
import os

# Initialize ClearML task
task = Task.init(project_name="PlantPipeline", task_name="Step 3 - Train Hybrid Model", task_type=Task.TaskTypes.training)

# Get processed dataset
dataset = Dataset.get(dataset_project="PlantPipeline", dataset_name="plant_processed_data_split")
path = Path(dataset.get_local_copy())

# Define paths
train_dir = path / "train"
valid_dir = path / "valid"
test_dir = path / "test"

# Data generators
gen = ImageDataGenerator(rescale=1./255)
train_gen = gen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")
val_gen = gen.flow_from_directory(valid_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")
test_gen = gen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode="categorical", shuffle=False)

# Model input
inp = Input(shape=(224, 224, 3))
m1 = MobileNetV2(include_top=False, input_tensor=inp, weights="imagenet")
m2 = DenseNet121(include_top=False, input_tensor=inp, weights="imagenet")
for layer in m1.layers + m2.layers:
    layer.trainable = False

# Fusion
x = Concatenate()([
    GlobalAveragePooling2D()(m1.output),
    GlobalAveragePooling2D()(m2.output)
])
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=inp, outputs=out)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Evaluate on test set
loss, acc = model.evaluate(test_gen)
print(f"✅ Test accuracy: {acc:.4f}, Test loss: {loss:.4f}")

# Save model
os.makedirs("model_output", exist_ok=True)
model_path = os.path.join("model_output", "hybrid_model.h5")
model.save(model_path)
Task.current_task().upload_artifact(name="hybrid_model", artifact_object=model_path)
task.close()
