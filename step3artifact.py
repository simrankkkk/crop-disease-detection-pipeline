from clearml import Task
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 🚀 Connect to ClearML Task
task = Task.init(project_name="PlantPipeline", task_name="step3 - hybrid model training", task_type="training")

# 🔗 Fetch the artifact uploaded by step2
step2_task = Task.get_task(project_name="PlantPipeline", task_name="step2 - preprocessing")
dataset_path = step2_task.artifacts["preprocessed_dataset"].get_local_copy()

train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "valid")

# ✅ Image generators (rescaling done here)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ✅ Build hybrid model
input_tensor = Input(shape=(224, 224, 3))

# MobileNetV2 branch
mnet = MobileNetV2(include_top=False, weights="imagenet", input_tensor=input_tensor)
for layer in mnet.layers:
    layer.trainable = False
x1 = GlobalAveragePooling2D()(mnet.output)

# DenseNet121 branch
dnet = DenseNet121(include_top=False, weights="imagenet", input_tensor=input_tensor)
for layer in dnet.layers:
    layer.trainable = False
x2 = GlobalAveragePooling2D()(dnet.output)

# Combine
x = Concatenate()([x1, x2])
x = Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=input_tensor, outputs=output)

# ✅ Compile and train
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_gen, validation_data=val_gen, epochs=5)

# ✅ Save and upload model
os.makedirs("model", exist_ok=True)
model_path = os.path.join("model", "hybrid_model.h5")
model.save(model_path)
task.upload_artifact("hybrid_model", model_path)

print("✅ Step 3 completed. Model saved and uploaded:", model_path)
