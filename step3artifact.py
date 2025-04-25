from clearml import Task
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Concatenate, Input
from tensorflow.keras.models import Model

# ✅ Connect to ClearML
task = Task.init(project_name="PlantPipeline", task_name="step3 - Train Hybrid Model")
step2_task = Task.get_task(task_id="32475789c3c24b8c9d4966ceefe1f30a")  # ✅ Replace with actual Step 2 task ID

# ✅ Load preprocessed dataset artifact from Step 2
dataset_path = step2_task.artifacts["preprocessed_dataset"].get_local_copy()
train_dir = os.path.join(dataset_path, "train")
valid_dir = os.path.join(dataset_path, "valid")

# ✅ Image parameters
img_size = (224, 224)
batch_size = 32

# ✅ Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

valid_gen = valid_datagen.flow_from_directory(
    valid_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

# ✅ Define hybrid model
input_tensor = Input(shape=(224, 224, 3))
base1 = MobileNetV2(include_top=False, input_tensor=input_tensor, weights='imagenet')
base2 = DenseNet121(include_top=False, input_tensor=input_tensor, weights='imagenet')

for layer in base1.layers:
    layer.trainable = False
for layer in base2.layers:
    layer.trainable = False

x1 = GlobalAveragePooling2D()(base1.output)
x2 = GlobalAveragePooling2D()(base2.output)
merged = Concatenate()([x1, x2])
output = Dense(train_gen.num_classes, activation='softmax')(merged)

model = Model(inputs=input_tensor, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train model
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=10
)

# ✅ Save model
model.save("hybrid_model.h5")
task.upload_artifact("trained_model", artifact_object="hybrid_model.h5")
task.close()
