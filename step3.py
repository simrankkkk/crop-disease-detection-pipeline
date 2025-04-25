# step3.py - Train ensemble model with ClearML tracking (MobileNetV2 + DenseNet121)

from clearml import Task
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Init ClearML Task
task = Task.init(project_name="PlantPipeline", task_name="Train Ensemble Model")

# ✅ Paths from ClearML-managed dataset
dataset_path = task.connect_configuration()  # Makes sure dataset path is tracked
train_dir = os.path.join(dataset_path["dataset_path"], "train")
valid_dir = os.path.join(dataset_path["dataset_path"], "valid")

# ✅ Image Parameters
img_height = 224
img_width = 224
batch_size = 32
num_classes = len(os.listdir(train_dir))

# ✅ Image Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

valid_gen = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# ✅ MobileNetV2 Branch
base_model_1 = MobileNetV2(include_top=False, input_shape=(img_height, img_width, 3), weights='imagenet')
x1 = GlobalAveragePooling2D()(base_model_1.output)

# ✅ DenseNet121 Branch
base_model_2 = DenseNet121(include_top=False, input_shape=(img_height, img_width, 3), weights='imagenet')
x2 = GlobalAveragePooling2D()(base_model_2.output)

# ✅ Concatenate outputs
combined = Concatenate()([x1, x2])
combined = Dropout(0.5)(combined)
output = Dense(num_classes, activation='softmax')(combined)

# ✅ Final Model
model = Model(inputs=[base_model_1.input], outputs=output)

# ✅ Freeze base layers
for layer in base_model_1.layers:
    layer.trainable = False
for layer in base_model_2.layers:
    layer.trainable = False

# ✅ Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=valid_gen
)

# ✅ Save model
output_dir = "./output_model"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "ensemble_model.h5")
model.save(model_path)
print(f"✅ Model saved to: {model_path}")
