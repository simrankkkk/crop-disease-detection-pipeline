# clearml_requirements: tensorflow, numpy, matplotlib, scikit-learn, clearml, opencv-python, pillow

from clearml import Task, Dataset
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# ✅ Start ClearML Task
task = Task.init(project_name="plantdataset", task_name="Step 3 - TF Hybrid Model (No Torch)")

# ✅ Load dataset from ClearML
dataset = Dataset.get(dataset_name="New Augmented Plant Disease Dataset")
dataset_path = dataset.get_local_copy()

# ✅ Image Dimensions & Setup
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3

train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    valid_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

n_classes = train_gen.num_classes

# ✅ Feature Extractors
input_layer = Input(shape=(224, 224, 3))

mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_layer)
densenet = DenseNet121(weights='imagenet', include_top=False, input_tensor=input_layer)

for layer in mobilenet.layers:
    layer.trainable = False

for layer in densenet.layers:
    layer.trainable = False

# ✅ Extract features & concatenate
mobilenet_out = GlobalAveragePooling2D()(mobilenet.output)
densenet_out = GlobalAveragePooling2D()(densenet.output)
merged = Concatenate()([mobilenet_out, densenet_out])

# ✅ Classifier
x = Dense(512, activation='relu')(merged)
x = Dropout(0.4)(x)
output = Dense(n_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# ✅ Evaluate
val_gen.reset()
y_true = val_gen.classes
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_true, y_pred_classes, target_names=list(val_gen.class_indices.keys()))
print("✅ Classification Report:\n", report)

# ✅ Save model
model.save("hybrid_model_tf.h5")
task.upload_artifact("hybrid_model_tf", "hybrid_model_tf.h5")
