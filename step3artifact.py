from clearml import Task, Dataset
import zipfile
import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ✅ Connect to ClearML and initialize task
task = Task.init(project_name="PlantPipeline", task_name="step3-final", task_type=Task.TaskTypes.training)

# ✅ Load artifact from Step 2 task
step2_task = Task.get_task(task_id="32475789c3c24b8c9d4966ceefef130a")
dataset_path = step2_task.artifacts["preprocessed_dataset"].get_local_copy()

# ✅ Unzip the artifact
unzipped_dir = os.path.join(dataset_path, "unzipped")
os.makedirs(unzipped_dir, exist_ok=True)
with zipfile.ZipFile(os.path.join(dataset_path, "preprocessed_data.zip"), 'r') as zip_ref:
    zip_ref.extractall(unzipped_dir)

train_dir = os.path.join(unzipped_dir, "train")
val_dir = os.path.join(unzipped_dir, "valid")

# ✅ Prepare data
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_gen = datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# ✅ Load base models
base1 = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base2 = DenseNet121(weights='imagenet', include_top=False, input_shape=img_size + (3,))

for layer in base1.layers + base2.layers:
    layer.trainable = False

# ✅ Concatenate features
x1 = GlobalAveragePooling2D()(base1.output)
x2 = GlobalAveragePooling2D()(base2.output)
x = Concatenate()([x1, x2])
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=[base1.input], outputs=output)  # Use base1.input since both have same input shape

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train model
model.fit(train_gen, validation_data=val_gen, epochs=5)

# ✅ Save model
os.makedirs("model_output", exist_ok=True)
model_path = os.path.join("model_output", "ensemble_model.h5")
model.save(model_path)

# ✅ Upload model as artifact
task.upload_artifact("trained_model", artifact_object=model_path)

print("✅ Step 3 completed successfully.")
