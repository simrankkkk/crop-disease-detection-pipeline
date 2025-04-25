from clearml import Task
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ✅ Connect to ClearML
task = Task.init(project_name="PlantPipeline", task_name="3artifact-final", task_type=Task.TaskTypes.training)

# ✅ Load artifact folder from Step 2
step2_task = Task.get_task(task_id="32475789c3c24b8c9d4966ceefef130a")
dataset_path = step2_task.artifacts["preprocessed_dataset"].get_local_copy()

# ✅ Use the actual directory (already unzipped)
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "valid")

# ✅ Image parameters
img_size = (224, 224)
batch_size = 32

# ✅ Generators
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_gen = datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# ✅ Base models
base1 = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base2 = DenseNet121(weights='imagenet', include_top=False, input_shape=img_size + (3,))

for layer in base1.layers + base2.layers:
    layer.trainable = False

# ✅ Concatenate outputs
x1 = GlobalAveragePooling2D()(base1.output)
x2 = GlobalAveragePooling2D()(base2.output)
x = Concatenate()([x1, x2])
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base1.input, outputs=output)

# ✅ Compile & train
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=5)

# ✅ Save & upload model
os.makedirs("model_output", exist_ok=True)
model_path = os.path.join("model_output", "ensemble_model.h5")
model.save(model_path)
task.upload_artifact("trained_model", artifact_object=model_path)

print("✅ Step 3 training complete.")
