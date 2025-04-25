task = Task.init(project_name="PlantPipeline", task_name="step 3: Train Model")

# step3_train_model.py
import os
from clearml import Task, Dataset, Model

if __name__ == '__main__':
    task = Task.init(
        project_name='PlantPipeline',
        task_name='Step3-TrainModel',
        task_type=Task.TaskTypes.training
    )

    # 1) Get processed images folder (from previous step)
    processed_folder = task.get_output('Step2-PreprocessData')
    if not processed_folder:
        # if running standalone, fetch from artifact
        processed_folder = task.artifacts['preprocessed_data'].get_local_copy()

    # 2) Build your generators
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2, DenseNet121
    from tensorflow.keras import layers, Input
    from tensorflow.keras.models import Model

    IMG_SIZE = (160,160)
    BATCH    = 32
    datagen  = ImageDataGenerator()

    train_gen = datagen.flow_from_directory(
        os.path.join(processed_folder, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode='categorical'
    )
    valid_gen = datagen.flow_from_directory(
        os.path.join(processed_folder, 'valid'),
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode='categorical'
    )

    # 3) Build the hybrid model
    input_layer = Input(shape=(160,160,3))
    mobi_base = MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_layer)
    dnet_base = DenseNet121(include_top=False, weights='imagenet', input_tensor=input_layer)
    for l in dnet_base.layers: l.trainable = False

    x1 = layers.GlobalAveragePooling2D()(mobi_base.output)
    x2 = layers.GlobalAveragePooling2D()(dnet_base.output)
    x  = layers.Concatenate()([x1, x2])
    x  = layers.Dense(256, activation='relu')(x)
    x  = layers.Dropout(0.4)(x)
    out= layers.Dense(train_gen.num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 4) Train
    history = model.fit(train_gen, validation_data=valid_gen, epochs=10)

    # 5) Save & upload
    os.makedirs('myModel', exist_ok=True)
    model_path = os.path.join('myModel','hybrid_model.h5')
    model.save(model_path)
    Model.upload(
        model_path=model_path,
        model_name='Hybrid-MobileNetV2-DenseNet121',
        model_project='PlantPipeline'
    )
    print(f"✅ Model trained & uploaded: {model_path}")
    return model_path
