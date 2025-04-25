from clearml import Task
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.applications import MobileNetV2, DenseNet121
import pickle

def step3():
    task = Task.init(project_name="PlantPipeline", task_name="step3 - Hybrid CNN Model Training")

    # Load preprocessed data
    preprocessed_dir = "preprocessed"
    X_train = np.load(os.path.join(preprocessed_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(preprocessed_dir, "X_val.npy"))
    y_train = np.load(os.path.join(preprocessed_dir, "y_train.npy"))
    y_val   = np.load(os.path.join(preprocessed_dir, "y_val.npy"))

    # Load label encoder to get number of classes
    with open(os.path.join(preprocessed_dir, "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    num_classes = len(encoder.classes_)

    # Build hybrid model: MobileNetV2 + DenseNet121
    input_tensor = Input(shape=(128, 128, 3))

    base_mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)
    base_densenet  = DenseNet121(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # Freeze base models
    for layer in base_mobilenet.layers:
        layer.trainable = False
    for layer in base_densenet.layers:
        layer.trainable = False

    # Combine outputs
    out_mobilenet = GlobalAveragePooling2D()(base_mobilenet.output)
    out_densenet  = GlobalAveragePooling2D()(base_densenet.output)
    merged        = Concatenate()([out_mobilenet, out_densenet])

    # Classification head
    x = Dense(512, activation='relu')(merged)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    # Save model
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "hybrid_model.h5")
    model.save(model_path)

    # Upload model artifact
    task.upload_artifact("hybrid_model", model_path)
    print("✅ Hybrid model training complete. Saved to:", model_path)

    return model_path

if __name__ == "__main__":
    step3()
