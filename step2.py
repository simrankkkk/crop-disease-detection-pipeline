# step2.py — Data preprocessing with rescaling, label encoding, and ClearML logging

from clearml import Task
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import pickle

# ✅ Initialize ClearML Task
task = Task.init(project_name="PlantPipeline", task_name="step2 - Data Preprocessing")

# ✅ Dataset already uploaded to ClearML, we retrieve it
from clearml import Dataset
dataset_path = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71").get_local_copy()

# ✅ Class labels
classes = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# ✅ Process images and labels
all_images = []
all_labels = []

image_dir = os.path.join(dataset_path, "New Plant Diseases Dataset(Augmented)")

for label in classes:
    class_path = os.path.join(image_dir, label)
    if not os.path.isdir(class_path):
        continue
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB").resize((128, 128))
            img_arr = np.array(img) / 255.0  # ✅ Normalize to [0,1]
            all_images.append(img_arr)
            all_labels.append(label)
        except:
            continue

# ✅ Encode labels
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(all_labels)

# ✅ Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    np.array(all_images), np.array(labels_encoded), test_size=0.2, random_state=42, stratify=labels_encoded
)

# ✅ Save preprocessed data
output_dir = "preprocessed"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "X_val.npy"), X_val)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "y_val.npy"), y_val)

with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)

print("✅ Preprocessing complete and data saved to:", output_dir)

# ✅ Log output dir to ClearML
task.upload_artifact("preprocessed", artifact_object=output_dir)

# ✅ Return path for next step
return output_dir
