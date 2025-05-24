# 🌿 Plant Disease Classification

## 📌 Project Overview
This project aims to classify plant diseases using a **pretrained deep learning model**. The model is fine-tuned to recognize **38 different plant diseases** based on leaf images. The project uses **TensorFlow 2** and **TensorFlow Hub** for loading and optimizing a pretrained model.

## 📂 Repository Structure
```
📦 Plant-Disease-Classification
 ├── ais.ipynb   # Jupyter Notebook containing the code for loading, fine-tuning, and evaluating the model
 ├── README.md   # This documentation file
```

## 🚀 How to Use
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/plant-disease-classification.git
cd plant-disease-classification
```

### 2️⃣ Open Jupyter Notebook
Make sure you have **Jupyter Notebook** installed. Then run:
```bash
jupyter notebook
```
Open `ais.ipynb` in Jupyter and run the cells sequentially.

### 3️⃣ Install Dependencies
If required, install the necessary Python packages:
```bash
pip install tensorflow tensorflow-hub matplotlib numpy pandas scikit-learn
```

### 4️⃣ Run the Model
Execute all cells in `ais.ipynb` to:
- Load the pretrained model
- Fine-tune it on your dataset
- Evaluate its performance with classification metrics and visualizations

## 📊 Evaluation Metrics & Visualizations
- **Confusion Matrix** to analyze misclassifications
- **Classification Report** (Precision, Recall, F1-score)
- **Accuracy Graphs** to track training and validation performance

## 🔹 Future Improvements
- Adding **data augmentation** for better generalization
- Implementing **hyperparameter tuning** for optimal performance
- Trying alternative models for comparison


# 🧠 Plant Disease Classification Project with ClearML Integration

This project demonstrates how to build and manage a machine learning pipeline using ClearML for a real-world AIS (AI Solution) application. It includes dataset tracking, preprocessing, and model training using a pretrained model, all organized into ClearML tasks and executed as a pipeline.

---

## 📁 Project Structure

```
AIS-ClearML/
├── ais.ipynb                       # Main development notebook
├── ais.py                          # Converted Python script from notebook
├── step1_dataset_artifact.py      # Step 1: Upload dataset to ClearML
├── step2_data_preprocessing.py    # Step 2: Preprocess the dataset
├── step3_train_model.py           # Step 3: Train model using ClearML
├── pipeline_from_tasks.py         # Define and run ClearML pipeline
├── model_artifacts/               # Optional: Saved model files
├── work_dataset/                  # Optional: Local dataset files
├── README.md                      # Project documentation
```

---

## 🚀 Project Goals

- Use a pretrained model and fine-tune it on a custom dataset.
- Integrate ClearML to:
  - Track datasets and experiments
  - Automate pipelines
  - Log metrics and model artifacts

---

## ⚙️ Setup Instructions

### 1. 📦 Install Dependencies

```bash
pip install clearml
```

---

### 2. 🔑 Configure ClearML

Run:

```bash
clearml-init
```

Fill in:

- ClearML Server (use `https://app.clear.ml` for hosted version)
- Access Key & Secret Key from your ClearML dashboard
- Project name: `AIS Project`

---

### 3. 🛠 Register Base Tasks

Run the scripts one by one to register each ClearML task:

```bash
python step1_dataset_artifact.py     # Upload dataset to ClearML
python step2_data_preprocessing.py   # Preprocess dataset
python step3_train_model.py          # Train pretrained model
```

---

### 4. 🌀 Run ClearML Agent

Create and start a ClearML agent to process pipeline tasks:

```bash
clearml-agent daemon --queue pipeline --detached
```

---

### 5. 🧪 Run the Pipeline

Use the predefined task-based pipeline controller:

```bash
python pipeline_from_tasks.py
```

The pipeline will execute the 3 steps in order:
- Upload Dataset → Preprocess Data → Train Model

Check your ClearML Dashboard to monitor task progress and view logs, metrics, and results.

---

## 📊 Features

✅ Task-based pipeline with `PipelineController.add_step(...)`  
✅ Dataset tracking with ClearML  
✅ Pretrained model fine-tuning and evaluation  
✅ Centralized experiment tracking (loss, accuracy, confusion matrix, etc.)  
✅ ClearML-hosted and self-hosted compatibility

---

## 🧪 Example Use Case

This AIS project uses a pretrained model to classify image data. The pipeline automates:
- Dataset upload (with metadata)
- Image resizing and preprocessing
- Transfer learning using a pretrained CNN model

---

## 💡 Future Improvements

- Add function-based ClearML pipeline using `add_function_step(...)`
- Deploy model as a ClearML-served inference endpoint
- Integrate model evaluation and auto-retraining based on thresholds

---

## 📧 Contact

For questions or collaboration: [Your Email or GitHub Profile]

---

## 🔗 Resources

- [ClearML Documentation](https://clear.ml/docs/)
- [ClearML Hosted Dashboard](https://app.clear.ml)



## 🤝 Contributions
Feel free to **fork**, **create issues**, or **submit pull requests** to improve the project!

## 📜 License
This project is open-source and available under the **MIT License**.

