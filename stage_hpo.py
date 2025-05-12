from clearml import Task, Dataset
from clearml.automation.optuna import HyperParameterOptimizer
from clearml.automation import UniformParameterRange
import json

# 🚀 Initialize ClearML task
task = Task.init(project_name="VisiblePipeline", task_name="step_hpo", task_type=Task.TaskTypes.optimizer)
print("🔗 HPO Task Connected")

# 📥 Load dataset (for logging consistency)
dataset = Dataset.get(dataset_name="plant_processed_data_split", dataset_project="VisiblePipeline", only_completed=True)
dataset_path = dataset.get_local_copy()
print("📂 Dataset Path:", dataset_path)

# ⚙️ Set up HPO
optimizer = HyperParameterOptimizer(
    base_task_id="681dd8e8c082451fb4a1c9d44e5e83e2",  # 👈 your actual step_train Task ID
    hyper_parameters={
        "Args/learning_rate": UniformParameterRange(0.0001, 0.01),
        "Args/dropout": UniformParameterRange(0.3, 0.5)
    },
    objective_metric_title="accuracy",
    objective_metric_series="val_accuracy",
    objective_metric_sign="max",
    max_number_of_concurrent_tasks=2,
    total_max_jobs=10,
    always_create_task=True,
    save_top_k_tasks_only=1
)

# 🧪 Run search
best_task = optimizer.run()

# 🏆 Save best parameters
best_params = best_task.get_parameters_as_dict()
filtered_params = {
    "learning_rate": float(best_params.get("Args/learning_rate", 0.001)),
    "dropout": float(best_params.get("Args/dropout", 0.4))
}
with open("best_params.json", "w") as f:
    json.dump(filtered_params, f)

task.upload_artifact("best_params", artifact_object="best_params.json")
task.close()
