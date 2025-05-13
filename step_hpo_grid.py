from clearml import Task, TaskTypes
from clearml.automation import UniformParameterRange, HyperParameterOptimizer
import json

# ✅ Start ClearML HPO Task
task = Task.init(project_name="VisiblePipeline", task_name="step_hpo_grid", task_type=TaskTypes.optimizer)
print("🔗 Connected to ClearML for HPO Grid Search")

# ✅ Use your working step_train task as base
template_task_id = "681dd8e8c082451fb4a1c9d44e5e83e2"  # ← step_train (baseline)

# ✅ Setup ClearML's native HPO (non-Optuna)
optimizer = HyperParameterOptimizer(
    base_task_id=template_task_id,
    hyper_parameters={
        "Args/learning_rate": UniformParameterRange(0.0005, 0.01),
        "Args/dropout": UniformParameterRange(0.3, 0.6)
    },
    objective_metric_title="accuracy",
    objective_metric_series="val_accuracy",
    objective_metric_sign="max",
    max_number_of_concurrent_tasks=2,
    total_max_jobs=5,
    min_iteration_per_job=1,
    always_create_task=True,
    save_top_k_tasks_only=1
)

# ✅ Run HPO
best_task = optimizer.run()
print("🏁 HPO complete")

# ✅ Extract best parameters
best_params = best_task.get_parameters_as_dict()
filtered_params = {
    "learning_rate": float(best_params.get("Args/learning_rate", 0.001)),
    "dropout": float(best_params.get("Args/dropout", 0.4))
}

# ✅ Save and upload best parameters
with open("best_params.json", "w") as f:
    json.dump(filtered_params, f)

task.upload_artifact(name="best_params", artifact_object="best_params.json")
task.close()
print("✅ Best parameters saved successfully.")
