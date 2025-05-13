from clearml import Task, TaskTypes
from clearml.automation import UniformParameterRange, HyperParameterOptimizer
import json

# ✅ Initialize ClearML HPO Task
task = Task.init(project_name="VisiblePipeline", task_name="step_hpo_grid", task_type=TaskTypes.optimizer)
print("🔗 Connected to ClearML for HPO Grid Search")

# ✅ Get or fallback to a manual base_task_id for testing
params = task.get_parameters_as_dict()
base_task_id = params.get("Args/base_task_id") or "134bd547929944de8ee9ee4765152e03"  # Your working step_train ID

print(f"📌 Using base_task_id = {base_task_id}")

# ✅ Define hyperparameter ranges
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=[
        UniformParameterRange("Args/learning_rate", 0.0005, 0.01, 0.002),
        UniformParameterRange("Args/dropout", 0.3, 0.6, 0.1)
    ],
    objective_metric_title="accuracy",
    objective_metric_series="val_accuracy",
    objective_metric_sign="max",
    max_number_of_concurrent_tasks=2,
    total_max_jobs=5,
    min_iteration_per_job=1,
    always_create_task=True,
    save_top_k_tasks_only=1
)

# ✅ Start HPO grid search
optimizer.start()

# ✅ Finish ClearML task
task.close()
print("✅ HPO grid search completed and task closed.")
