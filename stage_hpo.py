from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformParameterRange

# 1) Initialize the HPO controller task
task = Task.init(project_name="VisiblePipeline", task_name="stage_hpo")

# 2) The completed base training task to clone
base_task_id = "a9b6d3291e6846c1800476aabb057b06"

# 3) Define hyperparameter ranges as a LIST of UniformParameterRange objects
hyper_params = [
    UniformParameterRange(name="General/learning_rate", min_value=0.0001, max_value=0.01),
    UniformParameterRange(name="General/dropout",         min_value=0.3,    max_value=0.5),
    UniformParameterRange(name="General/dense_units",    min_value=128,    max_value=512),
]

# 4) Configure the optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=hyper_params,
    objective_metric_title="accuracy",
    objective_metric_series="val_accuracy",
    objective_metric_sign="max",
    max_iteration=8,
    total_max_jobs=8,
    min_iteration_per_job=1,
    max_iteration_per_job=1,
    save_top_k_tasks_only=1,
    execution_queue="default",
    clone_base_task_name_suffix="HPO_Trial"
)

# 5) Run HPO
optimizer.set_report_period(1)
optimizer.start()

# 6) Fetch the best trial via the classmethod
best_tasks = HyperParameterOptimizer.get_optimizer_top_experiments(
    objective_metric_title="accuracy",
    objective_metric_series="val_accuracy",
    objective_metric_sign="max",
    optimizer_task_id=task.id,
    top_k=1
)
if not best_tasks:
    print("❌ No best task found.")
    exit(1)

best = best_tasks[0]
print(f"\n🏆 BEST TASK ID: {best.id}")

# 7) Print its validation accuracy
metrics = best.get_last_scalar_metrics()
val_acc = metrics.get("accuracy", {}).get("val_accuracy", {}).get("value", "N/A")
print(f"📈 Best Validation Accuracy: {val_acc}")

# 8) And print out the winning hyperparameters
params = best.get_parameters()
print("📊 Best Hyperparameters:")
print(f"   • General/learning_rate: {params.get('General/learning_rate')}")
print(f"   • General/dropout:         {params.get('General/dropout')}")
print(f"   • General/dense_units:      {params.get('General/dense_units')}")
