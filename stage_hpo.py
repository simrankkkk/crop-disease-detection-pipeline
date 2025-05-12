from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformParameterRange

# ✅ Start task
task = Task.init(project_name="VisiblePipeline", task_name="stage_hpo")

# ✅ Base task to clone
base_task_id = "a9b6d3291e6846c1800476aabb057b06"

# ✅ Build as a list of tuples — this is the exact format ClearML expects
param_ranges = [
    ("General/learning_rate", UniformParameterRange(name="General/learning_rate", min_value=0.0001, max_value=0.01)),
    ("General/dropout", UniformParameterRange(name="General/dropout", min_value=0.3, max_value=0.5)),
    ("General/dense_units", UniformParameterRange(name="General/dense_units", min_value=128, max_value=512)),
]

# ✅ Convert to dict using to_dict manually
param_dict = {name: obj.to_dict() for name, obj in param_ranges}

# ✅ Set up the optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=param_dict,
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

# ✅ Best result logging
def print_best_result(hpo):
    best_task = hpo.get_best_task()
    if not best_task:
        print("❌ No best task found.")
        return
    print("\n🏆 BEST TASK ID:", best_task.id)
    val_acc = (
        best_task.get_last_scalar_metrics()
        .get("accuracy", {})
        .get("val_accuracy", {})
        .get("value", "N/A")
    )
    print(f"📈 Best Validation Accuracy: {val_acc}")
    print("📊 Best Hyperparameters:")
    for k, v in best_task.get_parameters().items():
        if any(h in k for h in ["learning_rate", "dropout", "dense_units"]):
            print(f"   - {k}: {v}")

# ✅ Run
optimizer.set_report_period(1)
optimizer.start()
print_best_result(optimizer)
