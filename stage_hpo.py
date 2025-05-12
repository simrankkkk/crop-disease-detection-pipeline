from clearml import Task
from clearml.automation import UniformParameterRange, HyperParameterOptimizer

# ✅ Initialize ClearML HPO Task
task = Task.init(project_name="VisiblePipeline", task_name="stage_hpo")

# ✅ Base task to clone for HPO trials
base_task_id = "a9b6d3291e6846c1800476aabb057b06"  # completed stage_train_hpo_3.py

# ✅ Define hyperparameter search space using correct argument names
param_ranges = {
    "General/learning_rate": UniformParameterRange(min_value=0.0001, max_value=0.01),
    "General/dropout": UniformParameterRange(min_value=0.3, max_value=0.5),
    "General/dense_units": UniformParameterRange(min_value=128, max_value=512),
}

# ✅ Create the HPO optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=param_ranges,
    objective_metric_title="accuracy",
    objective_metric_series="val_accuracy",
    objective_metric_sign="max",
    max_iteration=8,
    total_max_jobs=8,
    min_iteration_per_job=1,
    max_iteration_per_job=1,
    compute_time_limit=None,
    save_top_k_tasks_only=1,
    execution_queue="default",
    clone_base_task_name_suffix="HPO_Trial"
)

# ✅ Utility function to print the best result
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
    for key, value in best_task.get_parameters().items():
        if any(h in key for h in ["learning_rate", "dropout", "dense_units"]):
            print(f"   - {key}: {value}")

# ✅ Start HPO process
optimizer.set_report_period(1)
optimizer.start()

# ✅ Show summary of best trial
print_best_result(optimizer)
