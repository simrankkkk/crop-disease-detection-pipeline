from clearml import Task
from clearml.automation import UniformParameterRange, HyperParameterOptimizer

# ✅ Init task
task = Task.init(project_name="VisiblePipeline", task_name="stage_hpo")

# ✅ Base task (completed training job)
base_task_id = "a9b6d3291e6846c1800476aabb057b06"

# ✅ Use basic dict-style hyperparameter config
param_ranges = {
    "General/learning_rate": UniformParameterRange(0.0001, 0.01),
    "General/dropout": UniformParameterRange(0.3, 0.5),
    "General/dense_units": UniformParameterRange(128, 512),
}

# ✅ Set up the optimizer
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
    save_top_k_tasks_only=1,
    execution_queue="default",
    clone_base_task_name_suffix="HPO_Trial"
)

# ✅ Print best result
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

# ✅ Run it
optimizer.set_report_period(1)
optimizer.start()
print_best_result(optimizer)
