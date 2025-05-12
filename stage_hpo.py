from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformParameterRange
from clearml.automation.optuna import HyperParameter

# ✅ Init task
task = Task.init(project_name="VisiblePipeline", task_name="stage_hpo")

# ✅ Base training task (completed one)
base_task_id = "a9b6d3291e6846c1800476aabb057b06"

# ✅ Build correct hyperparameter definitions
param_ranges = [
    HyperParameter(
        name="General/learning_rate",
        type=HyperParameter.Type.Float,
        range=UniformParameterRange(min_value=0.0001, max_value=0.01)
    ),
    HyperParameter(
        name="General/dropout",
        type=HyperParameter.Type.Float,
        range=UniformParameterRange(min_value=0.3, max_value=0.5)
    ),
    HyperParameter(
        name="General/dense_units",
        type=HyperParameter.Type.Integer,
        range=UniformParameterRange(min_value=128, max_value=512)
    ),
]

# ✅ Setup the optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=param_ranges,  # now a list, not a dict!
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

# ✅ Print result
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
