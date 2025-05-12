from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformParameterRange

# 1) Initialize the HPO controller task
task = Task.init(project_name="VisiblePipeline", task_name="stage_hpo")

# 2) This is your completed training task to clone
base_task_id = "a9b6d3291e6846c1800476aabb057b06"

# 3) Define the search space as a LIST of UniformParameterRange objects
param_ranges = [
    UniformParameterRange(
        name="General/learning_rate",
        min_value=0.0001,
        max_value=0.01
    ),
    UniformParameterRange(
        name="General/dropout",
        min_value=0.3,
        max_value=0.5
    ),
    UniformParameterRange(
        name="General/dense_units",
        min_value=128,
        max_value=512
    ),
]

# 4) Create and configure the optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=param_ranges,
    objective_metric_title="accuracy",
    objective_metric_series="val_accuracy",
    objective_metric_sign="max",        # maximize validation accuracy
    max_iteration=8,                    # total trials
    execution_queue="default",
    save_top_k_tasks_only=1,
    clone_base_task_name_suffix="HPO_Trial"
)

# 5) Utility to print out the winning trial
def print_best_result(hpo):
    best = hpo.get_best_task()
    if not best:
        print("❌ No best task found.")
        return
    print(f"\n🏆 BEST TASK ID: {best.id}")
    val_acc = best.get_last_scalar_metrics().get("accuracy", {}) \
                   .get("val_accuracy", {}).get("value", "N/A")
    print(f"📈 Best Validation Accuracy: {val_acc}")
    print("📊 Best Hyperparameters:")
    for k, v in best.get_parameters().items():
        if any(p in k for p in ("learning_rate", "dropout", "dense_units")):
            print(f"   • {k}: {v}")

# 6) Run HPO and show the winner
optimizer.set_report_period(1)
optimizer.start()
print_best_result(optimizer)
