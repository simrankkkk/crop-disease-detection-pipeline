from clearml import Task, TaskTypes
from clearml.automation import UniformParameterRange, HyperParameterOptimizer
import json
import time

# ✅ Initialize ClearML task
task = Task.init(project_name="VisiblePipeline", task_name="step_hpo_grid", task_type=TaskTypes.optimizer)
params = task.get_parameters_as_dict()
base_task_id = params.get("Args/base_task_id") or "950c9256da504bf1ac395253816321a6"
print(f"🔗 Connected to ClearML\n📌 Using base_task_id = {base_task_id}")

# ✅ Define HPO search space (MUST match your baseline `task.connect()`)
search_space = [
    UniformParameterRange("General/learning_rate", min_value=0.0005, max_value=0.01, step_size=0.002),
    UniformParameterRange("General/dropout", min_value=0.3, max_value=0.6, step_size=0.1)
]

# ✅ Configure the HPO optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=search_space,
    objective_metric_title="accuracy",
    objective_metric_series="val_accuracy",
    objective_metric_sign="max",
    max_number_of_concurrent_tasks=5,
    total_max_jobs=5,
    always_create_task=True,
    min_iteration_per_job=1,
    save_top_k_tasks_only=1
)

# ✅ Launch HPO
print("🚀 Launching HPO optimization...")
best_task = optimizer.start()
print("⏳ All trials launched, waiting for them to finish...")

# ✅ Wait for all trial tasks using stable public method
trial_task_ids = best_task.get_parameters().get("General/parent", [])
if not trial_task_ids:
    trial_task_ids = optimizer._task.executed_children_tasks_ids  # fallback
print(f"📋 Trial task IDs: {trial_task_ids}")

# ✅ Wait until all trials are completed
from clearml.backend_interface.task.task import Task as BackendTask
while True:
    running = 0
    for tid in trial_task_ids:
        try:
            t = BackendTask.get_task(task_id=tid)
            if t.status not in ("completed", "closed", "failed"):
                running += 1
        except Exception:
            continue
    print(f"🔄 {running} trial(s) still running...")
    if running == 0:
        break
    time.sleep(15)

print("✅ All trial tasks completed.")

# ✅ Save and upload best hyperparameters
best_params = best_task.get_parameters()
with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)
task.upload_artifact("best_params", artifact_object="best_params.json")

# ✅ Print best results
print("\n🏆 Best Hyperparameters:")
for k, v in best_params.items():
    if "learning_rate" in k or "dropout" in k:
        print(f"🔧 {k} = {v}")

# ✅ Print final val_accuracy
metrics = best_task.get_last_scalar_metrics()
val_acc = metrics.get("accuracy", {}).get("val_accuracy", {}).get("value", None)
if val_acc is not None:
    print(f"📈 Best Validation Accuracy: {val_acc:.4f}")
else:
    print("⚠️ val_accuracy not found in scalar logs.")

task.close()
print("✅ HPO stage done. Best parameters uploaded.")
