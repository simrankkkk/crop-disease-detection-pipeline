from clearml import Task, TaskTypes
from clearml.automation import UniformParameterRange, HyperParameterOptimizer
import json
import time

# ✅ Initialize HPO Task
task = Task.init(project_name="VisiblePipeline", task_name="step_hpo_grid", task_type=TaskTypes.optimizer)
print("🔗 Connected to ClearML for HPO Grid Search")

params = task.get_parameters_as_dict()
base_task_id = params.get("Args/base_task_id") or "950c9256da504bf1ac395253816321a6"
print(f"📌 Using base_task_id = {base_task_id}")

# ✅ Configure HPO
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=[
        UniformParameterRange("General/learning_rate", 0.0005, 0.01, 0.002),
        UniformParameterRange("General/dropout", 0.3, 0.6, 0.1)
    ],
    objective_metric_title="accuracy",
    objective_metric_series="val_accuracy",
    objective_metric_sign="max",
    max_number_of_concurrent_tasks=5,
    total_max_jobs=5,
    min_iteration_per_job=1,
    always_create_task=True,
    save_top_k_tasks_only=1
)

# ✅ Start HPO
print("🚀 Launching HPO optimization...")
best_task = optimizer.start()

# ✅ Get all trial task IDs from optimizer
print("⏳ Waiting for all trial tasks to complete...")
trial_task_ids = [child.id for child in optimizer._top_tasks.values() if child]
print(f"🔍 Monitoring {len(trial_task_ids)} trial tasks...")

# ✅ Wait for trials to complete
from clearml.backend_interface.task.task import Task as TaskObj
while True:
    pending = 0
    for tid in trial_task_ids:
        t = TaskObj.get_task(tid)
        if t.status not in ("completed", "failed", "closed"):
            pending += 1
    print(f"⏳ {pending} trial(s) still running...")
    if pending == 0:
        break
    time.sleep(15)

print("✅ All HPO trials completed.")

# ✅ Save and upload best params
best_params = best_task.get_parameters()
with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)
task.upload_artifact("best_params", artifact_object="best_params.json")

# ✅ Print best results
print("\n🏆 Best Hyperparameters:")
for k, v in best_params.items():
    if "learning_rate" in k or "dropout" in k:
        print(f"🔧 {k} = {v}")

best_metrics = best_task.get_last_scalar_metrics()
val_acc = best_metrics.get("accuracy", {}).get("val_accuracy", {}).get("value", None)
if val_acc is not None:
    print(f"📈 Best Validation Accuracy: {val_acc:.4f}")
else:
    print("⚠️ Best validation accuracy not found in scalar logs.")

task.close()
print("✅ HPO grid task completed, best_params.json uploaded.")
