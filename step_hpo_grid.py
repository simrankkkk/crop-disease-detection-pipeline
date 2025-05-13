from clearml import Task, TaskTypes
from clearml.automation import UniformParameterRange, HyperParameterOptimizer
from clearml.backend_interface.task.task import Task as BackendTask
import time, json

# ✅ Start ClearML task
task = Task.init(project_name="VisiblePipeline", task_name="step_hpo_grid", task_type=TaskTypes.optimizer)
params = task.get_parameters_as_dict()
base_task_id = params.get("Args/base_task_id") or "950c9256da504bf1ac395253816321a6"
print(f"🔗 Connected to ClearML\n📌 Using base_task_id = {base_task_id}")

# ✅ Define search space (must match General)
search_space = [
    UniformParameterRange("General/learning_rate", 0.0005, 0.01, 0.002),
    UniformParameterRange("General/dropout", 0.3, 0.6, 0.1)
]

# ✅ Configure optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=search_space,
    objective_metric_title="accuracy",
    objective_metric_series="val_accuracy",
    objective_metric_sign="max",
    max_number_of_concurrent_tasks=5,
    total_max_jobs=5,
    min_iteration_per_job=1,
    always_create_task=True,
    save_top_k_tasks_only=1
)

# ✅ Launch HPO
print("🚀 Launching HPO optimization...")
optimizer.start()

# ✅ Get trial task IDs
trial_task_ids = optimizer._task.executed_children_tasks_ids
print(f"📋 Trial task IDs: {trial_task_ids}")

# ✅ Wait for all to finish
print("⏳ Waiting for trial tasks to finish...")
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

# ✅ Find the best task manually
best_task = None
best_val_acc = -1.0
for tid in trial_task_ids:
    try:
        t = BackendTask.get_task(task_id=tid)
        metrics = t.get_last_scalar_metrics()
        val_acc = metrics.get("accuracy", {}).get("val_accuracy", {}).get("value", None)
        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_task = t
    except Exception:
        continue

if not best_task:
    print("❌ Could not find a best task.")
    task.close()
    exit()

# ✅ Save best params
best_params = best_task.get_parameters()
with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)
task.upload_artifact("best_params", artifact_object="best_params.json")

# ✅ Report
print("\n🏆 Best Hyperparameters:")
for k, v in best_params.items():
    if "learning_rate" in k or "dropout" in k:
        print(f"🔧 {k} = {v}")
print(f"📈 Best Validation Accuracy: {best_val_acc:.4f}")

task.close()
print("✅ HPO complete. All trials processed. Best model ready.")
