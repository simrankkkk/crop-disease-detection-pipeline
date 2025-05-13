from clearml import Task, TaskTypes
from clearml.automation import HyperParameterOptimizer, UniformParameterRange
import time
import json

# ✅ Initialize HPO Task
task = Task.init(project_name="VisiblePipeline", task_name="step_hpo_grid", task_type=TaskTypes.optimizer)
print("🔗 Connected to ClearML")

# ✅ Pull baseline task
params = task.get_parameters_as_dict()
base_task_id = params.get("Args/base_task_id") or "950c9256da504bf1ac395253816321a6"
print(f"📌 Using base_task_id = {base_task_id}")

# ✅ Define HPO search space (must match General section of baseline task)
search_space = [
    UniformParameterRange("General/learning_rate", 0.0005, 0.01, 0.002),
    UniformParameterRange("General/dropout", 0.3, 0.6, 0.1)
]

# ✅ Create optimizer
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

# ✅ Start optimization
print("🚀 Launching HPO optimization...")
optimizer.start()
print("⏳ HPO launched all trial clones")

# ✅ Wait for child trial tasks to complete
from clearml.backend_interface.task.task import Task as BackendTask
print("⏳ Waiting for trial tasks to complete...")

# Get child task IDs reliably
trial_tasks = BackendTask.get_tasks(task_filter={"parent": task.id})
trial_ids = [t.id for t in trial_tasks]

while True:
    running = 0
    for tid in trial_ids:
        try:
            t = BackendTask.get_task(task_id=tid)
            if t.status not in ("completed", "failed", "closed"):
                running += 1
        except:
            continue
    print(f"🔄 {running} trial(s) still running...")
    if running == 0:
        break
    time.sleep(15)

print("✅ All trial tasks completed.")

# ✅ Find the best task by val_accuracy
best_task = None
best_val = -1.0
for tid in trial_ids:
    try:
        t = BackendTask.get_task(task_id=tid)
        metrics = t.get_last_scalar_metrics()
        val = metrics.get("accuracy", {}).get("val_accuracy", {}).get("value", None)
        if val is not None and val > best_val:
            best_val = val
            best_task = t
    except:
        continue

if not best_task:
    print("❌ No valid trial task found.")
    task.close()
    exit()

# ✅ Save and upload best params
best_params = best_task.get_parameters()
with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)
task.upload_artifact("best_params", artifact_object="best_params.json")

# ✅ Print results
print("\n🏆 Best Hyperparameters:")
for k, v in best_params.items():
    if "learning_rate" in k or "dropout" in k:
        print(f"🔧 {k} = {v}")
print(f"📈 Best val_accuracy: {best_val:.4f}")

task.close()
print("✅ step_hpo_grid.py finished and best_params.json uploaded.")
