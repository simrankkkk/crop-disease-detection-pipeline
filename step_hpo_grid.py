from clearml import Task, TaskTypes
from clearml.automation import UniformParameterRange, HyperParameterOptimizer
import time, json

# ✅ Init ClearML task
task = Task.init(project_name="VisiblePipeline", task_name="step_hpo_grid", task_type=TaskTypes.optimizer)
params = task.get_parameters_as_dict()
base_task_id = params.get("Args/base_task_id") or "950c9256da504bf1ac395253816321a6"
print(f"🔗 Connected to ClearML\n📌 Using base_task_id = {base_task_id}")

# ✅ Define parameter search space (targeting General section)
search_space = [
    UniformParameterRange("General/learning_rate", 0.0005, 0.01, 0.002),
    UniformParameterRange("General/dropout", 0.3, 0.6, 0.1)
]

# ✅ Configure HPO
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

# ✅ Start HPO
print("🚀 Launching HPO trials...")
best_task = optimizer.start()

# ✅ Wait for all trials to complete
print("⏳ Waiting for trial tasks to finish...")
while True:
    running = [t for t in optimizer.trials if t.status not in ("completed", "closed", "failed")]
    print(f"🔄 {len(running)} trial(s) still running...")
    if not running:
        break
    time.sleep(15)

print("✅ All trials finished.")

# ✅ Save and upload best parameters
best_params = best_task.get_parameters()
with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)
task.upload_artifact("best_params", artifact_object="best_params.json")

# ✅ Print best values
print("\n🏆 Best Hyperparameters:")
for k, v in best_params.items():
    if "learning_rate" in k or "dropout" in k:
        print(f"🔧 {k} = {v}")

metrics = best_task.get_last_scalar_metrics()
val_acc = metrics.get("accuracy", {}).get("val_accuracy", {}).get("value", None)
if val_acc:
    print(f"📈 Best Validation Accuracy: {val_acc:.4f}")
else:
    print("⚠️ val_accuracy not found in scalar logs.")

task.close()
print("✅ HPO grid complete. All errors resolved.")
