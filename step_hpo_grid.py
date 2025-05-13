from clearml import Task, TaskTypes
from clearml.automation import UniformParameterRange, HyperParameterOptimizer
import json
import time

# ✅ Start the HPO controller task
task = Task.init(project_name="VisiblePipeline", task_name="step_hpo_grid", task_type=TaskTypes.optimizer)
print("🔗 Connected to ClearML for HPO Grid Search")

params = task.get_parameters_as_dict()
base_task_id = params.get("Args/base_task_id") or "950c9256da504bf1ac395253816321a6"
print(f"📌 Using base_task_id = {base_task_id}")

# ✅ Set up the optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=[
        UniformParameterRange("General/learning_rate", 0.0005, 0.01, 0.002),
        UniformParameterRange("General/dropout", 0.3, 0.6, 0.1)
    ],
    objective_metric_title="accuracy",
    objective_metric_series="val_accuracy",
    objective_metric_sign="max",
    max_number_of_concurrent_tasks=2,
    total_max_jobs=5,
    min_iteration_per_job=1,
    always_create_task=False,
    save_top_k_tasks_only=1
)

# ✅ Try to start trials if needed
try:
    print("🚀 Starting HPO trials...")
    optimizer.start()

    while optimizer.is_running():
        print("⏳ HPO still running... waiting for trial tasks to complete.")
        time.sleep(10)

    print("✅ All HPO trials finished.")
except Exception as e:
    print(f"⚠️ Skipping optimizer.start(): {e}")

# ✅ Get best task and save as artifact
try:
    best_task = optimizer.get_top_tasks(top_k=1)[0]
    best_params = best_task.get_parameters_as_dict()
    best_task_id = best_task.id

    print("\n🎯 BEST PARAMETERS FOUND:")
    for k, v in best_params.get("General", {}).items():
        print(f"  {k}: {v}")
    print(f"\n🔁 Best Trial Task ID: {best_task_id}")

    result = {
        "best_params": best_params.get("General", {}),
        "best_task_id": best_task_id
    }

    with open("best_result.json", "w") as f:
        json.dump(result, f, indent=4)

    task.upload_artifact("best_result", "best_result.json")
    print("✅ Saved best_result.json for next step.")
except Exception as err:
    print(f"❌ Failed to fetch top task or save artifact: {err}")

task.close()
