from clearml import Task, TaskTypes
from clearml.automation import UniformParameterRange, HyperParameterOptimizer
import time

# ✅ Initialize the HPO Task
task = Task.init(project_name="VisiblePipeline", task_name="step_hpo_grid", task_type=TaskTypes.optimizer)
print("🔗 Connected to ClearML for HPO Grid Search")

# ✅ Use working baseline task ID
params = task.get_parameters_as_dict()
base_task_id = params.get("Args/base_task_id") or "950c9256da504bf1ac395253816321a6"
print(f"📌 Using base_task_id = {base_task_id}")

# ✅ Configure the HPO grid search
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
    always_create_task=True,
    save_top_k_tasks_only=1
)

# ✅ Start and monitor the optimization
print("🚀 Starting HPO grid search...")
optimizer.start()

# Optional: Wait for trial tasks to start showing up
print("⏳ Waiting 20s to allow trial tasks to start...")
time.sleep(20)
# ✅ Wait until all tasks finish
while optimizer.is_running():
    print("⏳ HPO still running... waiting for trials to finish.")
    time.sleep(10)

print("✅ All HPO trials completed.")

# ✅ Finalize the task
task.close()
print("✅ HPO grid search completed and task closed.")
