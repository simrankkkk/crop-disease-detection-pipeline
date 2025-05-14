from clearml import Task
import json
import time

# ✅ Start controller task
controller = Task.init(project_name="VisiblePipeline", task_name="step_hpo_manual_grid", task_type=Task.TaskTypes.controller)

# ✅ Define manual hyperparameter combinations
param_sets = [
    {"learning_rate": 0.001, "dropout": 0.3},
    {"learning_rate": 0.004, "dropout": 0.5},
]

baseline_task_id = "2d2455b6ba724f5c91cfd7f83607bcbd"
queue = "default"
submitted_tasks = []

for i, params in enumerate(param_sets):
    print(f"🔁 Creating trial {i+1} with params: {params}")

    baseline_task = Task.get_task(task_id=baseline_task_id)
    trial = Task.clone(source_task=baseline_task, name=f"hpo_trial_{i+1}", parent=controller.id)

    trial.set_parameter("General/learning_rate", params["learning_rate"])
    trial.set_parameter("General/dropout", params["dropout"])

    Task.enqueue(trial, queue_name=queue)
    submitted_tasks.append(trial.id)
    print(f"🚀 Enqueued: {trial.id}")

print("⏳ Waiting for all trials to complete...")
all_done = False
while not all_done:
    time.sleep(10)
    all_done = all(Task.get_task(tid).status in ["completed", "failed", "closed"] for tid in submitted_tasks)

# ✅ Track all results
best_task_id = None
best_score = -1
best_params = {}
all_results = []

for tid in submitted_tasks:
    t = Task.get_task(task_id=tid)
    scalars = t.get_reported_scalars()
    params = t.get_parameters().get("General", {})
    val_acc = -1

    try:
        val_accuracy_data = scalars.get("accuracy", {}).get("val_accuracy", {})
        if "y" in val_accuracy_data and isinstance(val_accuracy_data["y"], list) and val_accuracy_data["y"]:
            val_acc = max([float(v) for v in val_accuracy_data["y"] if isinstance(v, (float, int))])

        print(f"Trial {tid} | lr={params.get('learning_rate')} | dropout={params.get('dropout')} | val_accuracy={val_acc}")

        all_results.append({
            "task_id": tid,
            "val_accuracy": val_acc,
            "params": params
        })

        if val_acc > best_score:
            best_score = val_acc
            best_task_id = tid
            best_params = params

    except Exception as e:
        print(f"⚠️  Skipping {tid}: {e}")

result = {
    "best_task_id": best_task_id,
    "best_params": best_params,
    "all_results": all_results
}

with open("best_result.json", "w") as f:
    json.dump(result, f, indent=4)

controller.upload_artifact("best_result", artifact_object="best_result.json")
controller.close()
print(f"✅ Best trial: {best_task_id} with val_accuracy: {best_score}")
