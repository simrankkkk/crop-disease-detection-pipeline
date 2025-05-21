from clearml import Task
import json
import time

# ✅ Start HPO controller task
controller = Task.init(
    project_name="FinalProject",
    task_name="final_step_hpo",
    task_type=Task.TaskTypes.controller
)

# ✅ Define manual hyperparameter combinations
param_sets = [
    {"learning_rate": 0.001, "dropout": 0.3},
    {"learning_rate": 0.001, "dropout": 0.5},
    {"learning_rate": 0.0005, "dropout": 0.4}
]

params = controller.get_parameters()
baseline_task_id = params.get("Args/baseline_task_id")

if not baseline_task_id:
    raise ValueError("❌ baseline_task_id was not provided. This must be passed from the pipeline.")

print(f"📌 Using baseline task ID: {baseline_task_id}")
baseline_task = Task.get_task(task_id=baseline_task_id)

queue = "default"
submitted_tasks = []

# ✅ Clone baseline task for each hyperparam config
for i, p in enumerate(param_sets):
    print(f"🔁 Trial {i+1}: lr={p['learning_rate']}, dropout={p['dropout']}")

    trial = Task.clone(source_task=baseline_task, name=f"hpo_trial_{i+1}", parent=controller.id)
    trial.set_parameter("General/learning_rate", p["learning_rate"])
    trial.set_parameter("General/dropout", p["dropout"])
    Task.enqueue(trial, queue_name=queue)
    submitted_tasks.append(trial.id)
    print(f"🚀 Enqueued Trial: {trial.id}")

# ✅ Wait for all trials to finish
print("⏳ Waiting for all trials to complete...")
while not all(Task.get_task(tid).status in ["completed", "failed", "closed"] for tid in submitted_tasks):
    time.sleep(10)

# ✅ Evaluate and select best result
best_task_id = None
best_score = -1
best_params = {}
all_results = []

for tid in submitted_tasks:
    try:
        t = Task.get_task(task_id=tid)
        scalars = t.get_reported_scalars()

        val_acc = -1
        acc_data = scalars.get("accuracy", {}).get("val_accuracy", {})
        if "y" in acc_data and acc_data["y"]:
            val_acc = max([float(v) for v in acc_data["y"] if isinstance(v, (float, int))])

        trial_params = {
            "learning_rate": t.get_parameter("General/learning_rate"),
            "dropout": t.get_parameter("General/dropout")
        }

        print(f"📊 Trial {tid} - val_accuracy: {val_acc:.4f}")
        all_results.append({
            "task_id": tid,
            "val_accuracy": val_acc,
            "params": trial_params
        })

        if val_acc > best_score:
            best_score = val_acc
            best_task_id = tid
            best_params = trial_params

    except Exception as e:
        print(f"⚠️ Error parsing trial {tid}: {e}")

# ✅ Save best results
result = {
    "best_task_id": best_task_id,
    "best_params": best_params,
    "all_results": all_results
}

with open("best_result.json", "w") as f:
    json.dump(result, f, indent=4)

controller.upload_artifact("best_result", artifact_object="best_result.json")
controller.close()

print(f"✅ Best Trial: {best_task_id} with val_accuracy: {best_score:.4f}")
