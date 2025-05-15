# finalstep4.py

from clearml import Task
import json
import time

# ✅ Initialize ClearML HPO controller task
controller = Task.init(
    project_name="FinalProject",
    task_name="final_step_hpo",
    task_type=Task.TaskTypes.controller
)

# ✅ Get input parameters from pipeline
params = controller.get_parameters()
baseline_task_id = params.get("Args/baseline_task_id")
dataset_id = params.get("Args/dataset_id")

if not baseline_task_id:
    raise ValueError("❌ Missing Args/baseline_task_id.")
if not dataset_id:
    raise ValueError("❌ Missing Args/dataset_id.")

print(f"📌 Using baseline task: {baseline_task_id}")
print(f"📦 Dataset ID for pipeline pass-through: {dataset_id}")

# ✅ Define your manual HPO grid
param_grid = [
    {"learning_rate": 0.001, "dropout": 0.3},
    {"learning_rate": 0.004, "dropout": 0.5},
]

queue = "default"
submitted_tasks = []

# ✅ Submit trials
for i, trial_params in enumerate(param_grid):
    print(f"🔁 Submitting HPO trial {i+1} with {trial_params}")

    base_task = Task.get_task(task_id=baseline_task_id)
    trial = Task.clone(source_task=base_task, name=f"final_hpo_trial_{i+1}", parent=controller.id)

    trial.set_parameter("General/learning_rate", trial_params["learning_rate"])
    trial.set_parameter("General/dropout", trial_params["dropout"])
    trial.set_parameter("Args/dataset_id", dataset_id)

    Task.enqueue(trial, queue_name=queue)
    submitted_tasks.append(trial.id)

# ✅ Monitor trials
print("⏳ Waiting for trials to complete...")
all_done = False
while not all_done:
    time.sleep(10)
    all_done = all(Task.get_task(tid).status in ["completed", "failed", "closed"] for tid in submitted_tasks)

# ✅ Evaluate results
best_task_id = None
best_score = -1
best_params = {}
results = []

for tid in submitted_tasks:
    t = Task.get_task(tid)
    scalars = t.get_reported_scalars()
    val_acc = -1

    try:
        params = {
            "learning_rate": t.get_parameter("General/learning_rate"),
            "dropout": t.get_parameter("General/dropout")
        }
        val_curve = scalars.get("accuracy", {}).get("val_accuracy", {})
        if "y" in val_curve and val_curve["y"]:
            val_acc = max(float(v) for v in val_curve["y"])

        print(f"✅ Trial {tid} — val_accuracy={val_acc} — {params}")
        results.append({
            "task_id": tid,
            "val_accuracy": val_acc,
            "params": params
        })

        if val_acc > best_score:
            best_score = val_acc
            best_task_id = tid
            best_params = params

    except Exception as e:
        print(f"⚠️ Error evaluating trial {tid}: {e}")

# ✅ Save results to JSON artifact
output = {
    "best_task_id": best_task_id,
    "best_params": best_params,
    "all_results": results
}
with open("best_result.json", "w") as f:
    json.dump(output, f, indent=4)

controller.upload_artifact("best_result", artifact_object="best_result.json")
controller.close()

print(f"🎯 Best HPO Trial: {best_task_id} with val_accuracy={best_score}")
