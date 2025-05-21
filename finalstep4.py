from clearml import Task
import json
import time
import math

# ✅ Start HPO controller task
controller = Task.init(
    project_name="FinalProject",
    task_name="final_step_hpo",
    task_type=Task.TaskTypes.controller
)

# hyperparameter grid
param_sets = [
    {"learning_rate": 0.001, "dropout": 0.3},
    {"learning_rate": 0.001, "dropout": 0.5},
    {"learning_rate": 0.0005, "dropout": 0.4}
]

params = controller.get_parameters()
baseline_task_id = params.get("Args/baseline_task_id")
if not baseline_task_id:
    raise ValueError("❌ baseline_task_id was not provided.")

baseline = Task.get_task(task_id=baseline_task_id)
submitted = []

# clone & enqueue
for i, p in enumerate(param_sets, 1):
    print(f"🔁 Trial {i}: lr={p['learning_rate']}, dropout={p['dropout']}")
    trial = Task.clone(source_task=baseline, name=f"hpo_trial_{i}", parent=controller.id)
    trial.set_parameter("General/learning_rate", p["learning_rate"])
    trial.set_parameter("General/dropout",      p["dropout"])
    Task.enqueue(trial, queue_name="default")
    submitted.append(trial.id)
    print(f"🚀 Enqueued {trial.id}")

print("⏳ Waiting for all trials to complete...")
while not all(Task.get_task(t).status in ["completed","closed","failed"] for t in submitted):
    time.sleep(10)

# pick best
best_score = -math.inf
best_id, best_params = None, {}
all_results = []

for tid in submitted:
    t = Task.get_task(tid)
    scalars = t.get_reported_scalars()
    print(f"\n🔍 Scalars for {tid}: {list(scalars.keys())}")
    val_acc = -math.inf

    # search for “val_accuracy” in any context
    for ctx, series in scalars.items():
        if "val_accuracy" in series:
            ys = [float(v) for v in series["val_accuracy"]["y"] if isinstance(v, (int,float))]
            if ys:
                val_acc = max(ys)
            break

    if val_acc == -math.inf:
        print(f"⚠️ Could not find val_accuracy in {tid}, available series: {series.keys()}")
        val_acc = -1.0

    trial_params = {
        "learning_rate": t.get_parameter("General/learning_rate"),
        "dropout":      t.get_parameter("General/dropout")
    }
    print(f"📊 Trial {tid} → val_accuracy = {val_acc:.4f}")

    all_results.append({"task_id": tid, "val_accuracy": val_acc, "params": trial_params})
    if val_acc > best_score:
        best_score, best_id, best_params = val_acc, tid, trial_params

# save
result = {"best_task_id": best_id, "best_params": best_params, "all_results": all_results}
with open("best_result.json","w") as f:
    json.dump(result, f, indent=2)

controller.upload_artifact("best_result", artifact_object="best_result.json")
controller.close()
print(f"✅ Best Trial: {best_id} with val_accuracy: {best_score:.4f}")
