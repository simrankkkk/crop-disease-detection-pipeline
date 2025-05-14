from clearml import Task
import json
import time

# ✅ Start controller task
controller = Task.init(project_name="VisiblePipeline", task_name="step_hpo_manual_grid", task_type=Task.TaskTypes.controller)

# ✅ Define 2 manual hyperparameter combinations
param_sets = [
    {"learning_rate": 0.001, "dropout": 0.3},
    {"learning_rate": 0.004, "dropout": 0.5},
]

# ✅ ID of working baseline task to clone
baseline_task_id = "2d2455b6ba724f5c91cfd7f83607bcbd"

# ✅ Queue name
queue = "default"

# ✅ Track submitted trial task IDs
submitted_tasks = []

for i, params in enumerate(param_sets):
    print(f"🔁 Creating trial {i+1} with params: {params}")

    # Clone baseline
    baseline_task = Task.get_task(task_id=baseline_task_id)
    trial = Task.clone(source_task=baseline_task, name=f"hpo_trial_{i+1}", parent=controller.id)


    # Override hyperparameters
    trial.set_parameter("General/learning_rate", params["learning_rate"])
    trial.set_parameter("General/dropout", params["dropout"])

    # Enqueue the trial
    Task.enqueue(trial, queue_name=queue)
    submitted_tasks.append(trial.id)
    print(f"🚀 Enqueued: {trial.id}")

# ✅ Wait for trials to finish
print("⏳ Waiting for all trials to complete...")
all_done = False
while not all_done:
    time.sleep(10)
    all_done = all(Task.get_task(tid).status in ["completed", "failed", "closed"] for tid in submitted_tasks)

# ✅ Select best by val_accuracy
best_task_id = None
best_score = -1
best_params = {}

for tid in submitted_tasks:
    t = Task.get_task(task_id=tid)
    scalars = t.get_reported_scalars()
    try:
        val_acc = max(scalars["accuracy"]["val_accuracy"].values())
        print(f"✅ {tid} val_accuracy: {val_acc}")
        if val_acc > best_score:
            best_score = val_acc
            best_task_id = tid
            best_params = t.get_parameters().get("General", {})
    except:
        print(f"⚠️  No val_accuracy found for {tid}")

# ✅ Save best result
result = {"best_task_id": best_task_id, "best_params": best_params}
with open("best_result.json", "w") as f:
    json.dump(result, f, indent=4)

controller.upload_artifact("best_result", artifact_object="best_result.json")
controller.close()
print(f"✅ Best trial: {best_task_id} with val_accuracy: {best_score}")
