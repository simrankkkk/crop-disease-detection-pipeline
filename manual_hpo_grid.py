from clearml import Task
from clearml.backend_interface.task.task import Task as BackendTask
import itertools
import json
import time

# ✅ Init controller task
controller = Task.init(project_name="T3chOpsClearMLProject", task_name="manual_hpo_grid", task_type=Task.TaskTypes.optimizer)
params = controller.get_parameters_as_dict()

# ✅ Use dynamic baseline from pipeline override
base_task_id = params.get("Args/base_task_id")
if not base_task_id:
    raise ValueError("Missing Args/base_task_id — must be passed from pipeline.")

base_task = Task.get_task(task_id=base_task_id)
queue_name = "default"

# ✅ Define hyperparameter grid
learning_rates = [0.001, 0.003, 0.005]
dropouts = [0.3, 0.4, 0.5]
grid = list(itertools.product(learning_rates, dropouts))

# ✅ Launch trials
trial_tasks = []
for lr, dr in grid:
    trial = Task.clonfrom clearml import Task
from clearml.backend_interface.task.task import Task as BackendTask
import itertools
import json
import time

# ✅ Connect ClearML controller task
controller = Task.init(project_name="T3chOpsClearMLProject", task_name="manual_hpo_grid", task_type=Task.TaskTypes.optimizer)
queue_name = "default"

# ✅ Use your successful training task as the base
base_task_id = "812a4d293cda4f619e18b91217c10f57"
base_task = Task.get_task(task_id=base_task_id)

# ✅ Define hyperparameter grid
learning_rates = [0.001, 0.003, 0.005]
dropouts = [0.3, 0.4, 0.5]
grid = list(itertools.product(learning_rates, dropouts))

# ✅ Track launched trials
trial_tasks = []

for lr, dr in grid:
    # Clone base task
    trial = Task.clone(source_task=base_task, name=f"trial_lr{lr}_do{dr}", parent=controller.id)
    
    # Override parameters in "General"
    trial.set_parameters({
        "General/learning_rate": str(lr),
        "General/dropout": str(dr)
    })

    # Enqueue trial
    Task.enqueue(task=trial.id, queue_name=queue_name)
    print(f"🚀 Enqueued trial: lr={lr}, dropout={dr}, task_id={trial.id}")
    trial_tasks.append(trial.id)

# ✅ Wait for all trials to complete
print("⏳ Waiting for trials to finish...")
while True:
    still_running = 0
    for tid in trial_tasks:
        try:
            t = BackendTask.get_task(task_id=tid)
            if t.status not in ("completed", "closed", "failed"):
                still_running += 1
        except:
            continue
    print(f"🔄 {still_running} trials still running...")
    if still_running == 0:
        break
    time.sleep(15)

# ✅ Evaluate results
best_task = None
best_val = -1.0
for tid in trial_tasks:
    try:
        t = BackendTask.get_task(task_id=tid)
        metrics = t.get_last_scalar_metrics()
        val_acc = metrics.get("accuracy", {}).get("val_accuracy", {}).get("value", None)
        if val_acc is not None and val_acc > best_val:
            best_val = val_acc
            best_task = t
    except:
        continue

# ✅ Save best hyperparameters
if best_task:
    best_params = best_task.get_parameters()
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    controller.upload_artifact("best_params", artifact_object="best_params.json")
    
    print("\n🏆 Best Hyperparameters:")
    for k, v in best_params.items():
        if "learning_rate" in k or "dropout" in k:
            print(f"🔧 {k} = {v}")
    print(f"📈 Best val_accuracy: {best_val:.4f}")
else:
    print("❌ No completed trial had valid val_accuracy.")

controller.close()
print("✅ manual_hpo_grid.py finished.")
e(source_task=base_task, name=f"trial_lr{lr}_do{dr}", parent=controller.id)
    trial.set_parameters({
        "General/learning_rate": str(lr),
        "General/dropout": str(dr)
    })
    Task.enqueue(task=trial.id, queue_name=queue_name)
    print(f"🚀 Enqueued: lr={lr}, dropout={dr} → {trial.id}")
    trial_tasks.append(trial.id)

# ✅ Wait for trials to complete
print("⏳ Waiting for trial tasks to finish...")
while True:
    running = 0
    for tid in trial_tasks:
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

print("✅ All trials completed.")

# ✅ Find best trial by val_accuracy
best_task = None
best_val = -1.0
for tid in trial_tasks:
    try:
        t = BackendTask.get_task(task_id=tid)
        metrics = t.get_last_scalar_metrics()
        val = metrics.get("accuracy", {}).get("val_accuracy", {}).get("value", None)
        if val is not None and val > best_val:
            best_val = val
            best_task = t
    except:
        continue

# ✅ Save and upload best parameters
if best_task:
    best_params = best_task.get_parameters()
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    controller.upload_artifact("best_params", artifact_object="best_params.json")

    print("\n🏆 Best Hyperparameters:")
    for k, v in best_params.items():
        if "learning_rate" in k or "dropout" in k:
            print(f"🔧 {k} = {v}")
    print(f"📈 Best val_accuracy: {best_val:.4f}")
else:
    print("❌ No valid trials found.")

controller.close()
print("✅ manual_hpo_grid.py complete and pipeline-compatible.")
