# finalstep4.py

from clearml import Task
import json
import time
import math

# ─── Start HPO controller task ────────────────────────────────────────────────
controller = Task.init(
    project_name="FinalProject",
    task_name="final_step_hpo",
    task_type=Task.TaskTypes.controller
)

# ─── Hyperparameter grid ──────────────────────────────────────────────────────
param_sets = [
    {"learning_rate": 0.001, "dropout": 0.3},
    {"learning_rate": 0.001, "dropout": 0.5},
    {"learning_rate": 0.0005, "dropout": 0.4}
]

# ─── Grab the baseline task ID passed by the pipeline ─────────────────────────
baseline_task_id = controller.get_parameters().get("Args/baseline_task_id")
if not baseline_task_id:
    raise RuntimeError("✨ You must pass Args/baseline_task_id from the pipeline!")

baseline = Task.get_task(task_id=baseline_task_id)

# ─── Clone & enqueue trials ───────────────────────────────────────────────────
submitted = []
for i, p in enumerate(param_sets, start=1):
    print(f"🔁 Trial {i}: lr={p['learning_rate']}  dropout={p['dropout']}")
    trial = Task.clone(source_task=baseline, name=f"hpo_trial_{i}", parent=controller.id)
    trial.set_parameter("General/learning_rate", p["learning_rate"])
    trial.set_parameter("General/dropout",      p["dropout"])
    Task.enqueue(trial, queue_name="default")
    submitted.append(trial.id)
    print(f"🚀 Enqueued: {trial.id}")

# ─── Wait for all trials to finish ─────────────────────────────────────────────
print("⏳ Waiting for all trials to complete…")
while not all(
    Task.get_task(tid).status in ["completed", "closed", "failed"]
    for tid in submitted
):
    time.sleep(5)

# ─── Gather results & pick the best ────────────────────────────────────────────
best_score = -math.inf
best_id, best_params = None, {}
all_results = []

for tid in submitted:
    t = Task.get_task(task_id=tid)
    scalars = t.get_reported_scalars()

    # debug print of what contexts exist
    print(f"\n🔍 Scalars for {tid}: {list(scalars.keys())}")

    # try to find “val_accuracy” in any context
    val_acc = -math.inf
    found = False
    for ctx, series in scalars.items():
        if "val_accuracy" in series:
            ys = [float(v) for v in series["val_accuracy"].get("y", []) if isinstance(v, (int, float))]
            if ys:
                val_acc = max(ys)
            found = True
            break

    if not found:
        print(f"⚠️ Could not find ‘val_accuracy’ in {tid}; available contexts: {list(scalars.keys())}")
        val_acc = -1.0

    # pull back what we set
    trial_params = {
        "learning_rate": t.get_parameter("General/learning_rate"),
        "dropout":      t.get_parameter("General/dropout")
    }
    print(f"📊 Trial {tid} → val_accuracy = {val_acc:.4f}")

    all_results.append({
        "task_id": tid,
        "val_accuracy": val_acc,
        "params": trial_params
    })

    if val_acc > best_score:
        best_score = val_acc
        best_id = tid
        best_params = trial_params

# ─── Write out & upload the best result ───────────────────────────────────────
result = {
    "best_task_id": best_id,
    "best_params": best_params,
    "all_results": all_results
}
with open("best_result.json", "w") as f:
    json.dump(result, f, indent=2)

controller.upload_artifact("best_result", artifact_object="best_result.json")
controller.close()
print(f"✅ Best Trial: {best_id} with val_accuracy = {best_score:.4f}")
