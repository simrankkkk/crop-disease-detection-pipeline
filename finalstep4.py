# finalstep4.py

from clearml import Task
import json
import time
import math

# ─── 1) Launch controller ─────────────────────────────────────────────────────────
controller = Task.init(
    project_name="FinalProject",
    task_name="final_step_hpo",
    task_type=Task.TaskTypes.controller
)

# ─── 2) Hyperparameter grid ───────────────────────────────────────────────────────
param_sets = [
    {"learning_rate": 0.001,  "dropout": 0.3},
    {"learning_rate": 0.001,  "dropout": 0.5},
    {"learning_rate": 0.0005, "dropout": 0.4},
]

# ─── 3) Pull pipeline args ─────────────────────────────────────────────────────────
pipeline_params = controller.get_parameters()
baseline_id = pipeline_params.get("Args/baseline_task_id")
dataset_id  = pipeline_params.get("Args/dataset_id")

if not baseline_id or not dataset_id:
    raise RuntimeError("Must pass both Args/baseline_task_id and Args/dataset_id!")

baseline = Task.get_task(task_id=baseline_id)

# ─── 4) Clone, override, and enqueue each trial ────────────────────────────────────
submitted = []
for idx, hp in enumerate(param_sets, start=1):
    print(f"🔁 Trial {idx}: lr={hp['learning_rate']}  dropout={hp['dropout']}")
    trial = Task.clone(
        source_task=baseline,
        name=f"hpo_trial_{idx}",
        parent=controller.id
    )
    # important: copy the same dataset so training code actually loads data
    trial.set_parameter("Args/dataset_id",      dataset_id)
    trial.set_parameter("General/learning_rate", hp["learning_rate"])
    trial.set_parameter("General/dropout",       hp["dropout"])
    Task.enqueue(trial, queue_name="default")
    submitted.append(trial.id)
    print(f"🚀 Enqueued {trial.id}")

# ─── 5) Wait for completion ─────────────────────────────────────────────────────────
print("⏳ Waiting for all trials to complete...")
while not all(
    Task.get_task(tid).status in ["completed", "closed", "failed"]
    for tid in submitted
):
    time.sleep(5)

# ─── 6) Gather results & pick the best ─────────────────────────────────────────────
best_score  = -math.inf
best_id     = None
best_params = {}
all_results = []

for tid in submitted:
    t = Task.get_task(task_id=tid)
    scalars = t.get_reported_scalars()

    print(f"\n🔍 Scalars for {tid}: contexts = {list(scalars.keys())}")
    # hunt for your “val_accuracy” series
    val_acc = None
    for ctx, series in scalars.items():
        if "val_accuracy" in series:
            ys = [float(v) for v in series["val_accuracy"].get("y", []) if isinstance(v, (int, float))]
            if ys:
                val_acc = max(ys)
            break

    if val_acc is None:
        print(f"⚠️ No val_accuracy found in {tid}; defaulting to -1.0")
        val_acc = -1.0

    hp_values = {
        "learning_rate": t.get_parameter("General/learning_rate"),
        "dropout":       t.get_parameter("General/dropout")
    }
    print(f"📊 Trial {tid} → val_accuracy = {val_acc:.4f}")

    all_results.append({
        "task_id":      tid,
        "val_accuracy": val_acc,
        "params":       hp_values
    })

    if val_acc > best_score:
        best_score, best_id, best_params = val_acc, tid, hp_values

# ─── 7) Save & upload best result ─────────────────────────────────────────────────
result = {
    "best_task_id": best_id,
    "best_params":  best_params,
    "all_results":  all_results
}
with open("best_result.json", "w") as fp:
    json.dump(result, fp, indent=2)

controller.upload_artifact("best_result", artifact_object="best_result.json")
controller.close()

print(f"\n✅ HPO complete. Best trial is {best_id} with val_accuracy = {best_score:.4f}")
