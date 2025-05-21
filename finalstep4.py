# finalstep4.py

from clearml import Task
import json
import time
import math

# ─── 1) Launch the HPO controller task ─────────────────────────────────────────
controller = Task.init(
    project_name="FinalProject",
    task_name="final_step_hpo",
    task_type=Task.TaskTypes.controller
)

# ─── 2) Define your hyperparameter grid ─────────────────────────────────────────
param_sets = [
    {"learning_rate": 0.001,  "dropout": 0.3},
    {"learning_rate": 0.001,  "dropout": 0.5},
    {"learning_rate": 0.0005, "dropout": 0.4},
]

# ─── 3) Pull in the pipeline parameters ─────────────────────────────────────────
pipeline_params = controller.get_parameters()
baseline_id = pipeline_params.get("Args/baseline_task_id")
dataset_id  = pipeline_params.get("Args/dataset_id")

if not baseline_id or not dataset_id:
    raise RuntimeError("❌ Must pass both Args/baseline_task_id and Args/dataset_id from the pipeline!")

baseline_task = Task.get_task(task_id=baseline_id)

# ─── 4) Clone & execute each trial inline ────────────────────────────────────────
submitted = []
for idx, hp in enumerate(param_sets, start=1):
    print(f"\n🔁 Trial {idx}: learning_rate={hp['learning_rate']}  dropout={hp['dropout']}")
    trial = Task.clone(
        source_task=baseline_task,
        name=f"hpo_trial_{idx}",
        parent=controller.id
    )
    # override the dataset so finalstep3.py picks it up
    trial.set_parameter("Args/dataset_id",       dataset_id)
    trial.set_parameter("General/learning_rate", hp["learning_rate"])
    trial.set_parameter("General/dropout",       hp["dropout"])
    print(f"▶️ Executing trial inline: {trial.id}")
    # run in-process (no ClearML-Agent needed)
    trial.execute_remotely(queue_name=None, exit_process=False)
    submitted.append(trial.id)
    print(f"✅ Completed inline run for trial: {trial.id}")

# ─── 5) (Optional) ensure all trials have reached a terminal state ─────────────
print("\n⏳ Ensuring all trials have finished…")
while not all(
    Task.get_task(tid).status in ["completed", "closed", "failed"]
    for tid in submitted
):
    time.sleep(5)

# ─── 6) Gather results & select the best trial ─────────────────────────────────
best_score  = -math.inf
best_id     = None
best_params = {}
all_results = []

for tid in submitted:
    t = Task.get_task(task_id=tid)
    scalars = t.get_reported_scalars()
    print(f"\n🔍 Scalars for {tid}: contexts = {list(scalars.keys())}")

    # hunt for "val_accuracy" series in any context
    val_acc = None
    for ctx, series in scalars.items():
        if "val_accuracy" in series:
            ys = [
                float(v)
                for v in series["val_accuracy"].get("y", [])
                if isinstance(v, (int, float))
            ]
            if ys:
                val_acc = max(ys)
            break

    if val_acc is None:
        print(f"⚠️ No val_accuracy found in {tid}; defaulting to -1.0")
        val_acc = -1.0

    trial_hp = {
        "learning_rate": t.get_parameter("General/learning_rate"),
        "dropout":       t.get_parameter("General/dropout")
    }
    print(f"📊 Trial {tid} → val_accuracy = {val_acc:.4f}")

    all_results.append({
        "task_id":      tid,
        "val_accuracy": val_acc,
        "params":       trial_hp
    })

    if val_acc > best_score:
        best_score, best_id, best_params = val_acc, tid, trial_hp

# ─── 7) Save & upload the best result ───────────────────────────────────────────
result = {
    "best_task_id": best_id,
    "best_params":  best_params,
    "all_results":  all_results
}
with open("best_result.json", "w") as f:
    json.dump(result, f, indent=2)

controller.upload_artifact("best_result", artifact_object="best_result.json")
controller.close()

print(f"\n✅ HPO complete. Best trial: {best_id} with val_accuracy = {best_score:.4f}")
