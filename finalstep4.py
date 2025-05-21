# finalstep4.py

import subprocess, json
import math
from clearml import Task

# ─── 1) Init controller ─────────────────────────────────────────────────────────
controller = Task.init(
    project_name="FinalProject",
    task_name="final_step_hpo",
    task_type=Task.TaskTypes.controller
)

# ─── 2) Parameter grid ───────────────────────────────────────────────────────────
param_sets = [
    {"learning_rate": 0.001,  "dropout": 0.3},
    {"learning_rate": 0.001,  "dropout": 0.5},
    {"learning_rate": 0.0005, "dropout": 0.4},
]

# ─── 3) Pipeline args ────────────────────────────────────────────────────────────
p = controller.get_parameters()
dataset_id = p.get("Args/dataset_id")
if not dataset_id:
    raise RuntimeError("Args/dataset_id is required!")

# ─── 4) Run trials inline ─────────────────────────────────────────────────────────
results = []
for idx, hp in enumerate(param_sets, start=1):
    cmd = [
        "python", "finalstep3.py",
        "--dataset_id",    dataset_id,
        "--learning_rate", str(hp["learning_rate"]),
        "--dropout",       str(hp["dropout"]),
        "--epochs",        "1",
        "--image_size",    "160",
        "--train_ratio",   "0.1",
        "--val_ratio",     "0.5",
    ]
    print(f"\n🔁 Trial {idx}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout + proc.stderr

    # parse the __BEST_VAL__ line
    best_line = next(
        (l for l in out.splitlines() if l.startswith("__BEST_VAL__:")), 
        None
    )
    if best_line is None:
        raise RuntimeError(f"No __BEST_VAL__ found in trial {idx} output!")
    val_acc = float(best_line.split(":", 1)[1])
    print(f"📊 Trial {idx} → val_accuracy = {val_acc:.4f}")

    results.append({"hp": hp, "val_accuracy": val_acc})

# ─── 5) Select best ───────────────────────────────────────────────────────────────
best = max(results, key=lambda r: r["val_accuracy"])
best_hp    = best["hp"]
best_score = best["val_accuracy"]
print(f"\n✅ Best hyper‐params: {best_hp} → val_accuracy = {best_score:.4f}")

# ─── 6) Save & upload artifact ───────────────────────────────────────────────────
output = {
    "best_params": best_hp,
    "best_score":  best_score,
    "all_results": results
}
with open("best_result.json", "w") as f:
    json.dump(output, f, indent=2)

controller.upload_artifact("best_result", artifact_object="best_result.json")
controller.close()
