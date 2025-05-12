#!/usr/bin/env python
"""
step_hpo.py  – ClearML Hyper‑Parameter‑Optimizer (smoke‑test version)

• Clones your completed step_train task (Task‑ID below)
• Searches a small space (lr / batch_size / dropout) in 8 trials
• Saves best_params.json as an artifact for downstream steps
"""

from clearml import Task
from clearml.automation.opt import (
    HyperParameterOptimizer,
    UniformParameterRange,
    DiscreteParameterRange,
)
import json, pathlib

# ────────────────────────────────────────────────────────────────
# 1. Hard‑coded baseline Task‑ID (your successful step_train run)
#    — if you rerun step_train later, just update this string once.
# ────────────────────────────────────────────────────────────────
BASE_TASK_ID = "cec1224718df4369b7ae5592d6119dae"

# ────────────────────────────────────────────────────────────────
# 2. Init ClearML task
# ────────────────────────────────────────────────────────────────
task = Task.init(project_name="VisiblePipeline", task_name="step_hpo_test")

# ────────────────────────────────────────────────────────────────
# 3. Define search space (tiny for quick check)
# ────────────────────────────────────────────────────────────────
search_space = [
    UniformParameterRange("lr",          min_value=1e-4, max_value=1e-2, log_scale=True),
    DiscreteParameterRange("batch_size", values=[16, 32, 64]),
    DiscreteParameterRange("dropout",    values=[0.3, 0.4, 0.5]),
]

optimizer = HyperParameterOptimizer(
    base_task_id                 = BASE_TASK_ID,
    hyper_parameters             = search_space,
    objective_metric_title       = "val_accuracy",
    objective_metric_sign        = "max",
    max_total_number_of_configs  = 8,      # quick smoke test
    max_number_of_concurrent_tasks = 2,    # adjust per free workers
    optimizer_class              = "random_search",
    execute_queue                = "default",
)

# ────────────────────────────────────────────────────────────────
# 4. Launch search (blocks until finished)
# ────────────────────────────────────────────────────────────────
best_params = optimizer.start()

# ────────────────────────────────────────────────────────────────
# 5. Upload best params JSON for later pipeline steps
# ────────────────────────────────────────────────────────────────
p = pathlib.Path("best_params.json")
p.write_text(json.dumps(best_params, indent=2))
task.upload_artifact("best_params", str(p))

task.close()
print("✅ HPO completed – best_params.json uploaded.")
