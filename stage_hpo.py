#!/usr/bin/env python
"""
step_hpo.py  –  ClearML Hyper‑Parameter Optimiser controller
Quick test: searches lr, batch_size, dropout in 8 trials
Outputs best_params.json as an artifact.
"""

from clearml import Task
from clearml.automation.opt import (
    HyperParameterOptimizer,
    UniformParameterRange,
    DiscreteParameterRange
)
import argparse, json, pathlib

# ----------------------------------------------------------------------
# 1. CLI: we just need the Task‑ID of the baseline step_train run
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    '--base_task_id', required=True,
    help='Task‑ID of the completed step_train run to clone'
)
args = parser.parse_args()

task = Task.init(project_name="VisiblePipeline", task_name="step_hpo")

# ----------------------------------------------------------------------
# 2. Define a SMALL search space for the smoke test
#    (expand later once everything wires up)
# ----------------------------------------------------------------------
param_space = [
    UniformParameterRange('lr',          min_value=1e-4, max_value=1e-2, log_scale=True),
    DiscreteParameterRange('batch_size', values=[16, 32, 64]),
    DiscreteParameterRange('dropout',    values=[0.3, 0.4, 0.5]),
]

optimizer = HyperParameterOptimizer(
    base_task_id=args.base_task_id,              # clone trained task
    hyper_parameters=param_space,
    objective_metric_title='val_accuracy',
    objective_metric_sign='max',
    max_total_number_of_configs=8,               # quick test (expand later)
    max_number_of_concurrent_tasks=2,           # ↔ number of free workers
    optimizer_class='random_search',
    execute_queue='default',                    # your training queue
)

# ----------------------------------------------------------------------
# 3. Launch search – this blocks until all trials finish
# ----------------------------------------------------------------------
best_params = optimizer.start()

# ----------------------------------------------------------------------
# 4. Save / upload best params so the pipeline can re‑use them
# ----------------------------------------------------------------------
path = pathlib.Path('best_params.json')
path.write_text(json.dumps(best_params, indent=2))
task.upload_artifact(name='best_params', artifact_object=str(path))

task.close()
print("✅ HPO finished, best params saved.")
