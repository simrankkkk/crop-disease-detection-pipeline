# create_parent_task.py
import os

from clearml import Task

task = Task.init(
    project_name="FinalProject",
    task_name="FinalPipelineGroup",
    task_type=Task.TaskTypes.training
)

with open(os.environ["GITHUB_ENV"], "a") as f:
    f.write(f"PARENT_ID={task.id}\n")

task.close()
