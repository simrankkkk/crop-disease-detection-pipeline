from clearml import Task

task = Task.get_task(task_id="542d5b64c9a447e39963f59e9f17fda9")
scalars = task.get_reported_scalars()

from pprint import pprint
pprint(scalars["accuracy"]["val_accuracy"])
