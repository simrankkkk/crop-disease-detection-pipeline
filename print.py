from clearml import Task

task = Task.get_task(task_id="e23037ffa9414ae5ae89c2cbf6e4f666")
scalars = task.get_reported_scalars()

from pprint import pprint
pprint(scalars["accuracy"]["val_accuracy"])
