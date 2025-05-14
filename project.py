from clearml import Task

Task.init(
    project_name="VisiblePipeline",
    task_name="register_project_placeholder",
    task_type=Task.TaskTypes.testing
).close()

print("✅ Project 'VisiblePipeline' is now registered and discoverable.")
