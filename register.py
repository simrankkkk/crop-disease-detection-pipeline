from clearml import Task

# ✅ This task forces ClearML to create a flat, top-level project
task = Task.init(
    project_name="VisiblePipeline",      # No subfolders or nesting
    task_name="force_project_registration",
    task_type=Task.TaskTypes.training
)

# Optional logging if needed
print("✅ 'VisiblePipeline' has been force-registered as a project.")

# Close the task to finalize registration
task.close()
