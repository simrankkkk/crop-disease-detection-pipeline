from clearml import Task
import json

# ✅ Connect to completed HPO task
hpo_task = Task.get_task(task_id="de3a449c6ce74ba0afc6f9db86d06c26")
optimizer = hpo_task.automation()  # Access the HPO controller

# ✅ Extract the best trial
best_task = optimizer.optimizer.get_top_tasks(top_k=1)[0]
best_params = best_task.get_parameters_as_dict()
best_task_id = best_task.id

# ✅ Format the result
result = {
    "best_params": best_params.get("General", {}),
    "best_task_id": best_task_id
}

# ✅ Save locally
with open("best_result.json", "w") as f:
    json.dump(result, f, indent=4)

# ✅ Upload to HPO task as artifact
hpo_task.upload_artifact(name="best_result", artifact_object="best_result.json")
print(f"✅ Uploaded best_result.json to HPO task {hpo_task.id}")
