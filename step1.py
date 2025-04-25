from clearml import Task, Dataset

def step1():
    # Initialize ClearML Task
    Task.init(project_name="PlantPipeline", task_name="Step 1 - Load Dataset")

    # Load dataset by ID
    dataset = Dataset.get(dataset_id="105163c10d0a4bbaa06055807084ec71")
    local_path = dataset.get_local_copy()

    print("✅ Dataset path:", local_path)
    return local_path

if __name__ == "__main__":
    step1()
