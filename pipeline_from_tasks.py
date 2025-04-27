@PipelineDecorator.component(
    name="stage_preprocess",
    return_values=["processed_dataset_id"],
    execution_queue="default"
)
def stage_preprocess(uploaded_dataset_path):
    from clearml import Dataset
    from pathlib import Path
    from PIL import Image
    import shutil

    inp = Path(uploaded_dataset_path)
    out = Path("processed_data")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # resize loop (same as before)
    for cls in inp.iterdir():
        if not cls.is_dir(): continue
        dst = out / cls.name
        dst.mkdir(exist_ok=True)
        for img in cls.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            Image.open(img).convert("RGB").resize((224,224)).save(dst / img.name)

    # create, add files, upload, then finalize
    ds = Dataset.create(
        dataset_name="plant_processed_data",
        dataset_project="PlantPipeline"
    )
    ds.add_files(str(out))
    ds.upload()                       # ◀─ flush all files to the server
    processed_id = ds.finalize()      # ◀─ safe now, no pending uploads
    print("✅ Created processed dataset ID:", processed_id)
    return processed_id
