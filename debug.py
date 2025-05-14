from clearml import Task

Task.set_credentials(api_host='https://api.clear.ml', web_host='https://app.clear.ml')

projects = Task.get_projects()
print("\n✅ Your ClearML projects:")
for p in projects:
    print("-", p.name)
