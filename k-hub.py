import kagglehub

# Download latest version
path = kagglehub.dataset_download("jedwible/uszipcodes-20231227")

print("Path to dataset files:", path)
