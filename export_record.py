from pathlib import Path
from csv import DictWriter

RECORD_DIR_PATH = "lightning_logs/p=2"

record_dir = Path(RECORD_DIR_PATH)
ckpt_paths = list(record_dir.glob("**/*.ckpt"))

# Create a list to store the attribute dictionaries
attributes = []

for ckpt_path in ckpt_paths:

    attribute = {
        "name": str(ckpt_path).split("/")[2],
        "version": str(ckpt_path).split("/")[3].split("_")[1],
        "loss": eval(ckpt_path.stem.split("-")[2].split("=")[1]),
        "acc": eval(ckpt_path.stem.split("-")[3].split("=")[1])
    }

    # Append the attribute dictionary to the list
    attributes.append(attribute)

# Sort the attributes list by name and version
sorted_attributes = sorted(attributes, key=lambda x: (x["name"], x["version"]))

# Define the CSV file path
csv_file_path = "record.csv"

# Write the attributes to the CSV file
with open(csv_file_path, mode="w", newline="") as file:
    writer = DictWriter(file, fieldnames=["name", "version", "loss", "acc"])
    writer.writeheader()
    writer.writerows(sorted_attributes)