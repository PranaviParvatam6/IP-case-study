
from pathlib import Path

ROOT = Path(
    r"D:\4TH_YEAR\image processing\Fruit-Segmentation-Counting"
)

print("Project folder exists:", ROOT.exists())
print("\nSearching for folders named images or masks...\n")

for path in ROOT.rglob("*"):
    if path.is_dir() and path.name.lower() in {"images", "masks"}:
        print(path)

print("\nSearching for the sample image...\n")

for path in ROOT.rglob("20150919_174151_image1.png"):
    print(path)