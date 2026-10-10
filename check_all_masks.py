
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
MASK_DIR = PROJECT_DIR / "detection" / "detection" / "train" / "masks"
IMAGE_DIR = PROJECT_DIR / "detection" / "detection" / "train" / "images"

mask_files = sorted(MASK_DIR.glob("*.png"))

print("Total mask files:", len(mask_files))

issues = []
total_instances = 0

for mask_path in mask_files:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None or mask.ndim != 2:
        issues.append(f"{mask_path.name}: unreadable or not single-channel")
        continue

    image_path = IMAGE_DIR / mask_path.name
    image = cv2.imread(str(image_path))

    if image is None:
        issues.append(f"{mask_path.name}: matching image missing")
        continue

    if image.shape[:2] != mask.shape:
        issues.append(f"{mask_path.name}: image and mask sizes differ")
        continue

    labels = np.unique(mask)
    labels = labels[labels != 0]
    total_instances += len(labels)

    for label in labels:
        binary = (mask == label).astype(np.uint8)
        count, _ = cv2.connectedComponents(binary, connectivity=8)

        if count != 2:
            issues.append(
                f"{mask_path.name}: label {label} has "
                f"{count - 1} connected regions"
            )

print("Total candidate instances:", total_instances)
print("Issues found:", len(issues))

for issue in issues[:30]:
    print("-", issue)

if len(issues) > 30:
    print("Additional issues omitted:", len(issues) - 30)