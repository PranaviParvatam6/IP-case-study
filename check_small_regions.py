
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
MASK_DIR = PROJECT_DIR / "detection" / "detection" / "train" / "masks"

total_labels = 0
labels_with_small_regions = 0
small_regions = 0
small_area_pixels = 0

for mask_path in sorted(MASK_DIR.glob("*.png")):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None or mask.ndim != 2:
        continue

    for label in np.unique(mask):
        if label == 0:
            continue

        total_labels += 1
        binary = (mask == label).astype(np.uint8)

        count, component_map, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        found_small = False

        for component_id in range(1, count):
            area = stats[component_id, cv2.CC_STAT_AREA]

            if area < 10:
                small_regions += 1
                small_area_pixels += int(area)
                found_small = True

        if found_small:
            labels_with_small_regions += 1

print("Total nonzero labels:", total_labels)
print("Labels containing regions under 10 pixels:", labels_with_small_regions)
print("Total connected regions under 10 pixels:", small_regions)
print("Total pixels in those small regions:", small_area_pixels)