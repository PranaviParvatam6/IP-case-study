
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
name = "20150919_174151_image241.png"

image_path = PROJECT_DIR / "detection" / "detection" / "train" / "images" / name
mask_path = PROJECT_DIR / "detection" / "detection" / "train" / "masks" / name
label_path = PROJECT_DIR / "dataset_yolo" / "labels" / "train" / (Path(name).stem + ".txt")
output_dir = PROJECT_DIR / "label_comparison"
output_dir.mkdir(exist_ok=True)

image = cv2.imread(str(image_path))
mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

if image is None or mask is None:
    raise FileNotFoundError("Could not read the original image or mask.")

# Save the original mask as a visible grayscale image.
mask_display = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite(str(output_dir / "original_mask.png"), mask_display)

# Draw converted YOLO polygons.
overlay = image.copy()
height, width = image.shape[:2]

with open(label_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        if len(parts) < 7:
            continue

        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        points = []

        for i in range(0, len(coords), 2):
            points.append([
                round(coords[i] * width),
                round(coords[i + 1] * height)
            ])

        points = np.array(points, dtype=np.int32)
        cv2.polylines(overlay, [points], True, (0, 255, 0), 2)

cv2.imwrite(str(output_dir / "original_image.png"), image)
cv2.imwrite(str(output_dir / "yolo_polygons.png"), overlay)

print("Saved three files in:", output_dir)