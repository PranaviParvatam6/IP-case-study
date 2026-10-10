
from pathlib import Path
import cv2
import random

PROJECT_DIR = Path(__file__).resolve().parent

IMAGE_DIR = PROJECT_DIR / "detection" / "detection" / "train" / "images"
LABEL_DIR = PROJECT_DIR / "dataset_yolo" / "labels" / "train"
OUTPUT_DIR = PROJECT_DIR / "label_visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

label_files = sorted(LABEL_DIR.glob("*.txt"))

if not label_files:
    print("No label files found!")
    raise SystemExit

random.seed(42)
selected_files = random.sample(label_files, min(5, len(label_files)))

for label_file in selected_files:
    image_path = IMAGE_DIR / f"{label_file.stem}.png"
    image = cv2.imread(str(image_path))

    if image is None:
        print("Image not found:", image_path)
        continue

    height, width = image.shape[:2]
    count = 0

    for line in label_file.read_text().splitlines():
        parts = line.split()

        if len(parts) < 7:
            continue

        coords = [float(value) for value in parts[1:]]
        points = []

        for i in range(0, len(coords), 2):
            x = int(coords[i] * width)
            y = int(coords[i + 1] * height)
            points.append([x, y])

        polygon = __import__("numpy").array(points, dtype="int32")

        cv2.polylines(
            image, [polygon], True, (0, 255, 0), 2
        )
        count += 1

    output_path = OUTPUT_DIR / f"{label_file.stem}_checked.jpg"
    cv2.imwrite(str(output_path), image)
    print(f"{output_path.name}: {count} polygons")

print("\nOpen the label_visualizations folder to inspect the images.")