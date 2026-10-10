
from pathlib import Path
import random
import shutil

import cv2
import numpy as np

# Project and dataset paths
PROJECT_DIR = Path(
    r"D:\4TH_YEAR\image processing\Fruit-Segmentation-Counting"
)
SOURCE_DIR = PROJECT_DIR / "detection" / "detection"
OUTPUT_DIR = PROJECT_DIR / "dataset_yolo"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_SEED = 42
VALIDATION_RATIO = 0.15



def find_pairs(split):
    """Find images and masks in the expected dataset folders."""
    images_dir = SOURCE_DIR / split / "images"
    masks_dir = SOURCE_DIR / split / "masks"

    print(f"\nChecking split: {split}")
    print("Images path:", images_dir)
    print("Images folder exists:", images_dir.exists())
    print("Masks path:", masks_dir)
    print("Masks folder exists:", masks_dir.exists())

    if not images_dir.exists() or not masks_dir.exists():
        return []

    pairs = []

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        mask_path = masks_dir / (image_path.stem + ".png")

        if mask_path.exists():
            pairs.append((image_path, mask_path))

    return pairs


def mask_to_yolo(mask_path, image_width, image_height):
    """Convert each non-zero mask label into a polygon annotation."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a single-channel mask: {mask_path.name}"
        )

    if mask.shape != (image_height, image_width):
        raise ValueError(
            f"Image and mask sizes differ: {mask_path.name}"
        )

    annotations = []

    # Zero is treated as background.
    for instance_id in np.unique(mask):
        if instance_id == 0:
            continue

        binary_region = (mask == instance_id).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary_region,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue

            # Reduce the number of polygon points slightly.
            contour = cv2.approxPolyDP(
                contour, 1.0, True
            ).reshape(-1, 2)

            if len(contour) < 3:
                continue

            coordinates = []

            for x, y in contour:
                coordinates.extend([
                    min(max(x / image_width, 0.0), 1.0),
                    min(max(y / image_height, 0.0), 1.0)
                ])

            # Class 0 represents apple.
            line = "0 " + " ".join(
                f"{value:.6f}" for value in coordinates
            )
            annotations.append(line)

    return annotations


def prepare_split(pairs, split_name):
    """Copy images and create YOLO label files."""
    images_out = OUTPUT_DIR / "images" / split_name
    labels_out = OUTPUT_DIR / "labels" / split_name

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    converted = 0
    total_instances = 0

    for image_path, mask_path in pairs:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        height, width = image.shape[:2]

        annotations = mask_to_yolo(
            mask_path, width, height
        )

        # Copy the image, preserving its original extension.
        shutil.copy2(
            image_path,
            images_out / image_path.name
        )

        label_path = labels_out / (
            image_path.stem + ".txt"
        )

        label_path.write_text(
            "\n".join(annotations),
            encoding="utf-8"
        )

        converted += 1
        total_instances += len(annotations)

    print(
        f"{split_name}: {converted} images converted, "
        f"{total_instances} polygon annotations"
    )


def main():
    train_pairs = find_pairs("train")
    test_pairs = find_pairs("test")

    print("Training image-mask pairs:", len(train_pairs))
    print("Test image-mask pairs:", len(test_pairs))

    if not train_pairs:
        raise RuntimeError(
            "No training pairs found. Check the dataset folders."
        )

    # Keep a reproducible training/validation split.
    random.Random(RANDOM_SEED).shuffle(train_pairs)

    if len(train_pairs) >= 2:
        val_count = max(
            1, round(len(train_pairs) * VALIDATION_RATIO)
        )
        val_pairs = train_pairs[:val_count]
        train_pairs = train_pairs[val_count:]
    else:
        val_pairs = []

    prepare_split(train_pairs, "train")
    prepare_split(val_pairs, "val")

    if test_pairs:
        prepare_split(test_pairs, "test")

    # Create the YOLO dataset configuration.
    yaml_text = (
        f"path: {OUTPUT_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: apple\n"
    )

    (OUTPUT_DIR / "data.yaml").write_text(
        yaml_text, encoding="utf-8"
    )

    print("\nDataset conversion finished.")
    print("Output folder:", OUTPUT_DIR)
    print("Configuration:", OUTPUT_DIR / "data.yaml")


if __name__ == "__main__":
    main()