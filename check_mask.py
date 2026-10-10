
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent

MASK_PATH = (
    PROJECT_DIR
    / "detection"
    / "detection"
    / "train"
    / "masks"
    / "20150919_174151_image1.png"
)

mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_UNCHANGED)

if mask is None:
    print("ERROR: Mask file not found.")

else:
    print("Mask shape:", mask.shape)
    print("Mask data type:", mask.dtype)

    labels = np.unique(mask)

    print("Unique pixel values:", labels)
    print("Number of unique values:", len(labels))

    print("\nChecking each nonzero label:")

    for label in labels:
        if label == 0:
            continue

        binary = (mask == label).astype(np.uint8)

        count, component_map, stats, centroids = (
            cv2.connectedComponentsWithStats(binary, 8)
        )

        component_count = count - 1

        print(
            f"Label {label}: "
            f"{component_count} connected region(s)"
        )

    # Count nonzero labels as candidate fruit instances.
    fruit_count = np.count_nonzero(labels)

    print("\nTotal candidate fruit instances:", fruit_count)

    image_path = (
        PROJECT_DIR
        / "detection"
        / "detection"
        / "train"
        / "images"
        / "20150919_174151_image1.png"
    )

    image = cv2.imread(str(image_path))

    if image is not None:
        print("Original image shape:", image.shape)
        print("Mask shape:", mask.shape)

        print(
            "Image and mask dimensions match:",
            image.shape[:2] == mask.shape
        )
    else:
        print("ERROR: Original image file not found.")