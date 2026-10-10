
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Change this path to your extracted dataset folder.
DATASET_DIR = Path(r"D:\\4TH_YEAR\\image processing\\Fruit-Segmentation-Counting\detection\detection\\train")

images_dir = DATASET_DIR / "images"
masks_dir = DATASET_DIR / "masks"

print("Dataset exists:", DATASET_DIR.exists())
print("Images folder exists:", images_dir.exists())
print("Masks folder exists:", masks_dir.exists())

# Find available image and mask files.
image_files = sorted(
    p for p in images_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

mask_files = sorted(masks_dir.glob("*.png"))

print("Number of images:", len(image_files))
print("Number of masks:", len(mask_files))

if not image_files or not mask_files:
    raise FileNotFoundError(
        "No images or PNG masks found. Check DATASET_DIR."
    )

# Display the first image and mask for inspection.
image_path = image_files[0]
image = cv2.imread(str(image_path))

if image is None:
    raise ValueError(f"Could not read {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Select a mask with the same stem if possible.
matching_masks = [
    p for p in mask_files if p.stem == image_path.stem
]

mask_path = (
    matching_masks[0]
    if matching_masks
    else mask_files[0]
)

mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

if mask is None:
    raise ValueError(f"Could not read {mask_path}")

print("Image:", image_path.name)
print("Mask:", mask_path.name)
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask data type:", mask.dtype)
print("Unique mask values:", np.unique(mask)[:30])

# Display both files.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image)
axes[0].set_title("Original Orchard Image")

if mask.ndim == 2:
    axes[1].imshow(mask, cmap="gray")
else:
    axes[1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

axes[1].set_title("Dataset Mask")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

# Count distinct non-background labels in the mask
labels = np.unique(mask)
fruit_labels = labels[labels != 0]

print("Background label: 0")
print("Number of non-background labels:", len(fruit_labels))
print("First 30 non-background labels:", fruit_labels[:30])


# Display each mask label in a different color
colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
rng = np.random.default_rng(42)

for label in np.unique(mask):
    if label == 0:
        continue

    colored_mask[mask == label] = rng.integers(
        50, 256, size=3, dtype=np.uint8
    )

plt.figure(figsize=(12, 7))
plt.imshow(colored_mask)
plt.title("Fruit Instance Labels")
plt.axis("off")
plt.tight_layout()
plt.show()


# Inspect how many pixels belong to each mask label
labels, counts = np.unique(mask, return_counts=True)

print("\nLargest mask labels by pixel area:")
for label, count in sorted(
    zip(labels, counts), key=lambda item: item[1], reverse=True
)[:15]:
    print(f"Label {label}: {count} pixels")

# Display the original mask without changing its values
plt.figure(figsize=(10, 7))
plt.imshow(mask, cmap="nipy_spectral")
plt.colorbar(label="Mask label")
plt.title("Original Mask Labels")
plt.axis("off")
plt.tight_layout()
plt.show()


# Show original image, mask, and overlay
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image)
axes[0].set_title("Original Orchard Image")

axes[1].imshow(mask, cmap="nipy_spectral")
axes[1].set_title("Original Mask Labels")

axes[2].imshow(image)
axes[2].imshow(mask, cmap="nipy_spectral", alpha=0.55)
axes[2].set_title("Image + Mask Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


# Count connected regions for each non-background label
print("\nChecking individual mask labels:")

for label in np.unique(mask):
    if label == 0:
        continue

    region = (mask == label).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        region, connectivity=8
    )

    # Ignore component 0, which is the background
    areas = stats[1:, cv2.CC_STAT_AREA]

    if len(areas) > 0:
        print(
            f"Label {label}: "
            f"{len(areas)} connected region(s), "
            f"largest area = {areas.max()} pixels"
        )

        
# Draw each non-background mask region on the original image
visual_check = image.copy()

for label in np.unique(mask):
    if label == 0:
        continue

    region = (mask == label).astype(np.uint8)

    contours, _ = cv2.findContours(
        region,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        visual_check,
        contours,
        -1,
        (255, 0, 0),
        2
    )

plt.figure(figsize=(12, 8))
plt.imshow(visual_check)
plt.title("Detected Mask Region Boundaries")
plt.axis("off")
plt.tight_layout()
plt.show()