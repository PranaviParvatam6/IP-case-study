
from pathlib import Path
import csv
from ultralytics import YOLO

# ----------------------------------
# Project paths
# ----------------------------------
PROJECT_DIR = Path(__file__).resolve().parent


MODEL_PATH = PROJECT_DIR / (
    "runs/segment/runs/fruit_segmentation/"
    "first_training/weights/best.pt"
)

IMAGE_PATH = (
    PROJECT_DIR / "detection/detection/test/images"
)

CSV_PATH = PROJECT_DIR / "counting_evaluation.csv"

OUTPUT_DIR = PROJECT_DIR / "runs/fruit_segmentation"


def main():
    # Check model
    if not MODEL_PATH.is_file():
        print("ERROR: Trained model not found:")
        print(MODEL_PATH)
        return

    # Check test folder
    if not IMAGE_PATH.is_dir():
        print("ERROR: Test images folder not found:")
        print(IMAGE_PATH)
        return

    # Find images
    extensions = {".png", ".jpg", ".jpeg"}
    image_files = sorted(
        p for p in IMAGE_PATH.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not image_files:
        print("ERROR: No test images found.")
        return

    selected_images = image_files[:10]

    print("Loading trained model...")
    model = YOLO(str(MODEL_PATH))

    print("Trained model loaded successfully!")
    print("Model task:", model.task)
    print("Model names:", model.names)
    print("Model path:", MODEL_PATH)

    # Ask for actual counts
    actual_counts = {}

    print("\nEnter the VERIFIED actual fruit count")
    print("for each image. Count each fruit once.\n")

    for image_path in selected_images:
        while True:
            value = input(
                f"Actual count for {image_path.name}: "
            ).strip()

            try:
                actual_count = int(value)

                if actual_count < 0:
                    print("Count cannot be negative.")
                    continue

                actual_counts[image_path.name] = actual_count
                break

            except ValueError:
                print("Enter a whole number, such as 15.")

    # Predict fruit instances
    print("\nStarting predictions...")

    results = model.predict(
        source=[str(p) for p in selected_images],
        imgsz=416,
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="test_predictions",
        exist_ok=True,
        device="cpu",
        verbose=False
    )

    # Evaluate predictions
    rows = []

    for result in results:
        image_name = Path(result.path).name

        if result.masks is not None:
            predicted_count = len(result.masks.xy)
        else:
            predicted_count = 0

        actual_count = actual_counts[image_name]
        error = abs(actual_count - predicted_count)

        # Counting accuracy score
        if actual_count > 0:
            accuracy = max(
                0.0,
                (1 - error / actual_count) * 100
            )
        elif predicted_count == 0:
            accuracy = 100.0
        else:
            accuracy = 0.0

        rows.append({
            "Image Name": image_name,
            "Actual Fruit Count": actual_count,
            "Predicted Fruit Count": predicted_count,
            "Counting Error": error,
            "Counting Accuracy (%)": round(accuracy, 2)
        })

    # Save evaluation table to CSV
    fieldnames = [
        "Image Name",
        "Actual Fruit Count",
        "Predicted Fruit Count",
        "Counting Error",
        "Counting Accuracy (%)"
    ]

    with open(
        CSV_PATH, "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)

    # Display results
    print("\nFRUIT COUNTING EVALUATION")
    print("=" * 90)

    for row in rows:
        print(
            f"{row['Image Name']} | "
            f"Actual: {row['Actual Fruit Count']} | "
            f"Predicted: {row['Predicted Fruit Count']} | "
            f"Error: {row['Counting Error']} | "
            f"Accuracy: {row['Counting Accuracy (%)']}%"
        )

    # Overall metrics
    total_images = len(rows)
    total_actual = sum(
        row["Actual Fruit Count"] for row in rows
    )
    total_predicted = sum(
        row["Predicted Fruit Count"] for row in rows
    )
    total_absolute_error = sum(
        row["Counting Error"] for row in rows
    )

    mae = total_absolute_error / total_images

    if total_actual > 0:
        overall_accuracy = max(
            0.0,
            (1 - total_absolute_error / total_actual) * 100
        )
    elif total_predicted == 0:
        overall_accuracy = 100.0
    else:
        overall_accuracy = 0.0

    print("\nOVERALL RESULTS")
    print("-" * 40)
    print("Images evaluated:", total_images)
    print("Total actual fruits:", total_actual)
    print("Total predicted fruits:", total_predicted)
    print("Total absolute counting error:", total_absolute_error)
    print(f"Mean Absolute Error (MAE): {mae:.2f} fruits/image")
    print(f"Overall counting score: {overall_accuracy:.2f}%")

    print("\nCSV report saved to:")
    print(CSV_PATH)

    print("\nAnnotated images are saved in:")
    print(OUTPUT_DIR / "test_predictions")


if __name__ == "__main__":
    main()