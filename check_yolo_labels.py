
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LABEL_DIRS = [
    PROJECT_DIR / "dataset_yolo" / "labels" / "train",
    PROJECT_DIR / "dataset_yolo" / "labels" / "val",
]

for label_dir in LABEL_DIRS:
    print("\nChecking:", label_dir)

    if not label_dir.exists():
        print("ERROR: Folder not found!")
        continue

    files = sorted(label_dir.glob("*.txt"))
    print("Label files:", len(files))

    empty_files = []
    malformed_lines = []
    invalid_class = []
    annotation_count = 0

    for file in files:
        content = file.read_text(encoding="utf-8").strip()

        if not content:
            empty_files.append(file.name)
            continue

        for line_number, line in enumerate(content.splitlines(), start=1):
            parts = line.split()

            # YOLO segmentation labels need a class ID
            # followed by at least 3 polygon coordinate pairs.
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                malformed_lines.append(
                    f"{file.name}, line {line_number}: {line}"
                )
                continue

            try:
                values = [float(value) for value in parts]
            except ValueError:
                malformed_lines.append(
                    f"{file.name}, line {line_number}: non-numeric value"
                )
                continue

            if values[0] != 0:
                invalid_class.append(
                    f"{file.name}, line {line_number}: class {values[0]}"
                )

            coords = values[1:]
            if any(value < 0 or value > 1 for value in coords):
                malformed_lines.append(
                    f"{file.name}, line {line_number}: coordinates outside 0-1"
                )
                continue

            annotation_count += 1

    print("Total annotations:", annotation_count)
    print("Empty label files:", len(empty_files))
    print("Malformed annotations:", len(malformed_lines))
    print("Unexpected class IDs:", len(invalid_class))

    for issue in malformed_lines[:5]:
        print("Malformed:", issue)

    for issue in invalid_class[:5]:
        print("Unexpected class:", issue)

    for name in empty_files[:5]:
        print("Empty:", name)