from pathlib import Path
import csv
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DAY_ROOT = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day"
)

NIGHT_ROOT = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night"
)

VERIFICATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "reid"
    / "metadata"
    / "verification_log.csv"
)


NEGATIVE_FRAMES = [
    ("day", "AmurLeopard_1032.jpg"),
    ("day", "AmurLeopard_1033.jpg"),
    ("day", "AmurLeopard_1034.jpg"),
    ("day", "AmurLeopard_1035.jpg"),
    ("day", "AmurLeopard_1085.jpg"),
    ("day", "AmurLeopard_228.jpg"),
    ("day", "AmurLeopard_363.jpg"),
    ("day", "AmurLeopard_364.jpg"),
    ("day", "AmurLeopard_365.jpg"),
    ("day", "AmurLeopard_366.jpg"),
]


def get_root(dataset):
    if dataset == "day":
        return DAY_ROOT

    if dataset == "night":
        return NIGHT_ROOT

    raise ValueError(dataset)


def get_leopard_boxes(annotation_path):
    root = ET.parse(annotation_path).getroot()

    boxes = []

    for obj in root.findall("object"):

        name = obj.findtext("name")

        if name != "Leopard":
            continue

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = int(float(bbox.findtext("xmin")))
        ymin = int(float(bbox.findtext("ymin")))
        xmax = int(float(bbox.findtext("xmax")))
        ymax = int(float(bbox.findtext("ymax")))

        boxes.append(
            (xmin, ymin, xmax, ymax)
        )

    return boxes


def main():

    print("=" * 70)
    print("VanDrishti - Record Verified Negative Re-ID Pairs")
    print("=" * 70)

    existing_rows = []

    if VERIFICATION_FILE.exists():

        with open(
            VERIFICATION_FILE,
            "r",
            encoding="utf-8",
            newline="",
        ) as f:

            existing_rows = list(
                csv.DictReader(f)
            )

    next_id = len(existing_rows) + 1

    added = 0

    for dataset, image_name in NEGATIVE_FRAMES:

        dataset_root = get_root(dataset)

        annotation_path = (
            dataset_root
            / "Annotations"
            / image_name.replace(
                ".jpg",
                ".xml",
            )
        )

        boxes = get_leopard_boxes(
            annotation_path
        )

        if len(boxes) < 2:

            print(
                f"[SKIP] {image_name}: "
                f"found only {len(boxes)} leopard boxes"
            )

            continue

        bbox1 = boxes[0]
        bbox2 = boxes[1]

        row = {
            "pair_id": f"N{next_id:03d}",
            "dataset_1": dataset,
            "image_1": image_name,
            "dataset_2": dataset,
            "image_2": image_name,
            "species": "Leopard",
            "label": "0",
            "confidence": "high",
            "evidence": (
                "Two distinct Leopard instances "
                "annotated in the same frame"
            ),
            "source": "same_frame_multi_animal",
            "bbox_1": str(bbox1),
            "bbox_2": str(bbox2),
        }

        existing_rows.append(row)

        print(
            f"[ADDED] {row['pair_id']} | "
            f"{dataset} | {image_name} | "
            f"{bbox1} vs {bbox2}"
        )

        next_id += 1
        added += 1

    fieldnames = [
        "pair_id",
        "dataset_1",
        "image_1",
        "dataset_2",
        "image_2",
        "species",
        "label",
        "confidence",
        "evidence",
        "source",
        "bbox_1",
        "bbox_2",
    ]

    with open(
        VERIFICATION_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in existing_rows:

            # Older positive rows don't have bounding boxes.
            row.setdefault("bbox_1", "")
            row.setdefault("bbox_2", "")

            writer.writerow(row)

    print("\n" + "=" * 70)
    print(f"Negative pairs added: {added}")
    print(f"Total verification records: {len(existing_rows)}")
    print(f"[SAVED] {VERIFICATION_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()