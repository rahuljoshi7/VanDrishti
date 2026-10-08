from pathlib import Path
import csv
import ast
from PIL import Image


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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "reid"
    / "crops"
)

SAME_DIR = OUTPUT_ROOT / "same"
DIFFERENT_DIR = OUTPUT_ROOT / "different"


def get_dataset_root(dataset):
    """
    Return the root directory for the selected dataset.
    """

    if dataset == "day":
        return DAY_ROOT

    if dataset == "night":
        return NIGHT_ROOT

    raise ValueError(
        f"Unknown dataset: {dataset}"
    )


def parse_bbox(value):
    """
    Convert a string such as:

        "(669, 274, 1181, 494)"

    into a tuple of integers.
    """

    if not value:
        return None

    bbox = ast.literal_eval(value)

    if not isinstance(bbox, tuple):
        raise ValueError(
            f"Invalid bounding box: {value}"
        )

    if len(bbox) != 4:
        raise ValueError(
            f"Bounding box must contain 4 values: {value}"
        )

    return tuple(
        int(v)
        for v in bbox
    )


def crop_image(image, bbox):
    """
    Crop an animal from an image using VOC coordinates.
    """

    xmin, ymin, xmax, ymax = bbox

    width, height = image.size

    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))
    xmax = max(1, min(xmax, width))
    ymax = max(1, min(ymax, height))

    if xmax <= xmin or ymax <= ymin:
        raise ValueError(
            f"Invalid crop coordinates: {bbox}"
        )

    return image.crop(
        (
            xmin,
            ymin,
            xmax,
            ymax,
        )
    )


def process_pair(row):
    """
    Generate two animal crops for one verified pair.
    """

    pair_id = row["pair_id"]

    dataset_1 = row["dataset_1"]
    image_1 = row["image_1"]

    dataset_2 = row["dataset_2"]
    image_2 = row["image_2"]

    bbox_1 = parse_bbox(
        row["bbox_1"]
    )

    bbox_2 = parse_bbox(
        row["bbox_2"]
    )

    if bbox_1 is None or bbox_2 is None:

        print(
            f"[SKIP] {pair_id}: "
            "missing bounding box"
        )

        return False

    root_1 = get_dataset_root(
        dataset_1
    )

    root_2 = get_dataset_root(
        dataset_2
    )

    path_1 = (
        root_1
        / "JPEGImages"
        / image_1
    )

    path_2 = (
        root_2
        / "JPEGImages"
        / image_2
    )

    if not path_1.exists():

        print(
            f"[SKIP] {pair_id}: "
            f"missing {path_1}"
        )

        return False

    if not path_2.exists():

        print(
            f"[SKIP] {pair_id}: "
            f"missing {path_2}"
        )

        return False

    with Image.open(path_1) as image:

        image_1_rgb = image.convert(
            "RGB"
        )

    with Image.open(path_2) as image:

        image_2_rgb = image.convert(
            "RGB"
        )

    crop_1 = crop_image(
        image_1_rgb,
        bbox_1,
    )

    crop_2 = crop_image(
        image_2_rgb,
        bbox_2,
    )

    if row["label"] == "1":

        output_dir = SAME_DIR

    elif row["label"] == "0":

        output_dir = DIFFERENT_DIR

    else:

        print(
            f"[SKIP] {pair_id}: "
            f"unknown label {row['label']}"
        )

        return False

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_1 = (
        output_dir
        / f"{pair_id}_img1.jpg"
    )

    output_2 = (
        output_dir
        / f"{pair_id}_img2.jpg"
    )

    crop_1.save(
        output_1,
        quality=95,
    )

    crop_2.save(
        output_2,
        quality=95,
    )

    print(
        f"[CREATED] {pair_id} | "
        f"label={row['label']} | "
        f"{output_1.name} | "
        f"{output_2.name}"
    )

    return True


def main():

    print("=" * 70)
    print("VanDrishti - Re-ID Crop Dataset Creation")
    print("=" * 70)

    if not VERIFICATION_FILE.exists():

        print(
            f"[ERROR] Verification file not found:\n"
            f"{VERIFICATION_FILE}"
        )

        return

    SAME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIFFERENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        VERIFICATION_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    created = 0
    skipped = 0

    for row in rows:

        try:

            success = process_pair(
                row
            )

            if success:
                created += 1
            else:
                skipped += 1

        except Exception as e:

            print(
                f"[ERROR] {row['pair_id']}: "
                f"{e}"
            )

            skipped += 1

    print("\n" + "=" * 70)
    print("Crop generation completed.")
    print(f"Pairs processed successfully: {created}")
    print(f"Pairs skipped:                {skipped}")
    print(f"Output directory:              {OUTPUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()