from pathlib import Path
import csv
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DAY_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day"
)

NIGHT_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reid_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "reid"
    / "metadata"
    / "multi_animal_frames.csv"
)

TARGET_SPECIES = "Leopard"

MAX_FRAMES = 10

CROP_WIDTH = 360
CROP_HEIGHT = 280

LABEL_HEIGHT = 45

PADDING = 10


def get_dataset_root(dataset):

    if dataset == "day":
        return DAY_DIR

    if dataset == "night":
        return NIGHT_DIR

    raise ValueError(f"Unknown dataset: {dataset}")


def read_objects(annotation_path, target_species):

    root = ET.parse(annotation_path).getroot()

    objects = []

    for obj in root.findall("object"):

        name = obj.findtext("name")

        if name is None:
            continue

        name = name.strip()

        if name != target_species:
            continue

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = int(float(bbox.findtext("xmin")))
        ymin = int(float(bbox.findtext("ymin")))
        xmax = int(float(bbox.findtext("xmax")))
        ymax = int(float(bbox.findtext("ymax")))

        objects.append(
            {
                "species": name,
                "bbox": (xmin, ymin, xmax, ymax),
            }
        )

    return objects


def crop_image(image, bbox):

    xmin, ymin, xmax, ymax = bbox

    width, height = image.size

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)

    if xmax <= xmin or ymax <= ymin:
        return None

    crop = image.crop(
        (xmin, ymin, xmax, ymax)
    )

    crop.thumbnail(
        (
            CROP_WIDTH - 20,
            CROP_HEIGHT - 20,
        )
    )

    canvas = Image.new(
        "RGB",
        (
            CROP_WIDTH,
            CROP_HEIGHT,
        ),
        "white",
    )

    x = (
        CROP_WIDTH - crop.width
    ) // 2

    y = (
        CROP_HEIGHT - crop.height
    ) // 2

    canvas.paste(
        crop,
        (x, y),
    )

    return canvas


def main():

    print("=" * 70)
    print("VanDrishti - Negative Pair Visual Review")
    print("=" * 70)

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    candidates = [
        row
        for row in rows
        if row["species"] == TARGET_SPECIES
    ]

    candidates = candidates[:MAX_FRAMES]

    print(
        f"\nSelected {len(candidates)} "
        f"{TARGET_SPECIES} candidate frames."
    )

    output_rows = []

    for row in candidates:

        dataset = row["dataset"]
        image_name = row["image"]

        dataset_root = get_dataset_root(
            dataset
        )

        image_path = (
            dataset_root
            / "JPEGImages"
            / image_name
        )

        annotation_path = (
            dataset_root
            / "Annotations"
            / image_name.replace(
                ".jpg",
                ".xml",
            )
        )

        if not image_path.exists():
            print(
                f"[WARNING] Missing image: "
                f"{image_path}"
            )
            continue

        if not annotation_path.exists():
            print(
                f"[WARNING] Missing annotation: "
                f"{annotation_path}"
            )
            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        objects = read_objects(
            annotation_path,
            TARGET_SPECIES,
        )

        if len(objects) < 2:
            continue

        objects = objects[:2]

        crop1 = crop_image(
            image,
            objects[0]["bbox"],
        )

        crop2 = crop_image(
            image,
            objects[1]["bbox"],
        )

        if crop1 is None or crop2 is None:
            continue

        sheet_width = (
            CROP_WIDTH * 2
            + PADDING * 3
        )

        sheet_height = (
            CROP_HEIGHT
            + LABEL_HEIGHT
            + PADDING * 2
        )

        sheet = Image.new(
            "RGB",
            (
                sheet_width,
                sheet_height,
            ),
            "white",
        )

        draw = ImageDraw.Draw(sheet)

        sheet.paste(
            crop1,
            (PADDING, PADDING),
        )

        sheet.paste(
            crop2,
            (
                CROP_WIDTH
                + PADDING * 2,
                PADDING,
            ),
        )

        draw.text(
            (
                PADDING,
                CROP_HEIGHT + PADDING,
            ),
            "Animal 1",
            fill="black",
        )

        draw.text(
            (
                CROP_WIDTH
                + PADDING * 2,
                CROP_HEIGHT + PADDING,
            ),
            "Animal 2",
            fill="black",
        )

        output_path = (
            OUTPUT_DIR
            / f"negative_{dataset}_{image_name}"
        )

        sheet.save(
            output_path,
            quality=95,
        )

        output_rows.append(
            {
                "dataset": dataset,
                "image": image_name,
                "species": TARGET_SPECIES,
                "crop_1": str(
                    objects[0]["bbox"]
                ),
                "crop_2": str(
                    objects[1]["bbox"]
                ),
                "review_file": str(
                    output_path
                ),
            }
        )

        print(
            f"[SAVED] {output_path}"
        )

    print("\n" + "=" * 70)
    print(
        f"Generated {len(output_rows)} "
        "negative-pair review images."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()