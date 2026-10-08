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


def get_root(dataset):
    """
    Return the root directory for the selected dataset.
    """

    if dataset == "day":
        return DAY_ROOT

    if dataset == "night":
        return NIGHT_ROOT

    raise ValueError(f"Unknown dataset: {dataset}")


def get_species_boxes(annotation_path, species):
    """
    Extract all bounding boxes for a given species
    from a VOC XML annotation file.
    """

    root = ET.parse(annotation_path).getroot()

    boxes = []

    for obj in root.findall("object"):

        name = obj.findtext("name")

        if name is None:
            continue

        name = name.strip()

        if name != species:
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


def get_box(dataset, image_name, species):
    """
    Locate the VOC XML annotation corresponding to an image
    and return bounding boxes for the requested species.
    """

    root = get_root(dataset)

    annotation_path = (
        root
        / "Annotations"
        / image_name.replace(
            ".jpg",
            ".xml",
        )
    )

    if not annotation_path.exists():
        print(
            f"[WARNING] Annotation not found: "
            f"{annotation_path}"
        )

        return []

    return get_species_boxes(
        annotation_path,
        species,
    )


def normalize_species(species):
    """
    Convert filename-style species names into the
    annotation vocabulary used by the VOC dataset.

    Example:
        AmurLeopard -> Leopard
    """

    if species == "AmurLeopard":
        return "Leopard"

    return species


def main():

    print("=" * 70)
    print("VanDrishti - Complete Positive Pair Bounding Boxes")
    print("=" * 70)

    if not VERIFICATION_FILE.exists():

        print(
            f"[ERROR] Verification file not found:\n"
            f"{VERIFICATION_FILE}"
        )

        return

    with open(
        VERIFICATION_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    updated_rows = []

    for row in rows:

        # ----------------------------------------------------
        # Negative pairs already contain bounding boxes.
        # ----------------------------------------------------

        if row["label"] == "0":

            updated_rows.append(row)
            continue

        # ----------------------------------------------------
        # Positive pair
        # ----------------------------------------------------

        original_species = row["species"]

        # Convert filename-style name to the VOC annotation
        # class name.
        annotation_species = normalize_species(
            original_species
        )

        # ----------------------------------------------------
        # Image 1
        # ----------------------------------------------------

        boxes_1 = get_box(
            row["dataset_1"],
            row["image_1"],
            annotation_species,
        )

        # ----------------------------------------------------
        # Image 2
        # ----------------------------------------------------

        boxes_2 = get_box(
            row["dataset_2"],
            row["image_2"],
            annotation_species,
        )

        # ----------------------------------------------------
        # Validate Image 1
        # ----------------------------------------------------

        if len(boxes_1) != 1:

            print(
                f"[WARNING] {row['pair_id']} "
                f"{row['image_1']}: "
                f"expected exactly 1 "
                f"{annotation_species}, "
                f"found {len(boxes_1)}"
            )

            updated_rows.append(row)
            continue

        # ----------------------------------------------------
        # Validate Image 2
        # ----------------------------------------------------

        if len(boxes_2) != 1:

            print(
                f"[WARNING] {row['pair_id']} "
                f"{row['image_2']}: "
                f"expected exactly 1 "
                f"{annotation_species}, "
                f"found {len(boxes_2)}"
            )

            updated_rows.append(row)
            continue

        # ----------------------------------------------------
        # Store bounding boxes
        # ----------------------------------------------------

        row["bbox_1"] = str(
            boxes_1[0]
        )

        row["bbox_2"] = str(
            boxes_2[0]
        )

        # Standardize the species name in the metadata.
        row["species"] = annotation_species

        print(
            f"[UPDATED] {row['pair_id']} | "
            f"{row['image_1']} "
            f"{boxes_1[0]} | "
            f"{row['image_2']} "
            f"{boxes_2[0]}"
        )

        updated_rows.append(row)

    # --------------------------------------------------------
    # Save updated verification metadata
    # --------------------------------------------------------

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

        writer.writerows(
            updated_rows
        )

    print("\n" + "=" * 70)
    print(
        "Positive bounding-box completion finished."
    )
    print(
        f"[SAVED] {VERIFICATION_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()