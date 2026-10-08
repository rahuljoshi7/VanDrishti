from pathlib import Path
import csv
import xml.etree.ElementTree as ET

from PIL import Image


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DAY_IMAGE_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day\JPEGImages"
)

NIGHT_IMAGE_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night\JPEGImages"
)

DAY_XML_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day\Annotations"
)

NIGHT_XML_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night\Annotations"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "reid"
    / "crops"
    / "phase2b_same"
)

METADATA_FILE = (
    ROOT
    / "data"
    / "reid"
    / "metadata"
    / "phase2b_crop_metadata.csv"
)


# =========================================================
# PAIRS
# =========================================================

PAIRS = [
    {
        "pair_id": "B001",
        "dataset": "day",
        "image_1": "blackbear104.jpg",
        "image_2": "blackbear106.jpg",
        "species": "BlackBear",
        "difficulty": "medium",
        "confidence": "high",
        "evidence": (
            "Same continuous observation; "
            "same individual visible across frames"
        ),
    },
    {
        "pair_id": "B002",
        "dataset": "day",
        "image_1": "blackbear124.jpg",
        "image_2": "blackbear127.jpg",
        "species": "BlackBear",
        "difficulty": "medium",
        "confidence": "high",
        "evidence": (
            "Same continuous observation; "
            "same individual visible across frames"
        ),
    },
    {
        "pair_id": "B003",
        "dataset": "day",
        "image_1": "blackbear136.jpg",
        "image_2": "blackbear137.jpg",
        "species": "BlackBear",
        "difficulty": "easy",
        "confidence": "high",
        "evidence": (
            "Adjacent continuous observation; "
            "same individual"
        ),
    },
    {
        "pair_id": "L001",
        "dataset": "day",
        "image_1": "leopardcat__1146.jpg",
        "image_2": "leopardcat__1152.jpg",
        "species": "LeopardCat",
        "difficulty": "medium",
        "confidence": "high",
        "evidence": (
            "Same continuous observation; "
            "same individual"
        ),
    },
    {
        "pair_id": "L002",
        "dataset": "day",
        "image_1": "leopardcat__1154.jpg",
        "image_2": "leopardcat__1163.jpg",
        "species": "LeopardCat",
        "difficulty": "medium",
        "confidence": "high",
        "evidence": (
            "Same continuous observation; "
            "same individual"
        ),
    },
    {
        "pair_id": "L003",
        "dataset": "day",
        "image_1": "leopardcat__1161.jpg",
        "image_2": "leopardcat__1171.jpg",
        "species": "LeopardCat",
        "difficulty": "hard",
        "confidence": "high",
        "evidence": (
            "Same event with substantial temporal gap; "
            "same individual appearance"
        ),
    },
    {
        "pair_id": "M001",
        "dataset": "day",
        "image_1": "muskdeer_410.jpg",
        "image_2": "muskdeer_429.jpg",
        "species": "MuskDeer",
        "difficulty": "medium",
        "confidence": "high",
        "evidence": (
            "Same dated event; same individual"
        ),
    },
    {
        "pair_id": "M002",
        "dataset": "day",
        "image_1": "muskdeer_488.jpg",
        "image_2": "muskdeer_503.jpg",
        "species": "MuskDeer",
        "difficulty": "medium",
        "confidence": "high",
        "evidence": (
            "Same dated event; same individual"
        ),
    },
    {
        "pair_id": "M003",
        "dataset": "day",
        "image_1": "muskdeer_523.jpg",
        "image_2": "muskdeer_589.jpg",
        "species": "MuskDeer",
        "difficulty": "hard",
        "confidence": "high",
        "evidence": (
            "Same dated event; same individual"
        ),
    },
    {
        "pair_id": "S001",
        "dataset": "day",
        "image_1": "sikadeer1075.jpg",
        "image_2": "sikadeer1076.jpg",
        "species": "SikaDeer",
        "difficulty": "easy",
        "confidence": "high",
        "evidence": (
            "Adjacent continuous observation; "
            "same individual"
        ),
    },
]


# =========================================================
# SPECIES NORMALIZATION
# =========================================================

def normalize_species(name):

    name = (
        name
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
    )

    aliases = {
        "blackbear": "blackbear",
        "leopardcat": "leopardcat",
        "muskdeer": "muskdeer",
        "sikadeer": "sikadeer",
        "amurleopard": "leopard",
    }

    return aliases.get(
        name,
        name,
    )


# =========================================================
# DATASET PATHS
# =========================================================

def get_dataset_paths(dataset):

    if dataset == "day":

        return (
            DAY_IMAGE_DIR,
            DAY_XML_DIR,
        )

    if dataset == "night":

        return (
            NIGHT_IMAGE_DIR,
            NIGHT_XML_DIR,
        )

    raise ValueError(
        f"Unknown dataset: {dataset}"
    )


# =========================================================
# FIND IMAGE
# =========================================================

def find_image(
    image_dir,
    image_name,
):

    path = (
        image_dir
        / image_name
    )

    if path.exists():
        return path

    target = image_name.lower()

    for candidate in image_dir.iterdir():

        if (
            candidate.is_file()
            and candidate.name.lower()
            == target
        ):
            return candidate

    raise FileNotFoundError(
        f"Image not found: {image_name}"
    )


# =========================================================
# READ BOUNDING BOX
# =========================================================

def find_matching_bbox(
    xml_dir,
    image_name,
    expected_species,
):

    xml_path = (
        xml_dir
        / f"{Path(image_name).stem}.xml"
    )

    if not xml_path.exists():

        raise FileNotFoundError(
            f"XML not found: {xml_path}"
        )

    tree = ET.parse(
        xml_path
    )

    root = tree.getroot()

    expected = normalize_species(
        expected_species
    )

    matches = []

    for obj in root.findall("object"):

        name_node = obj.find("name")

        if name_node is None:
            continue

        species = (
            name_node.text
            or ""
        ).strip()

        if normalize_species(
            species
        ) != expected:

            continue

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = int(
            float(
                bbox.findtext(
                    "xmin",
                    "0",
                )
            )
        )

        ymin = int(
            float(
                bbox.findtext(
                    "ymin",
                    "0",
                )
            )
        )

        xmax = int(
            float(
                bbox.findtext(
                    "xmax",
                    "0",
                )
            )
        )

        ymax = int(
            float(
                bbox.findtext(
                    "ymax",
                    "0",
                )
            )
        )

        matches.append(
            (
                xmin,
                ymin,
                xmax,
                ymax,
            )
        )

    if len(matches) != 1:

        raise RuntimeError(
            f"Expected exactly one "
            f"{expected_species} object in "
            f"{xml_path}, found {len(matches)}"
        )

    return matches[0]


# =========================================================
# CREATE CROP
# =========================================================

def create_crop(
    image_path,
    bbox,
    output_path,
):

    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )

        width, height = (
            image.size
        )

        xmin, ymin, xmax, ymax = bbox

        # Clamp coordinates to image bounds.
        xmin = max(
            0,
            min(
                xmin,
                width - 1,
            ),
        )

        ymin = max(
            0,
            min(
                ymin,
                height - 1,
            ),
        )

        xmax = max(
            xmin + 1,
            min(
                xmax,
                width,
            ),
        )

        ymax = max(
            ymin + 1,
            min(
                ymax,
                height,
            ),
        )

        crop = image.crop(
            (
                xmin,
                ymin,
                xmax,
                ymax,
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        crop.save(
            output_path,
            quality=95,
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)

    print(
        "VanDrishti - Phase 2B "
        "Re-ID Crop Generation"
    )

    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_rows = []

    processed = 0
    failed = 0

    # -----------------------------------------------------
    # Process pairs
    # -----------------------------------------------------

    for pair in PAIRS:

        print()
        print(
            f"Processing {pair['pair_id']} "
            f"| {pair['species']}"
        )

        try:

            image_dir, xml_dir = (
                get_dataset_paths(
                    pair["dataset"]
                )
            )

            # ---------------------------------------------
            # Image 1
            # ---------------------------------------------

            image_1_path = find_image(
                image_dir,
                pair["image_1"],
            )

            bbox_1 = find_matching_bbox(
                xml_dir,
                pair["image_1"],
                pair["species"],
            )

            output_1 = (
                OUTPUT_DIR
                / f"{pair['pair_id']}_img1.jpg"
            )

            create_crop(
                image_1_path,
                bbox_1,
                output_1,
            )

            # ---------------------------------------------
            # Image 2
            # ---------------------------------------------

            image_2_path = find_image(
                image_dir,
                pair["image_2"],
            )

            bbox_2 = find_matching_bbox(
                xml_dir,
                pair["image_2"],
                pair["species"],
            )

            output_2 = (
                OUTPUT_DIR
                / f"{pair['pair_id']}_img2.jpg"
            )

            create_crop(
                image_2_path,
                bbox_2,
                output_2,
            )

            # ---------------------------------------------
            # Metadata
            # ---------------------------------------------

            metadata_rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "dataset": pair["dataset"],
                    "species": pair["species"],
                    "label": 1,
                    "difficulty": pair["difficulty"],
                    "confidence": pair["confidence"],
                    "evidence": pair["evidence"],
                    "image_1": pair["image_1"],
                    "image_2": pair["image_2"],
                    "crop_1": output_1.name,
                    "crop_2": output_2.name,
                    "bbox_1": str(bbox_1),
                    "bbox_2": str(bbox_2),
                }
            )

            processed += 1

            print(
                f"[OK] {pair['pair_id']}"
            )

            print(
                f"     Crop 1: {output_1.name}"
            )

            print(
                f"     Crop 2: {output_2.name}"
            )

        except Exception as exc:

            failed += 1

            print(
                f"[FAILED] {pair['pair_id']}"
            )

            print(
                f"         {exc}"
            )

    # -----------------------------------------------------
    # Write metadata
    # -----------------------------------------------------

    fieldnames = [
        "pair_id",
        "dataset",
        "species",
        "label",
        "difficulty",
        "confidence",
        "evidence",
        "image_1",
        "image_2",
        "crop_1",
        "crop_2",
        "bbox_1",
        "bbox_2",
    ]

    with open(
        METADATA_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            metadata_rows
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 70)

    print(
        "SUMMARY"
    )

    print("=" * 70)

    print(
        f"Pairs processed : {processed}"
    )

    print(
        f"Pairs failed    : {failed}"
    )

    print(
        f"Crops generated : {processed * 2}"
    )

    print()
    print(
        f"Crop directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        f"Metadata file:"
    )

    print(
        METADATA_FILE
    )

    print("=" * 70)


if __name__ == "__main__":
    main()