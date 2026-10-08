from pathlib import Path
import xml.etree.ElementTree as ET


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DAY_XML_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day\Annotations"
)

NIGHT_XML_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night\Annotations"
)


# =========================================================
# CANDIDATE PAIRS
# =========================================================

PAIRS = [
    {
        "pair_id": "B001",
        "dataset": "day",
        "image_1": "blackbear104.jpg",
        "image_2": "blackbear106.jpg",
        "species": "BlackBear",
    },
    {
        "pair_id": "B002",
        "dataset": "day",
        "image_1": "blackbear124.jpg",
        "image_2": "blackbear127.jpg",
        "species": "BlackBear",
    },
    {
        "pair_id": "B003",
        "dataset": "day",
        "image_1": "blackbear136.jpg",
        "image_2": "blackbear137.jpg",
        "species": "BlackBear",
    },
    {
        "pair_id": "L001",
        "dataset": "day",
        "image_1": "leopardcat__1146.jpg",
        "image_2": "leopardcat__1152.jpg",
        "species": "LeopardCat",
    },
    {
        "pair_id": "L002",
        "dataset": "day",
        "image_1": "leopardcat__1154.jpg",
        "image_2": "leopardcat__1163.jpg",
        "species": "LeopardCat",
    },
    {
        "pair_id": "L003",
        "dataset": "day",
        "image_1": "leopardcat__1161.jpg",
        "image_2": "leopardcat__1171.jpg",
        "species": "LeopardCat",
    },
    {
        "pair_id": "M001",
        "dataset": "day",
        "image_1": "muskdeer_410.jpg",
        "image_2": "muskdeer_429.jpg",
        "species": "MuskDeer",
    },
    {
        "pair_id": "M002",
        "dataset": "day",
        "image_1": "muskdeer_488.jpg",
        "image_2": "muskdeer_503.jpg",
        "species": "MuskDeer",
    },
    {
        "pair_id": "M003",
        "dataset": "day",
        "image_1": "muskdeer_523.jpg",
        "image_2": "muskdeer_589.jpg",
        "species": "MuskDeer",
    },
    {
        "pair_id": "S001",
        "dataset": "day",
        "image_1": "sikadeer1075.jpg",
        "image_2": "sikadeer1076.jpg",
        "species": "SikaDeer",
    },
]


# =========================================================
# SPECIES NORMALIZATION
# =========================================================

def normalize_species(name):
    """
    Normalize dataset class names for comparison.
    """

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
# XML PATH
# =========================================================

def get_xml_path(dataset, image_name):

    stem = Path(
        image_name
    ).stem

    if dataset == "day":

        xml_dir = DAY_XML_DIR

    elif dataset == "night":

        xml_dir = NIGHT_XML_DIR

    else:

        raise ValueError(
            f"Unknown dataset: {dataset}"
        )

    xml_path = (
        xml_dir
        / f"{stem}.xml"
    )

    return xml_path


# =========================================================
# READ XML
# =========================================================

def read_annotation(
    dataset,
    image_name,
    expected_species,
):

    xml_path = get_xml_path(
        dataset,
        image_name,
    )

    if not xml_path.exists():

        return {
            "exists": False,
            "xml_path": str(xml_path),
            "objects": [],
            "matching_objects": [],
        }

    tree = ET.parse(
        xml_path
    )

    root = tree.getroot()

    objects = []

    for obj in root.findall("object"):

        name_node = obj.find("name")

        if name_node is None:
            continue

        species = (
            name_node.text
            or ""
        ).strip()

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

        objects.append(
            {
                "species": species,
                "bbox": (
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ),
            }
        )

    expected = normalize_species(
        expected_species
    )

    matching_objects = [
        obj
        for obj in objects
        if normalize_species(
            obj["species"]
        ) == expected
    ]

    return {
        "exists": True,
        "xml_path": str(xml_path),
        "objects": objects,
        "matching_objects": matching_objects,
    }


# =========================================================
# PRINT FRAME INFORMATION
# =========================================================

def print_frame(
    dataset,
    image_name,
    expected_species,
):

    result = read_annotation(
        dataset,
        image_name,
        expected_species,
    )

    print()
    print(
        f"Image: {image_name}"
    )

    print(
        f"XML  : {result['xml_path']}"
    )

    if not result["exists"]:

        print(
            "STATUS: XML NOT FOUND"
        )

        return False

    print(
        f"Total annotated objects: "
        f"{len(result['objects'])}"
    )

    if not result["objects"]:

        print(
            "STATUS: NO OBJECTS"
        )

        return False

    print(
        "Annotated objects:"
    )

    for index, obj in enumerate(
        result["objects"],
        start=1,
    ):

        print(
            f"  {index}. "
            f"{obj['species']} "
            f"{obj['bbox']}"
        )

    print(
        f"Matching {expected_species}: "
        f"{len(result['matching_objects'])}"
    )

    if result["matching_objects"]:

        for index, obj in enumerate(
            result["matching_objects"],
            start=1,
        ):

            print(
                f"  MATCH {index}: "
                f"{obj['bbox']}"
            )

        return True

    print(
        "STATUS: EXPECTED SPECIES NOT FOUND"
    )

    return False


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print(
        "VanDrishti - Phase 2B "
        "Annotation Verification"
    )
    print("=" * 70)

    print()

    success_count = 0
    failure_count = 0

    for pair in PAIRS:

        print()
        print("-" * 70)

        print(
            f"PAIR {pair['pair_id']} | "
            f"{pair['dataset']} | "
            f"{pair['species']}"
        )

        print("-" * 70)

        result_1 = print_frame(
            pair["dataset"],
            pair["image_1"],
            pair["species"],
        )

        result_2 = print_frame(
            pair["dataset"],
            pair["image_2"],
            pair["species"],
        )

        if result_1 and result_2:

            success_count += 1

            print()
            print(
                f"[VALID] {pair['pair_id']} "
                f"has usable annotations."
            )

        else:

            failure_count += 1

            print()
            print(
                f"[CHECK REQUIRED] "
                f"{pair['pair_id']}"
            )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Valid pairs        : {success_count}"
    )

    print(
        f"Pairs needing check: {failure_count}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()