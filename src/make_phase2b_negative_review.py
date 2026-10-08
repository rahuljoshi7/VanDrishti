from pathlib import Path
import random
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt


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
    / "outputs"
    / "reid_review"
    / "phase2b_negative"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "phase2b_negative_candidates.jpg"
)


# =========================================================
# SETTINGS
# =========================================================

TARGET_PER_DATASET = 10

RANDOM_SEED = 42


# =========================================================
# PATHS
# =========================================================

def get_paths(dataset):

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
# NORMALIZE SPECIES
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
        "amurleopard": "leopard",
    }

    return aliases.get(
        name,
        name,
    )


# =========================================================
# READ XML
# =========================================================

def read_objects(
    xml_path,
):

    tree = ET.parse(
        xml_path
    )

    root = tree.getroot()

    objects = []

    for obj in root.findall(
        "object"
    ):

        name_node = obj.find(
            "name"
        )

        bbox = obj.find(
            "bndbox"
        )

        if (
            name_node is None
            or bbox is None
        ):
            continue

        species = (
            name_node.text
            or ""
        ).strip()

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
                "normalized_species":
                    normalize_species(
                        species
                    ),
                "bbox": (
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ),
            }
        )

    return objects


# =========================================================
# FIND MULTI-ANIMAL CANDIDATES
# =========================================================

def collect_candidates(
    dataset,
):

    image_dir, xml_dir = (
        get_paths(dataset)
    )

    candidates = []

    xml_files = sorted(
        xml_dir.glob("*.xml")
    )

    for xml_path in xml_files:

        objects = read_objects(
            xml_path
        )

        by_species = {}

        for obj in objects:

            species = obj[
                "normalized_species"
            ]

            by_species.setdefault(
                species,
                [],
            ).append(
                obj
            )

        for species, group in (
            by_species.items()
        ):

            if len(group) < 2:
                continue

            image_path = (
                image_dir
                / f"{xml_path.stem}.jpg"
            )

            if not image_path.exists():

                # Try case-insensitive lookup.
                matches = list(
                    image_dir.glob(
                        f"{xml_path.stem}.*"
                    )
                )

                if not matches:
                    continue

                image_path = matches[0]

            candidates.append(
                {
                    "dataset": dataset,
                    "image": image_path.name,
                    "xml": xml_path,
                    "species": species,
                    "objects": group,
                }
            )

    return candidates


# =========================================================
# CROP
# =========================================================

def crop_bbox(
    image,
    bbox,
):

    h, w = image.shape[:2]

    xmin, ymin, xmax, ymax = bbox

    xmin = max(
        0,
        min(xmin, w - 1),
    )

    ymin = max(
        0,
        min(ymin, h - 1),
    )

    xmax = max(
        xmin + 1,
        min(xmax, w),
    )

    ymax = max(
        ymin + 1,
        min(ymax, h),
    )

    return image[
        ymin:ymax,
        xmin:xmax,
    ]


# =========================================================
# CONTACT SHEET
# =========================================================

def create_review_sheet(
    candidates,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = 2
    rows = len(candidates)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            12,
            max(4 * rows, 8),
        ),
    )

    if rows == 1:
        axes = [axes]

    for index, candidate in enumerate(
        candidates
    ):

        dataset = candidate[
            "dataset"
        ]

        image_name = candidate[
            "image"
        ]

        species = candidate[
            "species"
        ]

        image_dir, _ = (
            get_paths(dataset)
        )

        image_path = (
            image_dir
            / image_name
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        objects = candidate[
            "objects"
        ]

        # -------------------------------------------------
        # Full image
        # -------------------------------------------------

        axes[index][0].imshow(
            image
        )

        axes[index][0].set_title(
            f"{dataset} | "
            f"{image_name}\n"
            f"Species: {species} | "
            f"{len(objects)} individuals",
            fontsize=9,
        )

        axes[index][0].axis(
            "off"
        )

        # -------------------------------------------------
        # Two individual crops
        # -------------------------------------------------

        crop1 = crop_bbox(
            image,
            objects[0]["bbox"],
        )

        crop2 = crop_bbox(
            image,
            objects[1]["bbox"],
        )

        # Place two crops side-by-side
        # in a single visualization.
        combined_width = (
            crop1.shape[1]
            + crop2.shape[1]
        )

        combined_height = max(
            crop1.shape[0],
            crop2.shape[0],
        )

        combined = (
            255
            * (
                __import__(
                    "numpy"
                ).ones(
                    (
                        combined_height,
                        combined_width,
                        3,
                    ),
                    dtype="uint8",
                )
            )
        )

        combined[
            :crop1.shape[0],
            :crop1.shape[1],
        ] = crop1

        combined[
            :crop2.shape[0],
            crop1.shape[1]:
            crop1.shape[1]
            + crop2.shape[1],
        ] = crop2

        axes[index][1].imshow(
            combined.astype(
                "uint8"
            )
        )

        axes[index][1].set_title(
            "Individual A | Individual B\n"
            "Strong DIFFERENT evidence",
            fontsize=9,
        )

        axes[index][1].axis(
            "off"
        )

    fig.suptitle(
        "VanDrishti Phase 2B — "
        "Candidate DIFFERENT Pairs\n"
        "Two same-species animals in the same frame",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FILE,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[SAVED] {OUTPUT_FILE}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)

    print(
        "VanDrishti - Phase 2B "
        "Negative Pair Candidate Generation"
    )

    print("=" * 70)

    random.seed(
        RANDOM_SEED
    )

    day_candidates = (
        collect_candidates(
            "day"
        )
    )

    night_candidates = (
        collect_candidates(
            "night"
        )
    )

    print()
    print(
        f"Day multi-animal candidates   : "
        f"{len(day_candidates)}"
    )

    print(
        f"Night multi-animal candidates : "
        f"{len(night_candidates)}"
    )

    # -----------------------------------------------------
    # Sample candidates
    # -----------------------------------------------------

    day_sample = random.sample(
        day_candidates,
        min(
            TARGET_PER_DATASET,
            len(day_candidates),
        ),
    )

    night_sample = random.sample(
        night_candidates,
        min(
            TARGET_PER_DATASET,
            len(night_candidates),
        ),
    )

    selected = (
        day_sample
        + night_sample
    )

    print()
    print(
        f"Review candidates: "
        f"{len(selected)}"
    )

    for index, candidate in enumerate(
        selected,
        start=1,
    ):

        print(
            f"{index:02d}. "
            f"{candidate['dataset']} | "
            f"{candidate['image']} | "
            f"{candidate['species']}"
        )

    create_review_sheet(
        selected
    )

    print()
    print("=" * 70)

    print(
        "Candidate generation complete."
    )

    print()
    print(
        "Review image:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 70)


if __name__ == "__main__":
    main()