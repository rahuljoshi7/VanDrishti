from pathlib import Path
import xml.etree.ElementTree as ET
import os
import shutil


# ============================================================
# VanDrishti - Pascal VOC to YOLO Dataset Converter
# ============================================================
#
# Purpose:
#   Convert Pascal VOC annotations into YOLO annotation format.
#
# IMPORTANT:
#   - Original Day/Night datasets are READ ONLY.
#   - Original images and XML files are never modified.
#   - Generated YOLO data is stored inside the VanDrishti project.
#   - Images are hard-linked when possible to save disk space.
#
# Dataset:
#   Day   -> 15,313 images / 16,247 objects
#   Night -> 10,344 images / 10,699 objects
#
# Expected total:
#   25,657 images
#   26,946 objects
#   17 classes
#
# ============================================================


# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti"
)


# ------------------------------------------------------------
# ORIGINAL DATASETS
# READ ONLY
# ------------------------------------------------------------

DATASETS = {

    "day": Path(
        r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day"
    ),

    "night": Path(
        r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night"
    ),

}


# ------------------------------------------------------------
# OUTPUT DATASET
# ------------------------------------------------------------

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "yolo"
)


# ------------------------------------------------------------
# VAN DRISHTI CLASS LIST
# ------------------------------------------------------------
#
# YOLO class IDs start from 0.
#
# 0  = RaccoonDog
# 1  = Hare
# 2  = MuskDeer
# ...
# 16 = Cow
#
# ------------------------------------------------------------

CLASS_NAMES = [
    "RaccoonDog",
    "Hare",
    "MuskDeer",
    "LeopardCat",
    "RedFox",
    "WildBoar",
    "SikaDeer",
    "RoeDeer",
    "AmurTiger",
    "Weasel",
    "Leopard",
    "Sable",
    "BlackBear",
    "Badger",
    "Y.T.Marten",
    "Dog",
    "Cow",
]


CLASS_TO_ID = {
    class_name: class_id
    for class_id, class_name in enumerate(CLASS_NAMES)
}


# ------------------------------------------------------------
# EXPECTED DATASET COUNTS
# ------------------------------------------------------------

EXPECTED_COUNTS = {

    "day": {
        "images": 15313,
        "objects": 16247,
    },

    "night": {
        "images": 10344,
        "objects": 10699,
    },

}


# ============================================================
# FUNCTION: CONVERT VOC BOUNDING BOX TO YOLO
# ============================================================

def convert_bbox(size, bbox):
    """
    Convert Pascal VOC bounding box:

        xmin, ymin, xmax, ymax

    into YOLO format:

        x_center, y_center, width, height

    YOLO coordinates are normalized between 0 and 1.
    """

    image_width, image_height = size

    xmin, ymin, xmax, ymax = bbox

    # --------------------------------------------------------
    # Clamp coordinates to image boundaries
    # --------------------------------------------------------

    xmin = max(0, min(xmin, image_width))
    xmax = max(0, min(xmax, image_width))

    ymin = max(0, min(ymin, image_height))
    ymax = max(0, min(ymax, image_height))

    # --------------------------------------------------------
    # Calculate width and height
    # --------------------------------------------------------

    box_width = xmax - xmin
    box_height = ymax - ymin

    # Invalid bounding box
    if box_width <= 0 or box_height <= 0:
        return None

    # --------------------------------------------------------
    # Calculate center
    # --------------------------------------------------------

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    x_center = x_center / image_width
    y_center = y_center / image_height

    box_width = box_width / image_width
    box_height = box_height / image_height

    return (
        x_center,
        y_center,
        box_width,
        box_height,
    )


# ============================================================
# FUNCTION: CREATE HARDLINK OR COPY
# ============================================================

def create_hardlink_or_copy(source, destination):
    """
    Create a hardlink to the original image whenever possible.

    Hardlinks allow us to use the same physical image data
    without creating another full copy of the large dataset.

    If hardlink creation fails, the image is copied instead.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # If destination already exists
    if destination.exists():

        return "existing"

    try:

        # ----------------------------------------------------
        # Try hardlink first
        # ----------------------------------------------------

        os.link(
            source,
            destination
        )

        return "hardlink"

    except OSError:

        # ----------------------------------------------------
        # Fallback to normal copy
        # ----------------------------------------------------

        shutil.copy2(
            source,
            destination
        )

        return "copy"


# ============================================================
# FUNCTION: FIND IMAGE FILES
# ============================================================

def find_image_files(image_dir):
    """
    Find all JPEG images without duplicate counting.

    Important:
        Path.glob("*.jpg") and Path.glob("*.JPG") can refer
        to the same Windows files.

    Therefore we scan the directory once and check:

        file.suffix.lower()

    This prevents the 2x counting problem.
    """

    valid_extensions = {
        ".jpg",
        ".jpeg",
    }

    image_files = []

    for path in image_dir.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() in valid_extensions:

            image_files.append(path)

    return sorted(
        image_files,
        key=lambda p: p.name.lower()
    )


# ============================================================
# FUNCTION: CLEAN OUTPUT DATASET
# ============================================================

def clean_output_dataset():
    """
    Delete only the previously generated YOLO dataset.

    IMPORTANT:
        This does NOT touch the original datasets.

    It only removes:

        VanDrishti/data/processed/yolo
    """

    if OUTPUT_ROOT.exists():

        print()
        print("=" * 60)
        print("Cleaning previous generated YOLO dataset")
        print("=" * 60)

        print(
            f"Removing: {OUTPUT_ROOT}"
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

        print("Previous generated data removed.")

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FUNCTION: PROCESS ONE DATASET
# ============================================================

def process_dataset(
    dataset_name,
    dataset_root
):

    # --------------------------------------------------------
    # Input directories
    # --------------------------------------------------------

    image_dir = (
        dataset_root
        / "JPEGImages"
    )

    annotation_dir = (
        dataset_root
        / "Annotations"
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    output_image_dir = (
        OUTPUT_ROOT
        / dataset_name
        / "images"
    )

    output_label_dir = (
        OUTPUT_ROOT
        / dataset_name
        / "labels"
    )

    output_image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Verify input directories
    # --------------------------------------------------------

    if not image_dir.exists():

        print(
            f"[ERROR] Image directory not found:"
        )

        print(image_dir)

        return {
            "images": 0,
            "objects": 0,
            "skipped": 0,
        }

    if not annotation_dir.exists():

        print(
            f"[ERROR] Annotation directory not found:"
        )

        print(annotation_dir)

        return {
            "images": 0,
            "objects": 0,
            "skipped": 0,
        }

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_files = find_image_files(
        image_dir
    )

    print()
    print("=" * 60)
    print(
        f"Processing {dataset_name.upper()} dataset"
    )
    print("=" * 60)

    print(
        f"Images found: {len(image_files)}"
    )

    # --------------------------------------------------------
    # Expected count
    # --------------------------------------------------------

    expected_images = EXPECTED_COUNTS[
        dataset_name
    ]["images"]

    if len(image_files) != expected_images:

        print()
        print(
            "[WARNING] Image count does not match "
            "the previous dataset audit."
        )

        print(
            f"Expected : {expected_images}"
        )

        print(
            f"Found    : {len(image_files)}"
        )

        print()

    else:

        print(
            f"Image count verified: {expected_images}"
        )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    converted = 0
    skipped = 0

    total_objects = 0
    invalid_boxes = 0

    hardlinks = 0
    copies = 0
    existing = 0

    unknown_classes = set()

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        # ----------------------------------------------------
        # Corresponding XML
        # ----------------------------------------------------

        xml_path = (
            annotation_dir
            / f"{image_path.stem}.xml"
        )

        # ----------------------------------------------------
        # Check XML exists
        # ----------------------------------------------------

        if not xml_path.exists():

            print(
                f"[WARNING] Missing annotation:"
                f" {xml_path.name}"
            )

            skipped += 1

            continue

        try:

            # ------------------------------------------------
            # Read XML
            # ------------------------------------------------

            tree = ET.parse(
                xml_path
            )

            root = tree.getroot()

            # ------------------------------------------------
            # Read image size
            # ------------------------------------------------

            size_node = root.find(
                "size"
            )

            if size_node is None:

                print(
                    f"[WARNING] Missing image size:"
                    f" {xml_path.name}"
                )

                skipped += 1

                continue

            width_text = size_node.findtext(
                "width"
            )

            height_text = size_node.findtext(
                "height"
            )

            if width_text is None or height_text is None:

                print(
                    f"[WARNING] Invalid image dimensions:"
                    f" {xml_path.name}"
                )

                skipped += 1

                continue

            width = int(
                width_text
            )

            height = int(
                height_text
            )

            if width <= 0 or height <= 0:

                print(
                    f"[WARNING] Invalid image size:"
                    f" {xml_path.name}"
                )

                skipped += 1

                continue

            # ------------------------------------------------
            # Store YOLO annotation lines
            # ------------------------------------------------

            yolo_lines = []

            # ------------------------------------------------
            # Read objects
            # ------------------------------------------------

            for obj in root.findall(
                "object"
            ):

                class_name = obj.findtext(
                    "name"
                )

                # --------------------------------------------
                # Check class
                # --------------------------------------------

                if class_name not in CLASS_TO_ID:

                    unknown_classes.add(
                        class_name
                    )

                    continue

                # --------------------------------------------
                # Find bounding box
                # --------------------------------------------

                bbox_node = obj.find(
                    "bndbox"
                )

                if bbox_node is None:

                    invalid_boxes += 1

                    continue

                # --------------------------------------------
                # Read bounding coordinates
                # --------------------------------------------

                xmin_text = bbox_node.findtext(
                    "xmin"
                )

                ymin_text = bbox_node.findtext(
                    "ymin"
                )

                xmax_text = bbox_node.findtext(
                    "xmax"
                )

                ymax_text = bbox_node.findtext(
                    "ymax"
                )

                if None in (
                    xmin_text,
                    ymin_text,
                    xmax_text,
                    ymax_text,
                ):

                    invalid_boxes += 1

                    continue

                xmin = float(
                    xmin_text
                )

                ymin = float(
                    ymin_text
                )

                xmax = float(
                    xmax_text
                )

                ymax = float(
                    ymax_text
                )

                # --------------------------------------------
                # Convert VOC → YOLO
                # --------------------------------------------

                converted_bbox = convert_bbox(
                    (width, height),
                    (
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                    )
                )

                if converted_bbox is None:

                    invalid_boxes += 1

                    continue

                (
                    x_center,
                    y_center,
                    box_width,
                    box_height,
                ) = converted_bbox

                # --------------------------------------------
                # Class ID
                # --------------------------------------------

                class_id = CLASS_TO_ID[
                    class_name
                ]

                # --------------------------------------------
                # YOLO annotation
                # --------------------------------------------

                yolo_line = (
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}"
                )

                yolo_lines.append(
                    yolo_line
                )

                total_objects += 1

            # ------------------------------------------------
            # Create label file
            # ------------------------------------------------

            label_path = (
                output_label_dir
                / f"{image_path.stem}.txt"
            )

            with open(
                label_path,
                "w",
                encoding="utf-8"
            ) as label_file:

                label_file.write(
                    "\n".join(
                        yolo_lines
                    )
                )

            # ------------------------------------------------
            # Link/copy image
            # ------------------------------------------------

            output_image_path = (
                output_image_dir
                / image_path.name
            )

            result = create_hardlink_or_copy(
                image_path,
                output_image_path
            )

            if result == "hardlink":

                hardlinks += 1

            elif result == "copy":

                copies += 1

            elif result == "existing":

                existing += 1

            converted += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if index % 500 == 0:

                percentage = (
                    index
                    / len(image_files)
                    * 100
                )

                print(
                    f"Progress: "
                    f"{index}/{len(image_files)} "
                    f"({percentage:.1f}%)"
                )

        except ET.ParseError as error:

            print(
                f"[ERROR] Invalid XML:"
                f" {xml_path.name}"
            )

            print(
                f"        {error}"
            )

            skipped += 1

        except Exception as error:

            print(
                f"[ERROR] Failed:"
                f" {image_path.name}"
            )

            print(
                f"        {error}"
            )

            skipped += 1

    # --------------------------------------------------------
    # Dataset result
    # --------------------------------------------------------

    print()
    print(
        f"{dataset_name.upper()} conversion complete."
    )

    print(
        f"Converted images : {converted}"
    )

    print(
        f"Skipped images   : {skipped}"
    )

    print(
        f"Objects          : {total_objects}"
    )

    print(
        f"Hardlinks        : {hardlinks}"
    )

    print(
        f"Copies            : {copies}"
    )

    print(
        f"Existing          : {existing}"
    )

    print(
        f"Invalid boxes    : {invalid_boxes}"
    )

    # --------------------------------------------------------
    # Object count verification
    # --------------------------------------------------------

    expected_objects = EXPECTED_COUNTS[
        dataset_name
    ]["objects"]

    print()

    if total_objects == expected_objects:

        print(
            f"[OK] Object count verified: "
            f"{total_objects}"
        )

    else:

        print(
            "[WARNING] Object count differs "
            "from previous audit."
        )

        print(
            f"Expected : {expected_objects}"
        )

        print(
            f"Found    : {total_objects}"
        )

    # --------------------------------------------------------
    # Unknown classes
    # --------------------------------------------------------

    if unknown_classes:

        print()
        print(
            "UNKNOWN CLASSES FOUND:"
        )

        for class_name in sorted(
            unknown_classes
        ):

            print(
                f"  - {class_name}"
            )

    else:

        print(
            "[OK] No unknown classes found."
        )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "images": converted,
        "objects": total_objects,
        "skipped": skipped,
    }


# ============================================================
# FUNCTION: CREATE DATA.YAML
# ============================================================

def create_dataset_yaml():
    """
    Create the YOLO dataset configuration.

    NOTE:
        We intentionally DO NOT point train and validation
        to the same images.

        The actual train/validation split will be created
        in the next step using a sequence-aware strategy.
    """

    yaml_path = (
        OUTPUT_ROOT
        / "data.yaml"
    )

    yaml_content = f"""# ============================================================
# VanDrishti YOLO Dataset Configuration
# ============================================================

path: {OUTPUT_ROOT.as_posix()}

# These directories will be created during the
# sequence-aware train/validation split.

train: train/images
val: val/images

names:
"""

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        yaml_content += (
            f"  {class_id}: "
            f"{class_name}\n"
        )

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as yaml_file:

        yaml_file.write(
            yaml_content
        )

    print()
    print("=" * 60)
    print("Created YOLO data.yaml")
    print("=" * 60)

    print(
        yaml_path
    )


# ============================================================
# FUNCTION: CREATE CLASS LIST
# ============================================================

def create_class_list():
    """
    Create a simple classes.txt file.

    This is useful for debugging and documentation.
    """

    class_file = (
        OUTPUT_ROOT
        / "classes.txt"
    )

    with open(
        class_file,
        "w",
        encoding="utf-8"
    ) as f:

        for class_id, class_name in enumerate(
            CLASS_NAMES
        ):

            f.write(
                f"{class_id}: {class_name}\n"
            )

    print(
        f"Created class list:"
    )

    print(
        class_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("VANDRISHTI VOC → YOLO CONVERTER")
    print("=" * 60)

    print()
    print(
        "Original datasets:"
    )

    print(
        "  DAY  : "
        r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day"
    )

    print(
        "  NIGHT: "
        r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night"
    )

    print()
    print(
        "Output dataset:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "The original datasets will NOT be modified."
    )

    # --------------------------------------------------------
    # Clean only generated output
    # --------------------------------------------------------

    clean_output_dataset()

    # --------------------------------------------------------
    # Process Day
    # --------------------------------------------------------

    day_result = process_dataset(
        "day",
        DATASETS["day"]
    )

    # --------------------------------------------------------
    # Process Night
    # --------------------------------------------------------

    night_result = process_dataset(
        "night",
        DATASETS["night"]
    )

    # --------------------------------------------------------
    # Combined totals
    # --------------------------------------------------------

    total_images = (
        day_result["images"]
        + night_result["images"]
    )

    total_objects = (
        day_result["objects"]
        + night_result["objects"]
    )

    total_skipped = (
        day_result["skipped"]
        + night_result["skipped"]
    )

    # --------------------------------------------------------
    # Create configuration files
    # --------------------------------------------------------

    create_dataset_yaml()

    create_class_list()

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL DATASET VERIFICATION")
    print("=" * 60)

    print(
        f"Day images      : "
        f"{day_result['images']}"
    )

    print(
        f"Night images    : "
        f"{night_result['images']}"
    )

    print(
        f"Total images    : "
        f"{total_images}"
    )

    print()

    print(
        f"Day objects     : "
        f"{day_result['objects']}"
    )

    print(
        f"Night objects   : "
        f"{night_result['objects']}"
    )

    print(
        f"Total objects   : "
        f"{total_objects}"
    )

    print()

    print(
        f"Total skipped   : "
        f"{total_skipped}"
    )

    # --------------------------------------------------------
    # Expected totals
    # --------------------------------------------------------

    expected_total_images = (
        EXPECTED_COUNTS["day"]["images"]
        + EXPECTED_COUNTS["night"]["images"]
    )

    expected_total_objects = (
        EXPECTED_COUNTS["day"]["objects"]
        + EXPECTED_COUNTS["night"]["objects"]
    )

    print()
    print(
        f"Expected images : "
        f"{expected_total_images}"
    )

    print(
        f"Expected objects: "
        f"{expected_total_objects}"
    )

    # --------------------------------------------------------
    # Check totals
    # --------------------------------------------------------

    print()

    if (
        total_images
        == expected_total_images
    ):

        print(
            "[SUCCESS] Image count matches audit."
        )

    else:

        print(
            "[WARNING] Image count does NOT "
            "match audit."
        )

    if (
        total_objects
        == expected_total_objects
    ):

        print(
            "[SUCCESS] Object count matches audit."
        )

    else:

        print(
            "[WARNING] Object count does NOT "
            "match audit."
        )

    if total_skipped == 0:

        print(
            "[SUCCESS] No images were skipped."
        )

    else:

        print(
            f"[WARNING] {total_skipped} "
            f"images were skipped."
        )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CONVERSION FINISHED")
    print("=" * 60)

    print()

    print(
        "Original datasets were NOT modified."
    )

    print()

    print(
        "YOLO dataset created at:"
    )

    print(
        OUTPUT_ROOT
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Train/validation splitting has NOT "
        "been performed yet."
    )

    print(
        "We will perform a sequence-aware split "
        "before YOLO training."
    )

    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()