from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter


# ============================================================
# VanDrishti - Wildlife Dataset Audit
# This script ONLY READS the dataset.
# It does NOT modify, move, rename, or delete files.
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day"
)

IMAGE_DIR = DATASET_PATH / "JPEGImages"
ANNOTATION_DIR = DATASET_PATH / "Annotations"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():

    print("=" * 65)
    print("        VanDrishti - Day DATASET AUDIT")
    print("=" * 65)

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not DATASET_PATH.exists():
        print(f"\nERROR: Dataset path does not exist:")
        print(DATASET_PATH)
        return

    if not IMAGE_DIR.exists():
        print(f"\nERROR: JPEGImages folder not found:")
        print(IMAGE_DIR)
        return

    if not ANNOTATION_DIR.exists():
        print(f"\nERROR: Annotations folder not found:")
        print(ANNOTATION_DIR)
        return

    print(f"\nDataset:")
    print(DATASET_PATH)

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    print("\nScanning images...")

    images = [
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Total images: {len(images):,}")

    # --------------------------------------------------------
    # Find annotations
    # --------------------------------------------------------

    print("Scanning annotations...")

    xml_files = list(ANNOTATION_DIR.glob("*.xml"))

    print(f"Total XML annotations: {len(xml_files):,}")

    # --------------------------------------------------------
    # File matching
    # --------------------------------------------------------

    image_stems = {p.stem for p in images}
    xml_stems = {p.stem for p in xml_files}

    images_without_annotations = image_stems - xml_stems
    annotations_without_images = xml_stems - image_stems

    print("\nFile matching")
    print("-" * 65)
    print(f"Images without annotations: {len(images_without_annotations):,}")
    print(f"Annotations without images: {len(annotations_without_images):,}")

    # --------------------------------------------------------
    # Analyze XML files
    # --------------------------------------------------------

    class_counter = Counter()
    resolution_counter = Counter()
    objects_per_image = Counter()

    total_objects = 0
    invalid_xml = 0

    print("\nAnalyzing annotations...")

    for xml_file in xml_files:

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Image dimensions
            size = root.find("size")

            if size is not None:
                width = size.findtext("width")
                height = size.findtext("height")

                if width and height:
                    resolution = f"{width}x{height}"
                    resolution_counter[resolution] += 1

            # Objects
            objects = root.findall("object")

            objects_per_image[len(objects)] += 1
            total_objects += len(objects)

            for obj in objects:

                name = obj.findtext("name")

                if name:
                    class_counter[name.strip()] += 1

        except Exception as e:

            invalid_xml += 1

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("                    AUDIT RESULTS")
    print("=" * 65)

    print("\n1. DATASET SIZE")
    print("-" * 65)
    print(f"Images              : {len(images):,}")
    print(f"Annotations         : {len(xml_files):,}")
    print(f"Annotated objects   : {total_objects:,}")

    print("\n2. CLASSES / SPECIES")
    print("-" * 65)

    if class_counter:
        print(f"Number of classes   : {len(class_counter)}")

        for class_name, count in class_counter.most_common():
            print(f"{class_name:<30} {count:,}")

    else:
        print("No classes found.")

    print("\n3. IMAGE RESOLUTIONS")
    print("-" * 65)

    for resolution, count in resolution_counter.most_common():
        print(f"{resolution:<20} {count:,}")

    print("\n4. OBJECTS PER IMAGE")
    print("-" * 65)

    for object_count, image_count in sorted(objects_per_image.items()):
        print(
            f"{object_count} object(s) : {image_count:,} image(s)"
        )

    print("\n5. DATA CONSISTENCY")
    print("-" * 65)
    print(
        f"Images without XML       : "
        f"{len(images_without_annotations):,}"
    )
    print(
        f"XML without image        : "
        f"{len(annotations_without_images):,}"
    )
    print(f"Invalid XML files        : {invalid_xml:,}")

    # --------------------------------------------------------
    # Sample annotations
    # --------------------------------------------------------

    print("\n6. SAMPLE ANNOTATIONS")
    print("-" * 65)

    for xml_file in xml_files[:5]:

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            filename = root.findtext("filename")
            objects = root.findall("object")

            print(f"\nFile: {xml_file.name}")
            print(f"Image: {filename}")

            for obj in objects:

                name = obj.findtext("name")

                bbox = obj.find("bndbox")

                if bbox is not None:
                    xmin = bbox.findtext("xmin")
                    ymin = bbox.findtext("ymin")
                    xmax = bbox.findtext("xmax")
                    ymax = bbox.findtext("ymax")

                    print(
                        f"  Class: {name}"
                    )

                    print(
                        f"  Box  : "
                        f"({xmin}, {ymin}) -> "
                        f"({xmax}, {ymax})"
                    )

        except Exception:
            pass

    print("\n" + "=" * 65)
    print("              DATASET AUDIT COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()