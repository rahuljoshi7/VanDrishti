from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter
import re


# ============================================================
# VanDrishti - Dataset Metadata & Filename Audit
# READ-ONLY: Does not modify the dataset.
# ============================================================

DATASETS = {
    "DAY": Path(
        r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day"
    ),
    "NIGHT": Path(
        r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night"
    ),
}


def inspect_xml(xml_file):
    """Extract useful metadata from one VOC XML file."""

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        result = {
            "filename": root.findtext("filename"),
            "folder": root.findtext("folder"),
            "database": root.findtext("source/database")
            if root.find("source") is not None else None,
            "objects": [],
        }

        for obj in root.findall("object"):
            name = obj.findtext("name")

            result["objects"].append(name)

        return result

    except Exception as e:
        return None


def filename_pattern_analysis(image_dir):

    patterns = Counter()
    examples = {}

    images = list(image_dir.glob("*"))

    for image in images:

        if not image.is_file():
            continue

        stem = image.stem

        # Remove trailing number.
        match = re.match(r"^(.*?)[_-](\d+)$", stem)

        if match:
            prefix = match.group(1)
            patterns[prefix] += 1

            if prefix not in examples:
                examples[prefix] = []

            if len(examples[prefix]) < 5:
                examples[prefix].append(stem)

        else:
            patterns["NO_TRAILING_NUMBER"] += 1

            if "NO_TRAILING_NUMBER" not in examples:
                examples["NO_TRAILING_NUMBER"] = []

            if len(examples["NO_TRAILING_NUMBER"]) < 5:
                examples["NO_TRAILING_NUMBER"].append(stem)

    return patterns, examples


def analyze_dataset(name, dataset_path):

    print("\n")
    print("=" * 70)
    print(f"                    {name} DATASET")
    print("=" * 70)

    image_dir = dataset_path / "JPEGImages"
    annotation_dir = dataset_path / "Annotations"

    if not image_dir.exists():
        print("ERROR: JPEGImages folder not found.")
        return

    if not annotation_dir.exists():
        print("ERROR: Annotations folder not found.")
        return

    # --------------------------------------------------------
    # Filename analysis
    # --------------------------------------------------------

    print("\n1. FILENAME PATTERN ANALYSIS")
    print("-" * 70)

    patterns, examples = filename_pattern_analysis(image_dir)

    print(f"Unique filename prefixes: {len(patterns)}")

    for prefix, count in patterns.most_common():

        print(f"\n{prefix}: {count:,} images")

        if prefix in examples:
            print("Examples:")

            for example in examples[prefix]:
                print(f"  {example}")

    # --------------------------------------------------------
    # XML metadata analysis
    # --------------------------------------------------------

    print("\n2. XML METADATA ANALYSIS")
    print("-" * 70)

    xml_files = list(annotation_dir.glob("*.xml"))

    folder_values = Counter()
    database_values = Counter()
    filename_values = Counter()
    class_values = Counter()

    no_folder = 0
    no_database = 0
    no_filename = 0

    sample_xmls = xml_files[:20]

    for xml_file in sample_xmls:

        data = inspect_xml(xml_file)

        if data is None:
            continue

        if data["folder"]:
            folder_values[data["folder"]] += 1
        else:
            no_folder += 1

        if data["database"]:
            database_values[data["database"]] += 1
        else:
            no_database += 1

        if data["filename"]:
            filename_values[data["filename"]] += 1
        else:
            no_filename += 1

        for obj in data["objects"]:
            if obj:
                class_values[obj] += 1

    print("\nFolder values found:")
    if folder_values:
        for value, count in folder_values.items():
            print(f"  {value}: {count}")
    else:
        print("  None")

    print("\nDatabase values found:")
    if database_values:
        for value, count in database_values.items():
            print(f"  {value}: {count}")
    else:
        print("  None")

    print("\nMissing metadata in first 20 XML files:")
    print(f"  Missing folder  : {no_folder}")
    print(f"  Missing database: {no_database}")
    print(f"  Missing filename: {no_filename}")

    # --------------------------------------------------------
    # Filename-number analysis
    # --------------------------------------------------------

    print("\n3. FILENAME NUMBER ANALYSIS")
    print("-" * 70)

    numbered = 0
    numbers = []

    for image in image_dir.iterdir():

        if not image.is_file():
            continue

        match = re.search(r"[_-](\d+)$", image.stem)

        if match:
            numbered += 1
            numbers.append(int(match.group(1)))

    print(f"Images ending with a number: {numbered:,}")

    if numbers:
        print(f"Smallest number: {min(numbers)}")
        print(f"Largest number : {max(numbers)}")
        print(f"Unique numbers : {len(set(numbers)):,}")

    # --------------------------------------------------------
    # Sample XML
    # --------------------------------------------------------

    print("\n4. SAMPLE XML METADATA")
    print("-" * 70)

    for xml_file in sample_xmls[:3]:

        data = inspect_xml(xml_file)

        if data:
            print(f"\nXML: {xml_file.name}")
            print(f"  Filename : {data['filename']}")
            print(f"  Folder   : {data['folder']}")
            print(f"  Database : {data['database']}")
            print(f"  Classes  : {data['objects']}")

    print("\n" + "=" * 70)


def main():

    print("=" * 70)
    print("       VanDrishti - METADATA & FILENAME AUDIT")
    print("=" * 70)

    for name, path in DATASETS.items():
        analyze_dataset(name, path)

    print("\n")
    print("=" * 70)
    print("                 AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()