from pathlib import Path
import xml.etree.ElementTree as ET
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DAY_ANN = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day\Annotations"
)

NIGHT_ANN = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night\Annotations"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "reid" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "multi_animal_frames.csv"


def inspect_annotations(annotation_dir, dataset_name):

    results = []

    xml_files = list(annotation_dir.glob("*.xml"))

    print(f"\n[{dataset_name}] XML files: {len(xml_files):,}")

    for xml_path in xml_files:

        try:
            root = ET.parse(xml_path).getroot()

        except Exception as e:
            print(f"[WARNING] Could not parse {xml_path.name}: {e}")
            continue

        class_counts = {}

        for obj in root.findall("object"):

            name_element = obj.find("name")

            if name_element is None:
                continue

            species = name_element.text.strip()

            class_counts[species] = class_counts.get(species, 0) + 1

        for species, count in class_counts.items():

            if count >= 2:

                results.append(
                    {
                        "dataset": dataset_name,
                        "image": root.findtext("filename"),
                        "species": species,
                        "instances": count,
                        "annotation": str(xml_path),
                    }
                )

    return results


def main():

    print("=" * 70)
    print("VanDrishti - Same-Species Multi-Animal Frame Search")
    print("=" * 70)

    day_results = inspect_annotations(
        DAY_ANN,
        "day",
    )

    night_results = inspect_annotations(
        NIGHT_ANN,
        "night",
    )

    results = day_results + night_results

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "image",
                "species",
                "instances",
                "annotation",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n===== SUMMARY =====")

    print(
        f"Day same-species multi-animal frames: "
        f"{len(day_results):,}"
    )

    print(
        f"Night same-species multi-animal frames: "
        f"{len(night_results):,}"
    )

    print(
        f"Total candidate frames: "
        f"{len(results):,}"
    )

    print(f"\n[SAVED] {OUTPUT_FILE}")

    print("\nFirst 20 candidates:")

    for item in results[:20]:

        print(
            f"{item['dataset']:5} | "
            f"{item['species']:15} | "
            f"{item['instances']} instances | "
            f"{item['image']}"
        )


if __name__ == "__main__":
    main()