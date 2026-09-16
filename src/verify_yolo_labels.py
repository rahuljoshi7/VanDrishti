from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti"
)

DAY_ROOT = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day"
)

NIGHT_ROOT = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night"
)

YOLO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "yolo"
)


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
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}


def verify_sample(dataset_name, dataset_root, image_name):

    print()
    print("=" * 70)
    print(f"{dataset_name.upper()} - {image_name}")
    print("=" * 70)

    xml_path = (
        dataset_root
        / "Annotations"
        / f"{Path(image_name).stem}.xml"
    )

    label_path = (
        YOLO_ROOT
        / dataset_name
        / "labels"
        / f"{Path(image_name).stem}.txt"
    )

    if not xml_path.exists():
        print(f"[ERROR] XML not found: {xml_path}")
        return

    if not label_path.exists():
        print(f"[ERROR] YOLO label not found: {label_path}")
        return

    # --------------------------------------------------------
    # Read XML
    # --------------------------------------------------------

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")

    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    print(f"Image size: {width} x {height}")

    print()
    print("ORIGINAL VOC ANNOTATION")
    print("-" * 70)

    xml_objects = []

    for obj in root.findall("object"):

        class_name = obj.findtext("name")

        bbox = obj.find("bndbox")

        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))

        xml_objects.append(
            (
                class_name,
                xmin,
                ymin,
                xmax,
                ymax,
            )
        )

        print(
            f"{class_name:15s} "
            f"VOC bbox = "
            f"({xmin:.1f}, {ymin:.1f}, "
            f"{xmax:.1f}, {ymax:.1f})"
        )

    # --------------------------------------------------------
    # Read YOLO
    # --------------------------------------------------------

    print()
    print("GENERATED YOLO ANNOTATION")
    print("-" * 70)

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    yolo_objects = []

    for line in lines:

        values = line.split()

        if len(values) != 5:

            print(
                f"[ERROR] Invalid YOLO line:"
                f" {line}"
            )

            continue

        class_id = int(values[0])

        x_center = float(values[1])
        y_center = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        class_name = CLASS_NAMES[class_id]

        # Convert YOLO back to VOC pixels
        xmin = (
            x_center - box_width / 2
        ) * width

        ymin = (
            y_center - box_height / 2
        ) * height

        xmax = (
            x_center + box_width / 2
        ) * width

        ymax = (
            y_center + box_height / 2
        ) * height

        yolo_objects.append(
            (
                class_name,
                xmin,
                ymin,
                xmax,
                ymax,
            )
        )

        print(
            f"{class_name:15s} "
            f"YOLO = "
            f"{x_center:.6f}, "
            f"{y_center:.6f}, "
            f"{box_width:.6f}, "
            f"{box_height:.6f}"
        )

        print(
            f"{'':15s} "
            f"Back-converted VOC = "
            f"({xmin:.1f}, {ymin:.1f}, "
            f"{xmax:.1f}, {ymax:.1f})"
        )

    # --------------------------------------------------------
    # Compare number of objects
    # --------------------------------------------------------

    print()
    print("VERIFICATION")
    print("-" * 70)

    print(
        f"VOC objects : {len(xml_objects)}"
    )

    print(
        f"YOLO objects: {len(yolo_objects)}"
    )

    if len(xml_objects) == len(yolo_objects):

        print(
            "[OK] Object count matches."
        )

    else:

        print(
            "[ERROR] Object count does NOT match."
        )


def main():

    print()
    print("=" * 70)
    print("VANDRISHTI - YOLO LABEL VERIFICATION")
    print("=" * 70)

    # A few representative samples
    samples = [

        (
            "day",
            DAY_ROOT,
            "AmurTiger_0.jpg",
        ),

        (
            "day",
            DAY_ROOT,
            "SikaDeer_0.jpg",
        ),

        (
            "night",
            NIGHT_ROOT,
            "Hare_516.jpg",
        ),

        (
            "night",
            NIGHT_ROOT,
            "AmurLeopard_1187.jpg",
        ),

    ]

    for dataset_name, dataset_root, image_name in samples:

        verify_sample(
            dataset_name,
            dataset_root,
            image_name,
        )

    print()
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()