from pathlib import Path
from collections import Counter, defaultdict
import random
import shutil
import re
import os


# ============================================================
# VanDrishti - Sequence-Aware + Class-Balanced Train/Val Split
# ============================================================
#
# Goals:
#   1. Keep short sequences together.
#   2. Split very large sequences into contiguous chunks.
#   3. Target approximately 80% train / 20% validation.
#   4. Keep every class represented in validation.
#   5. Keep the split reproducible.
#
# IMPORTANT:
#   Original VOC data is never modified.
#
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti"
)

YOLO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "yolo"
)

SOURCE_DATASETS = {
    "day": YOLO_ROOT / "day",
    "night": YOLO_ROOT / "night",
}

TRAIN_ROOT = YOLO_ROOT / "train"
VAL_ROOT = YOLO_ROOT / "val"


# ------------------------------------------------------------
# Split settings
# ------------------------------------------------------------

TRAIN_RATIO = 0.80
VAL_RATIO = 0.20

RANDOM_SEED = 42

# Sequences longer than this may be divided into chunks.
# This prevents a 765-image Hare sequence from destroying
# the class balance.
MAX_SEQUENCE_CHUNK = 50


# ------------------------------------------------------------
# Classes
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


# ============================================================
# Filename helpers
# ============================================================

def extract_numeric_suffix(filename):
    """
    Example:

        Hare_516.jpg
        -> 516
    """

    stem = Path(filename).stem

    match = re.search(
        r"(\d+)$",
        stem
    )

    if match:
        return int(match.group(1))

    return None


def extract_group_name(filename):
    """
    Example:

        Hare_516.jpg
        -> Hare_
    """

    stem = Path(filename).stem

    match = re.match(
        r"^(.*?)(\d+)$",
        stem
    )

    if match:
        return match.group(1)

    return stem


# ============================================================
# Read classes from YOLO label
# ============================================================

def get_classes_from_label(label_path):
    """
    Return all class IDs present in a YOLO label.
    """

    classes = set()

    if not label_path.exists():
        return classes

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])

                if 0 <= class_id < len(CLASS_NAMES):
                    classes.add(class_id)

            except ValueError:
                continue

    return classes


# ============================================================
# Build consecutive sequences
# ============================================================

def build_sequences(image_files):
    """
    Build consecutive filename sequences.

    Same prefix + consecutive numbers = sequence.

    Large sequences are later divided into chunks.
    """

    prefix_groups = defaultdict(list)

    for image_path in image_files:

        prefix = extract_group_name(
            image_path.name
        )

        number = extract_numeric_suffix(
            image_path.name
        )

        prefix_groups[
            (prefix, number is not None)
        ].append(
            image_path
        )

    sequences = []

    for (
        key,
        files
    ) in prefix_groups.items():

        prefix, has_number = key

        # ----------------------------------------------------
        # No numeric suffix
        # ----------------------------------------------------

        if not has_number:

            for image_path in files:
                sequences.append(
                    [image_path]
                )

            continue

        # ----------------------------------------------------
        # Sort numerically
        # ----------------------------------------------------

        files = sorted(
            files,
            key=lambda p: extract_numeric_suffix(
                p.name
            )
        )

        current = []
        previous_number = None

        for image_path in files:

            current_number = (
                extract_numeric_suffix(
                    image_path.name
                )
            )

            if previous_number is None:

                current = [
                    image_path
                ]

            elif current_number == previous_number + 1:

                current.append(
                    image_path
                )

            else:

                if current:
                    sequences.append(
                        current
                    )

                current = [
                    image_path
                ]

            previous_number = current_number

        if current:
            sequences.append(
                current
            )

    return sequences


# ============================================================
# Split large sequence into contiguous chunks
# ============================================================

def chunk_sequence(sequence):
    """
    Divide a large sequence into contiguous chunks.

    Example:

        765 images

    becomes:

        50 + 50 + 50 + ... + remainder

    The order of frames is preserved.
    """

    if len(sequence) <= MAX_SEQUENCE_CHUNK:

        return [sequence]

    chunks = []

    for start in range(
        0,
        len(sequence),
        MAX_SEQUENCE_CHUNK
    ):

        chunk = sequence[
            start:
            start + MAX_SEQUENCE_CHUNK
        ]

        chunks.append(
            chunk
        )

    return chunks


# ============================================================
# Build all groups
# ============================================================

def build_groups(image_files):
    """
    Build sequence groups and split very large sequences
    into contiguous chunks.
    """

    sequences = build_sequences(
        image_files
    )

    groups = []

    for sequence in sequences:

        chunks = chunk_sequence(
            sequence
        )

        groups.extend(
            chunks
        )

    return groups


# ============================================================
# Determine classes represented by group
# ============================================================

def group_classes(
    group,
    labels_root
):
    """
    Find which animal classes occur in this group.
    """

    classes = set()

    for image_path in group:

        label_path = (
            labels_root
            / f"{image_path.stem}.txt"
        )

        classes.update(
            get_classes_from_label(
                label_path
            )
        )

    return classes


# ============================================================
# Clean old split
# ============================================================

def clean_split_directories():

    print()
    print("=" * 70)
    print("CLEANING PREVIOUS TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    for directory in [
        TRAIN_ROOT,
        VAL_ROOT,
    ]:

        if directory.exists():

            print(
                f"Removing: {directory}"
            )

            shutil.rmtree(
                directory
            )

    print(
        "Old train/validation directories removed."
    )


# ============================================================
# Create directories
# ============================================================

def create_output_directories():

    for root in [
        TRAIN_ROOT,
        VAL_ROOT,
    ]:

        (
            root / "images"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            root / "labels"
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# Hardlink / copy
# ============================================================

def link_or_copy(
    source,
    destination
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        os.link(
            source,
            destination
        )

        return "hardlink"

    except OSError:

        shutil.copy2(
            source,
            destination
        )

        return "copy"


# ============================================================
# Count classes in groups
# ============================================================

def calculate_group_class_counts(
    groups,
    labels_root
):

    group_counts = []

    for group in groups:

        counter = Counter()

        for image_path in group:

            label_path = (
                labels_root
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():
                continue

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    try:

                        class_id = int(
                            parts[0]
                        )

                        if (
                            0 <= class_id
                            < len(CLASS_NAMES)
                        ):

                            counter[class_id] += 1

                    except ValueError:
                        continue

        group_counts.append(
            counter
        )

    return group_counts


# ============================================================
# Balanced group assignment
# ============================================================

def assign_groups(
    groups,
    group_counts
):
    """
    Greedy class-balanced assignment.

    The algorithm tries to keep validation near 20% for
    every class while maintaining the overall 80/20 split.

    Large sequence chunks remain contiguous.
    """

    rng = random.Random(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Shuffle group order
    # --------------------------------------------------------

    indices = list(
        range(len(groups))
    )

    rng.shuffle(
        indices
    )

    # --------------------------------------------------------
    # Total class counts
    # --------------------------------------------------------

    total_counts = Counter()

    for counter in group_counts:

        total_counts.update(
            counter
        )

    target_val = {
        class_id:
        count * VAL_RATIO
        for class_id, count
        in total_counts.items()
    }

    current_val = Counter()

    val_groups = set()

    train_groups = set()

    # --------------------------------------------------------
    # Process groups with high class contribution first
    # --------------------------------------------------------

    def group_weight(index):

        counter = group_counts[index]

        return (
            sum(counter.values()),
            max(
                counter.values()
            )
            if counter
            else 0
        )

    indices.sort(
        key=group_weight,
        reverse=True
    )

    # --------------------------------------------------------
    # Greedy assignment
    # --------------------------------------------------------

    for index in indices:

        counter = group_counts[index]

        if not counter:

            train_groups.add(
                index
            )

            continue

        # --------------------------------------------
        # Calculate benefit of putting this group
        # into validation.
        # --------------------------------------------

        current_error = 0.0
        val_error = 0.0

        for class_id, total in total_counts.items():

            target = target_val[class_id]

            before = current_val.get(
                class_id,
                0
            )

            after = before + counter.get(
                class_id,
                0
            )

            current_error += abs(
                before - target
            )

            val_error += abs(
                after - target
            )

        # --------------------------------------------
        # Number of validation images
        # --------------------------------------------

        current_val_images = sum(
            len(groups[i])
            for i in val_groups
        )

        total_images = sum(
            len(group)
            for group in groups
        )

        target_val_images = (
            total_images
            * VAL_RATIO
        )

        # --------------------------------------------
        # If validation is still under target,
        # favour validation when it improves class
        # balance.
        # --------------------------------------------

        if (
            val_error < current_error
            and current_val_images
            < target_val_images
        ):

            val_groups.add(
                index
            )

            current_val.update(
                counter
            )

        else:

            train_groups.add(
                index
            )

    # --------------------------------------------------------
    # Ensure every class has validation representation
    # --------------------------------------------------------

    for class_id in range(
        len(CLASS_NAMES)
    ):

        validation_objects = sum(
            group_counts[i].get(
                class_id,
                0
            )
            for i in val_groups
        )

        if validation_objects > 0:
            continue

        # Find a train group containing this class.
        candidates = []

        for index in train_groups:

            count = group_counts[index].get(
                class_id,
                0
            )

            if count > 0:

                candidates.append(
                    (
                        count,
                        index
                    )
                )

        if candidates:

            candidates.sort()

            _, selected = candidates[0]

            train_groups.remove(
                selected
            )

            val_groups.add(
                selected
            )

            current_val.update(
                group_counts[selected]
            )

    return (
        train_groups,
        val_groups
    )


# ============================================================
# Populate split
# ============================================================

def populate_split(
    groups,
    split_root
):

    image_output = (
        split_root
        / "images"
    )

    label_output = (
        split_root
        / "labels"
    )

    image_count = 0
    label_count = 0

    hardlinks = 0
    copies = 0

    for group in groups:

        for image_path in group:

            # --------------------------------------------
            # Source label directory
            # --------------------------------------------

            source_dataset = (
                image_path.parent.parent
            )

            label_path = (
                source_dataset
                / "labels"
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():

                print(
                    f"[WARNING] Missing label:"
                    f" {label_path.name}"
                )

                continue

            # --------------------------------------------
            # Destination paths
            # --------------------------------------------

            destination_image = (
                image_output
                / image_path.name
            )

            destination_label = (
                label_output
                / label_path.name
            )

            # --------------------------------------------
            # Image
            # --------------------------------------------

            result = link_or_copy(
                image_path,
                destination_image
            )

            if result == "hardlink":
                hardlinks += 1
            else:
                copies += 1

            # --------------------------------------------
            # Label
            # --------------------------------------------

            link_or_copy(
                label_path,
                destination_label
            )

            image_count += 1
            label_count += 1

    return {
        "images": image_count,
        "labels": label_count,
        "hardlinks": hardlinks,
        "copies": copies,
    }


# ============================================================
# Analyze final split
# ============================================================

def analyze_split():

    train_labels = (
        TRAIN_ROOT / "labels"
    )

    val_labels = (
        VAL_ROOT / "labels"
    )

    train_counts = Counter()
    val_counts = Counter()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for label_path in train_labels.glob("*.txt"):

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])

                    if 0 <= class_id < len(CLASS_NAMES):
                        train_counts[class_id] += 1

                except ValueError:
                    continue

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    for label_path in val_labels.glob("*.txt"):

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])

                    if 0 <= class_id < len(CLASS_NAMES):
                        val_counts[class_id] += 1

                except ValueError:
                    continue

    # --------------------------------------------------------
    # Print table
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("FINAL CLASS DISTRIBUTION")
    print("=" * 90)

    print(
        f"{'Class':<18}"
        f"{'Train':>12}"
        f"{'Val':>12}"
        f"{'Total':>12}"
        f"{'Val %':>12}"
    )

    print("-" * 90)

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        train = train_counts.get(
            class_id,
            0
        )

        val = val_counts.get(
            class_id,
            0
        )

        total = train + val

        val_percentage = (
            val / total * 100
            if total > 0
            else 0
        )

        print(
            f"{class_name:<18}"
            f"{train:>12,}"
            f"{val:>12,}"
            f"{total:>12,}"
            f"{val_percentage:>11.2f}%"
        )

    print("-" * 90)

    total_train = sum(
        train_counts.values()
    )

    total_val = sum(
        val_counts.values()
    )

    total_objects = (
        total_train
        + total_val
    )

    print(
        f"{'TOTAL':<18}"
        f"{total_train:>12,}"
        f"{total_val:>12,}"
        f"{total_objects:>12,}"
    )

    return (
        total_train,
        total_val,
        total_objects
    )


# ============================================================
# Create data.yaml
# ============================================================

def create_dataset_yaml():

    yaml_path = (
        YOLO_ROOT
        / "data.yaml"
    )

    yaml_content = f"""# VanDrishti YOLO Dataset

path: {YOLO_ROOT.as_posix()}

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
    ) as f:

        f.write(
            yaml_content
        )

    print()
    print(
        f"[OK] data.yaml updated:"
    )

    print(
        yaml_path
    )


# ============================================================
# Process one dataset
# ============================================================

def process_dataset(
    dataset_name,
    dataset_root
):

    image_dir = (
        dataset_root
        / "images"
    )

    label_dir = (
        dataset_root
        / "labels"
    )

    image_files = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in {".jpg", ".jpeg"}
        ],
        key=lambda p: p.name.lower()
    )

    print()
    print("=" * 70)
    print(
        f"{dataset_name.upper()} DATASET"
    )
    print("=" * 70)

    print(
        f"Images: {len(image_files):,}"
    )

    # --------------------------------------------------------
    # Build groups
    # --------------------------------------------------------

    groups = build_groups(
        image_files
    )

    sequence_groups = build_sequences(
        image_files
    )

    large_sequences = sum(
        1
        for sequence in sequence_groups
        if len(sequence) > MAX_SEQUENCE_CHUNK
    )

    print(
        f"Original sequence groups: "
        f"{len(sequence_groups):,}"
    )

    print(
        f"Large sequences chunked: "
        f"{large_sequences:,}"
    )

    print(
        f"Final groups: "
        f"{len(groups):,}"
    )

    # --------------------------------------------------------
    # Calculate class counts
    # --------------------------------------------------------

    group_counts = calculate_group_class_counts(
        groups,
        label_dir
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_indices, val_indices = assign_groups(
        groups,
        group_counts
    )

    train_groups = [
        groups[i]
        for i in train_indices
    ]

    val_groups = [
        groups[i]
        for i in val_indices
    ]

    train_images = sum(
        len(group)
        for group in train_groups
    )

    val_images = sum(
        len(group)
        for group in val_groups
    )

    print()
    print(
        "Split:"
    )

    print(
        f"Train images: "
        f"{train_images:,}"
    )

    print(
        f"Validation images: "
        f"{val_images:,}"
    )

    print(
        f"Train ratio: "
        f"{train_images / len(image_files) * 100:.2f}%"
    )

    print(
        f"Validation ratio: "
        f"{val_images / len(image_files) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Return groups
    # --------------------------------------------------------

    return (
        train_groups,
        val_groups
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "VANDRISHTI - BALANCED SEQUENCE-AWARE SPLIT"
    )
    print("=" * 70)

    print()
    print(
        "Train ratio:"
        f" {TRAIN_RATIO * 100:.0f}%"
    )

    print(
        "Validation ratio:"
        f" {VAL_RATIO * 100:.0f}%"
    )

    print(
        "Maximum sequence chunk:"
        f" {MAX_SEQUENCE_CHUNK}"
    )

    print(
        "Random seed:"
        f" {RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # Verify source
    # --------------------------------------------------------

    for dataset_name, dataset_root in SOURCE_DATASETS.items():

        if not dataset_root.exists():

            print()
            print(
                f"[ERROR] Missing dataset:"
            )

            print(
                dataset_root
            )

            return

    # --------------------------------------------------------
    # Clean old split
    # --------------------------------------------------------

    clean_split_directories()

    create_output_directories()

    # --------------------------------------------------------
    # Build Day groups
    # --------------------------------------------------------

    day_train, day_val = process_dataset(
        "day",
        SOURCE_DATASETS["day"]
    )

    # --------------------------------------------------------
    # Build Night groups
    # --------------------------------------------------------

    night_train, night_val = process_dataset(
        "night",
        SOURCE_DATASETS["night"]
    )

    # --------------------------------------------------------
    # Populate train
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING TRAINING DATA")
    print("=" * 70)

    train_result_day = populate_split(
        day_train,
        TRAIN_ROOT
    )

    train_result_night = populate_split(
        night_train,
        TRAIN_ROOT
    )

    # --------------------------------------------------------
    # Populate validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING VALIDATION DATA")
    print("=" * 70)

    val_result_day = populate_split(
        day_val,
        VAL_ROOT
    )

    val_result_night = populate_split(
        night_val,
        VAL_ROOT
    )

    # --------------------------------------------------------
    # Final image counts
    # --------------------------------------------------------

    train_images = (
        train_result_day["images"]
        + train_result_night["images"]
    )

    val_images = (
        val_result_day["images"]
        + val_result_night["images"]
    )

    total_images = (
        train_images
        + val_images
    )

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    create_dataset_yaml()

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    (
        train_objects,
        val_objects,
        total_objects
    ) = analyze_split()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SPLIT SUMMARY")
    print("=" * 70)

    print(
        f"Training images     : "
        f"{train_images:,}"
    )

    print(
        f"Validation images   : "
        f"{val_images:,}"
    )

    print(
        f"Total images        : "
        f"{total_images:,}"
    )

    print()

    print(
        f"Training objects    : "
        f"{train_objects:,}"
    )

    print(
        f"Validation objects  : "
        f"{val_objects:,}"
    )

    print(
        f"Total objects       : "
        f"{total_objects:,}"
    )

    print()

    if total_images == 25657:

        print(
            "[SUCCESS] All 25,657 images included."
        )

    else:

        print(
            "[WARNING] Image total is not 25,657."
        )

    if total_objects == 26946:

        print(
            "[SUCCESS] All 26,946 objects included."
        )

    else:

        print(
            "[WARNING] Object total is not 26,946."
        )

    print()
    print(
        "Original VOC dataset was not modified."
    )

    print()
    print(
        "Train:"
    )

    print(
        TRAIN_ROOT
    )

    print()
    print(
        "Validation:"
    )

    print(
        VAL_ROOT
    )

    print()
    print("=" * 70)
    print(
        "BALANCED SEQUENCE-AWARE SPLIT COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()