from pathlib import Path
from collections import Counter


# ============================================================
# VanDrishti - Train / Validation Class Distribution Analysis
# ============================================================
#
# Purpose:
#   Analyze the number of objects for every animal class in
#   the training and validation datasets.
#
# This script:
#   - Does NOT modify any dataset files.
#   - Reads YOLO .txt label files only.
#   - Counts objects class-wise.
#   - Calculates train/validation percentages.
#   - Reports missing classes.
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


TRAIN_LABELS = (
    YOLO_ROOT
    / "train"
    / "labels"
)


VAL_LABELS = (
    YOLO_ROOT
    / "val"
    / "labels"
)


# ------------------------------------------------------------
# Class mapping
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
# COUNT OBJECTS
# ============================================================

def count_objects(label_directory):

    counts = Counter()

    label_files = sorted(
        label_directory.glob("*.txt")
    )

    empty_labels = 0
    invalid_lines = 0

    for label_file in label_files:

        with open(
            label_file,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if not lines:

            empty_labels += 1
            continue

        for line in lines:

            parts = line.split()

            if len(parts) != 5:

                invalid_lines += 1

                continue

            try:

                class_id = int(
                    parts[0]
                )

                if (
                    class_id < 0
                    or class_id >= len(CLASS_NAMES)
                ):

                    invalid_lines += 1
                    continue

                counts[class_id] += 1

            except ValueError:

                invalid_lines += 1

    return (
        counts,
        len(label_files),
        empty_labels,
        invalid_lines,
    )


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

def print_distribution(
    train_counts,
    val_counts
):

    train_total = sum(
        train_counts.values()
    )

    val_total = sum(
        val_counts.values()
    )

    total_objects = (
        train_total
        + val_total
    )

    print()
    print("=" * 90)
    print(
        "CLASS-WISE OBJECT DISTRIBUTION"
    )
    print("=" * 90)

    print()

    header = (
        f"{'Class':<18}"
        f"{'Train':>12}"
        f"{'Val':>12}"
        f"{'Total':>12}"
        f"{'Train %':>12}"
        f"{'Val %':>12}"
    )

    print(header)

    print("-" * 90)

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        train_count = train_counts.get(
            class_id,
            0
        )

        val_count = val_counts.get(
            class_id,
            0
        )

        total_count = (
            train_count
            + val_count
        )

        if total_count > 0:

            train_percentage = (
                train_count
                / total_count
                * 100
            )

            val_percentage = (
                val_count
                / total_count
                * 100
            )

        else:

            train_percentage = 0
            val_percentage = 0

        print(
            f"{class_name:<18}"
            f"{train_count:>12,}"
            f"{val_count:>12,}"
            f"{total_count:>12,}"
            f"{train_percentage:>11.2f}%"
            f"{val_percentage:>11.2f}%"
        )

    print("-" * 90)

    print(
        f"{'TOTAL':<18}"
        f"{train_total:>12,}"
        f"{val_total:>12,}"
        f"{total_objects:>12,}"
    )

    print()


# ============================================================
# CHECK DISTRIBUTION
# ============================================================

def analyze_balance(
    train_counts,
    val_counts
):

    print()
    print("=" * 90)
    print(
        "DISTRIBUTION CHECK"
    )
    print("=" * 90)

    warnings = []

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        train_count = train_counts.get(
            class_id,
            0
        )

        val_count = val_counts.get(
            class_id,
            0
        )

        total = (
            train_count
            + val_count
        )

        if total == 0:

            warnings.append(
                f"{class_name}: "
                f"NO OBJECTS FOUND"
            )

            continue

        val_percentage = (
            val_count
            / total
            * 100
        )

        # ----------------------------------------------------
        # Flag unusually low validation representation
        # ----------------------------------------------------

        if val_percentage < 10:

            warnings.append(
                f"{class_name}: "
                f"only {val_percentage:.2f}% "
                f"in validation"
            )

        # ----------------------------------------------------
        # Flag unusually high validation representation
        # ----------------------------------------------------

        elif val_percentage > 30:

            warnings.append(
                f"{class_name}: "
                f"{val_percentage:.2f}% "
                f"in validation"
            )

    if warnings:

        print()

        print(
            "[WARNING] Classes requiring attention:"
        )

        for warning in warnings:

            print(
                f"  - {warning}"
            )

    else:

        print()

        print(
            "[OK] All classes have reasonable "
            "validation representation."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print(
        "VANDRISHTI - DATASET SPLIT ANALYSIS"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not TRAIN_LABELS.exists():

        print(
            "[ERROR] Training labels directory "
            "not found:"
        )

        print(
            TRAIN_LABELS
        )

        return

    if not VAL_LABELS.exists():

        print(
            "[ERROR] Validation labels directory "
            "not found:"
        )

        print(
            VAL_LABELS
        )

        return

    # --------------------------------------------------------
    # Count training objects
    # --------------------------------------------------------

    print()
    print(
        "Reading training labels..."
    )

    (
        train_counts,
        train_files,
        train_empty,
        train_invalid,
    ) = count_objects(
        TRAIN_LABELS
    )

    print(
        f"Training label files: "
        f"{train_files:,}"
    )

    # --------------------------------------------------------
    # Count validation objects
    # --------------------------------------------------------

    print()
    print(
        "Reading validation labels..."
    )

    (
        val_counts,
        val_files,
        val_empty,
        val_invalid,
    ) = count_objects(
        VAL_LABELS
    )

    print(
        f"Validation label files: "
        f"{val_files:,}"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print_distribution(
        train_counts,
        val_counts
    )

    # --------------------------------------------------------
    # Balance
    # --------------------------------------------------------

    analyze_balance(
        train_counts,
        val_counts
    )

    # --------------------------------------------------------
    # File integrity
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "LABEL FILE INTEGRITY"
    )
    print("=" * 90)

    print(
        f"Training label files : "
        f"{train_files:,}"
    )

    print(
        f"Validation label files: "
        f"{val_files:,}"
    )

    print(
        f"Empty training labels : "
        f"{train_empty:,}"
    )

    print(
        f"Empty validation labels: "
        f"{val_empty:,}"
    )

    print(
        f"Invalid train lines   : "
        f"{train_invalid:,}"
    )

    print(
        f"Invalid val lines     : "
        f"{val_invalid:,}"
    )

    if (
        train_empty == 0
        and val_empty == 0
        and train_invalid == 0
        and val_invalid == 0
    ):

        print()
        print(
            "[SUCCESS] All label files are valid."
        )

    else:

        print()
        print(
            "[WARNING] Some label files require "
            "additional inspection."
        )

    # --------------------------------------------------------
    # Object totals
    # --------------------------------------------------------

    train_total = sum(
        train_counts.values()
    )

    val_total = sum(
        val_counts.values()
    )

    print()
    print("=" * 90)
    print(
        "OBJECT TOTALS"
    )
    print("=" * 90)

    print(
        f"Training objects   : "
        f"{train_total:,}"
    )

    print(
        f"Validation objects : "
        f"{val_total:,}"
    )

    print(
        f"Total objects      : "
        f"{train_total + val_total:,}"
    )

    # --------------------------------------------------------
    # Expected total
    # --------------------------------------------------------

    expected_total = 26946

    print()
    print(
        f"Expected objects   : "
        f"{expected_total:,}"
    )

    if (
        train_total + val_total
        == expected_total
    ):

        print(
            "[SUCCESS] Object total matches "
            "the original dataset audit."
        )

    else:

        print(
            "[WARNING] Object total does NOT "
            "match the original audit."
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "ANALYSIS COMPLETE"
    )
    print("=" * 90)


if __name__ == "__main__":

    main()