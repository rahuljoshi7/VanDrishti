from pathlib import Path
import re
import csv


# ============================================================
# VanDrishti - Re-ID Candidate Sequence Discovery
# ============================================================
#
# IMPORTANT:
# This script ONLY discovers candidate image sequences.
# It does NOT assume that consecutive filenames represent
# the same individual animal.
#
# Original dataset files are READ ONLY.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DAY_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day\JPEGImages"
)

NIGHT_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night\JPEGImages"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "reid" / "candidates"

MIN_SEQUENCE_LENGTH = 5


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def extract_numbers(filename):
    """
    Extract all numeric groups from a filename.

    Example:
        Hare_000516.jpg
        -> ['000516']
    """

    return re.findall(r"\d+", filename)


def get_sequence_number(filename):
    """
    Use the final numeric group in the filename as the
    sequence number.

    Example:
        Hare_000516.jpg
        -> 516
    """

    numbers = extract_numbers(filename)

    if not numbers:
        return None

    return int(numbers[-1])


def discover_sequences(dataset_dir, dataset_name):
    """
    Discover consecutive filename sequences.

    Returns a list of dictionaries.
    """

    if not dataset_dir.exists():
        print(f"[WARNING] Dataset directory not found:")
        print(f"          {dataset_dir}")
        return []

    images = [
        p
        for p in dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"\n[{dataset_name}] Images found: {len(images):,}")

    # Group images by prefix before the final number.
    groups = {}

    for image_path in images:

        sequence_number = get_sequence_number(image_path.name)

        if sequence_number is None:
            continue

        numbers = extract_numbers(image_path.name)

        # Remove the final numeric sequence from the filename.
        final_number = numbers[-1]

        prefix = image_path.name.rsplit(final_number, 1)[0]

        key = prefix.lower()

        groups.setdefault(key, []).append(
            (sequence_number, image_path)
        )

    candidates = []

    for prefix, items in groups.items():

        items.sort(key=lambda x: x[0])

        current_sequence = []

        previous_number = None

        for number, path in items:

            if previous_number is None or number == previous_number + 1:
                current_sequence.append((number, path))

            else:
                if len(current_sequence) >= MIN_SEQUENCE_LENGTH:
                    candidates.append(
                        {
                            "dataset": dataset_name,
                            "prefix": prefix,
                            "length": len(current_sequence),
                            "start": current_sequence[0][0],
                            "end": current_sequence[-1][0],
                            "images": [str(x[1]) for x in current_sequence],
                        }
                    )

                current_sequence = [(number, path)]

            previous_number = number

        # Handle final sequence.
        if len(current_sequence) >= MIN_SEQUENCE_LENGTH:
            candidates.append(
                {
                    "dataset": dataset_name,
                    "prefix": prefix,
                    "length": len(current_sequence),
                    "start": current_sequence[0][0],
                    "end": current_sequence[-1][0],
                    "images": [str(x[1]) for x in current_sequence],
                }
            )

    return candidates


def save_candidates(candidates, dataset_name):

    output_dir = OUTPUT_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "candidate_sequences.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "dataset",
                "prefix",
                "sequence_length",
                "start_number",
                "end_number",
                "candidate_images",
            ]
        )

        for candidate in candidates:

            writer.writerow(
                [
                    candidate["dataset"],
                    candidate["prefix"],
                    candidate["length"],
                    candidate["start"],
                    candidate["end"],
                    " | ".join(candidate["images"]),
                ]
            )

    print(f"[SAVED] {csv_path}")

    return csv_path


def print_summary(candidates, dataset_name):

    print(f"\n===== {dataset_name.upper()} SUMMARY =====")

    print(
        f"Candidate sequences (length >= {MIN_SEQUENCE_LENGTH}): "
        f"{len(candidates):,}"
    )

    if not candidates:
        return

    lengths = [c["length"] for c in candidates]

    print(f"Shortest candidate: {min(lengths)}")
    print(f"Longest candidate:  {max(lengths)}")
    print(f"Average length:     {sum(lengths) / len(lengths):.2f}")

    print("\nTop 10 longest candidate sequences:")

    top_candidates = sorted(
        candidates,
        key=lambda x: x["length"],
        reverse=True,
    )[:10]

    for i, candidate in enumerate(top_candidates, start=1):

        print(
            f"{i:2}. "
            f"{candidate['prefix']} "
            f"{candidate['start']}-{candidate['end']} "
            f"({candidate['length']} images)"
        )


def main():

    print("=" * 70)
    print("VanDrishti - Re-ID Candidate Sequence Discovery")
    print("=" * 70)

    print("\nThis script performs candidate discovery only.")
    print("It does NOT assign individual identities.")

    day_candidates = discover_sequences(
        DAY_DIR,
        "day",
    )

    night_candidates = discover_sequences(
        NIGHT_DIR,
        "night",
    )

    save_candidates(
        day_candidates,
        "day",
    )

    save_candidates(
        night_candidates,
        "night",
    )

    print_summary(
        day_candidates,
        "day",
    )

    print_summary(
        night_candidates,
        "night",
    )

    print("\n" + "=" * 70)
    print("Candidate discovery completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()