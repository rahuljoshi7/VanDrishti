from pathlib import Path
from collections import defaultdict
import re


DATASETS = {
    "DAY": Path(
        r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day"
    ),
    "NIGHT": Path(
        r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night"
    ),
}


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def analyze_dataset(name, dataset_path):

    print("\n" + "=" * 70)
    print(f"                 {name} SEQUENCE AUDIT")
    print("=" * 70)

    image_dir = dataset_path / "JPEGImages"

    groups = defaultdict(list)

    for image in image_dir.iterdir():

        if not image.is_file():
            continue

        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = image.stem

        # Find trailing number
        match = re.match(r"^(.*?)[_-](\d+)$", stem)

        if match:
            prefix = match.group(1)
            number = int(match.group(2))

            groups[prefix].append(number)

    print(f"\nFilename groups found: {len(groups)}")

    # --------------------------------------------------------
    # Analyze consecutive runs
    # --------------------------------------------------------

    print("\nCONSECUTIVE NUMBER RUNS")
    print("-" * 70)

    total_runs = 0
    long_runs = []

    for prefix, numbers in groups.items():

        numbers = sorted(set(numbers))

        if not numbers:
            continue

        start = numbers[0]
        previous = numbers[0]

        for number in numbers[1:]:

            if number == previous + 1:
                previous = number

            else:
                length = previous - start + 1

                total_runs += 1

                if length >= 5:
                    long_runs.append(
                        (length, prefix, start, previous)
                    )

                start = number
                previous = number

        # Last run
        length = previous - start + 1

        total_runs += 1

        if length >= 5:
            long_runs.append(
                (length, prefix, start, previous)
            )

    print(f"Total number runs: {total_runs:,}")

    long_runs.sort(reverse=True)

    print("\nLongest runs (5+ consecutive numbers):")

    for length, prefix, start, end in long_runs[:30]:

        print(
            f"{prefix:<20} "
            f"{start:>8} -> {end:<8} "
            f"length = {length}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nSUMMARY")
    print("-" * 70)

    print(
        f"Groups containing numbered images: "
        f"{len(groups):,}"
    )

    print(
        f"Runs with 5+ consecutive numbers: "
        f"{len(long_runs):,}"
    )

    if long_runs:

        print("\nLargest observed sequence-like runs:")

        for length, prefix, start, end in long_runs[:10]:

            print(
                f"  {prefix}: "
                f"{start} - {end} "
                f"({length} images)"
            )


def main():

    print("=" * 70)
    print("          VanDrishti - SEQUENCE ANALYSIS")
    print("=" * 70)

    for name, path in DATASETS.items():
        analyze_dataset(name, path)

    print("\n" + "=" * 70)
    print("                  AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()