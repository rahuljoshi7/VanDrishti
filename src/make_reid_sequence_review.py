from pathlib import Path
import math

import cv2
import matplotlib.pyplot as plt
import pandas as pd


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_FILE = (
    ROOT
    / "data"
    / "reid"
    / "metadata"
    / "reid_candidate_analysis.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "reid_review"
    / "phase2b"
)

# Original FIRC dataset locations.
# We read the original images directly and do not copy them
# into the VanDrishti project.
DAY_IMAGE_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day\JPEGImages"
)

NIGHT_IMAGE_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night\JPEGImages"
)


# =========================================================
# FIND ORIGINAL DATASET IMAGE
# =========================================================

def find_dataset_image(dataset, prefix, number):
    """
    Find the original dataset image corresponding to
    a candidate sequence number.

    Parameters
    ----------
    dataset : str
        "day" or "night"

    prefix : str
        Filename prefix stored in candidate_sequences.csv.

    number : int
        Numeric sequence number.

    Returns
    -------
    pathlib.Path
        Path to the matching image.
    """

    # -----------------------------------------------------
    # Select correct original dataset directory
    # -----------------------------------------------------

    if dataset == "day":

        image_dir = DAY_IMAGE_DIR

    elif dataset == "night":

        image_dir = NIGHT_IMAGE_DIR

    else:

        raise ValueError(
            f"Unknown dataset: {dataset}"
        )

    # -----------------------------------------------------
    # Verify directory
    # -----------------------------------------------------

    if not image_dir.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n"
            f"{image_dir}"
        )

    # -----------------------------------------------------
    # Construct expected filename
    # -----------------------------------------------------

    number_str = str(int(number))

    exact_stem = (
        f"{prefix}{number_str}"
    )

    # -----------------------------------------------------
    # Try common extensions
    # -----------------------------------------------------

    extensions = [
        ".jpg",
        ".jpeg",
        ".JPG",
        ".JPEG",
        ".png",
        ".PNG",
    ]

    for extension in extensions:

        exact_path = (
            image_dir
            / f"{exact_stem}{extension}"
        )

        if exact_path.exists():

            return exact_path

    # -----------------------------------------------------
    # Fallback: case-insensitive stem matching
    # -----------------------------------------------------

    target_stem = exact_stem.lower()

    for path in image_dir.iterdir():

        if not path.is_file():
            continue

        if path.stem.lower() == target_stem:

            return path

    # -----------------------------------------------------
    # Nothing found
    # -----------------------------------------------------

    raise FileNotFoundError(
        f"Could not find image for:\n"
        f"  Dataset : {dataset}\n"
        f"  Prefix  : {prefix}\n"
        f"  Number  : {number}\n"
        f"  Expected: {exact_stem}\n"
        f"  Search  : {image_dir}"
    )


# =========================================================
# SAMPLE SEQUENCE NUMBERS
# =========================================================

def sample_numbers(start, end, count=12):
    """
    Select representative frames across a sequence.

    The first and last frames are always included.
    Remaining frames are sampled deterministically.
    """

    total = end - start + 1

    # If the sequence is already short,
    # use every frame.
    if total <= count:

        return list(
            range(
                start,
                end + 1,
            )
        )

    # -----------------------------------------------------
    # Deterministic sampling
    # -----------------------------------------------------

    sampled = (
        pd.Series(
            range(
                start,
                end + 1,
            )
        )
        .sample(
            n=count,
            random_state=42,
        )
        .sort_values()
        .tolist()
    )

    sampled = [
        int(value)
        for value in sampled
    ]

    # -----------------------------------------------------
    # Always include sequence boundaries
    # -----------------------------------------------------

    sampled[0] = start
    sampled[-1] = end

    return sorted(
        set(sampled)
    )


# =========================================================
# CREATE CONTACT SHEET
# =========================================================

def create_contact_sheet(
    dataset,
    prefix,
    start,
    end,
    count=12,
):
    """
    Create a visual review contact sheet for one
    candidate sequence.
    """

    # -----------------------------------------------------
    # Select representative frames
    # -----------------------------------------------------

    numbers = sample_numbers(
        start,
        end,
        count,
    )

    images = []

    # -----------------------------------------------------
    # Load images
    # -----------------------------------------------------

    for number in numbers:

        path = find_dataset_image(
            dataset,
            prefix,
            number,
        )

        image = cv2.imread(
            str(path)
        )

        if image is None:

            raise RuntimeError(
                f"Could not read image:\n"
                f"{path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        images.append(
            (
                number,
                path.name,
                image,
            )
        )

    # -----------------------------------------------------
    # Contact-sheet layout
    # -----------------------------------------------------

    columns = 4

    rows = math.ceil(
        len(images) / columns
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            16,
            4 * rows,
        ),
    )

    # Handle both single-axis and multi-axis cases.
    if hasattr(
        axes,
        "flat",
    ):

        axes = list(
            axes.flat
        )

    else:

        axes = [axes]

    # -----------------------------------------------------
    # Hide all axes initially
    # -----------------------------------------------------

    for ax in axes:

        ax.axis("off")

    # -----------------------------------------------------
    # Draw images
    # -----------------------------------------------------

    for ax, (
        number,
        filename,
        image,
    ) in zip(
        axes,
        images,
    ):

        ax.imshow(
            image
        )

        ax.set_title(
            f"{filename}\n"
            f"Sequence number: {number}",
            fontsize=8,
        )

        ax.axis("off")

    # -----------------------------------------------------
    # Overall title
    # -----------------------------------------------------

    fig.suptitle(
        "VanDrishti Re-ID Review\n"
        f"{dataset} | {prefix} | "
        f"{start}–{end}\n"
        f"{len(images)} sampled frames",
        fontsize=14,
    )

    fig.tight_layout()

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Safe output filename
    # -----------------------------------------------------

    safe_prefix = (
        prefix
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"{dataset}_"
            f"{safe_prefix}"
            f"{start}_{end}.jpg"
        )
    )

    # -----------------------------------------------------
    # Save contact sheet
    # -----------------------------------------------------

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"[SAVED] {output_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)

    print(
        "VanDrishti - Phase 2B "
        "Re-ID Sequence Review"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # Verify candidate metadata
    # -----------------------------------------------------

    if not CANDIDATE_FILE.exists():

        raise FileNotFoundError(
            f"Candidate file not found:\n"
            f"{CANDIDATE_FILE}"
        )

    # -----------------------------------------------------
    # Load candidate table
    # -----------------------------------------------------

    df = pd.read_csv(
        CANDIDATE_FILE
    )

    print()
    print(
        f"Candidate sequences available: "
        f"{len(df)}"
    )

    # -----------------------------------------------------
    # Verify original dataset directories
    # -----------------------------------------------------

    print()
    print(
        f"Day dataset   : {DAY_IMAGE_DIR}"
    )

    print(
        f"Day exists    : {DAY_IMAGE_DIR.exists()}"
    )

    print(
        f"Night dataset : {NIGHT_IMAGE_DIR}"
    )

    print(
        f"Night exists  : {NIGHT_IMAGE_DIR.exists()}"
    )

    # -----------------------------------------------------
    # Selected Phase 2B candidate sequences
    # -----------------------------------------------------

    selections = [

        # Night Hare
        (
            "night",
            "hare_",
            516,
            1280,
        ),

        # Night Amur Tiger
        (
            "night",
            "amurtiger_",
            847,
            1057,
        ),

        # Night Amur Leopard
        (
            "night",
            "amurleopard_",
            1186,
            1374,
        ),

        # Day Musk Deer
        (
            "day",
            "muskdeer_",
            410,
            589,
        ),

        # Day Sika Deer
        (
            "day",
            "sikadeer",
            1057,
            1199,
        ),

        # Night Raccoon Dog
        (
            "night",
            "raccoondog",
            128,
            226,
        ),

        # Day Black Bear
        (
            "day",
            "blackbear",
            100,
            143,
        ),

        # Night Red Fox
        (
            "night",
            "redfox",
            1265,
            1310,
        ),

        # Day Wild Boar
        (
            "day",
            "wildboar",
            955,
            998,
        ),

        # Day Leopard Cat
        (
            "day",
            "leopardcat__",
            1146,
            1182,
        ),
    ]

    # -----------------------------------------------------
    # Generate contact sheets
    # -----------------------------------------------------

    for (
        dataset,
        prefix,
        start,
        end,
    ) in selections:

        print()

        print(
            f"Creating review: "
            f"{dataset} | "
            f"{prefix} | "
            f"{start}-{end}"
        )

        create_contact_sheet(
            dataset,
            prefix,
            start,
            end,
            count=12,
        )

    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    print()

    print("=" * 70)

    print(
        "Phase 2B contact-sheet generation complete."
    )

    print()

    print(
        f"Review directory:"
    )

    print(
        OUTPUT_DIR
    )

    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()