from pathlib import Path
import math

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = (
    ROOT
    / "data"
    / "reid"
    / "metadata"
    / "phase2b_crop_metadata.csv"
)

CROP_DIR = (
    ROOT
    / "data"
    / "reid"
    / "crops"
    / "phase2b_same"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "reid_review"
    / "phase2b"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "phase2b_crop_verification.jpg"
)


def main():

    print("=" * 70)
    print("VanDrishti - Phase 2B Crop Verification")
    print("=" * 70)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA_FILE}"
        )

    df = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"Pairs in metadata: {len(df)}"
    )

    rows = []

    for _, row in df.iterrows():

        crop_1 = CROP_DIR / row["crop_1"]
        crop_2 = CROP_DIR / row["crop_2"]

        if not crop_1.exists():
            raise FileNotFoundError(
                f"Crop not found: {crop_1}"
            )

        if not crop_2.exists():
            raise FileNotFoundError(
                f"Crop not found: {crop_2}"
            )

        image_1 = Image.open(
            crop_1
        ).convert("RGB")

        image_2 = Image.open(
            crop_2
        ).convert("RGB")

        rows.append(
            (
                row["pair_id"],
                row["species"],
                row["difficulty"],
                image_1,
                image_2,
            )
        )

    columns = 2
    rows_count = len(rows)

    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(12, 4 * rows_count),
    )

    if rows_count == 1:
        axes = [axes]

    for index, (
        pair_id,
        species,
        difficulty,
        image_1,
        image_2,
    ) in enumerate(rows):

        ax1 = axes[index][0]
        ax2 = axes[index][1]

        ax1.imshow(image_1)
        ax1.set_title(
            f"{pair_id} | {species} | "
            f"{difficulty}\nIMG1",
            fontsize=10,
        )
        ax1.axis("off")

        ax2.imshow(image_2)
        ax2.set_title(
            f"{pair_id} | {species} | "
            f"{difficulty}\nIMG2",
            fontsize=10,
        )
        ax2.axis("off")

    fig.suptitle(
        "VanDrishti Phase 2B — "
        "Ground-Truth Re-ID Crop Verification",
        fontsize=16,
    )

    fig.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_FILE,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        f"[SAVED] {OUTPUT_FILE}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()