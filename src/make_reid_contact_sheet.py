from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reid_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# First pilot sequences for manual inspection
# ------------------------------------------------------------

DAY_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day\JPEGImages"
)

NIGHT_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_night\voc_night\JPEGImages"
)


SEQUENCES = {
    "day_amurleopard_1_5": [
        DAY_DIR / f"AmurLeopard_{i}.jpg"
        for i in range(1, 6)
    ],

    "day_amurleopard_156_176": [
        DAY_DIR / f"AmurLeopard_{i}.jpg"
        for i in range(156, 177)
    ],

    "night_amurleopard_28_33": [
        NIGHT_DIR / f"AmurLeopard_{i}.jpg"
        for i in range(28, 34)
    ],

    "night_amurleopard_89_96": [
        NIGHT_DIR / f"AmurLeopard_{i}.jpg"
        for i in range(89, 97)
    ],
}


THUMB_WIDTH = 320
THUMB_HEIGHT = 220
LABEL_HEIGHT = 45

COLUMNS = 4


def create_contact_sheet(name, image_paths):

    valid_images = []

    for path in image_paths:

        if not path.exists():
            print(f"[WARNING] Missing: {path}")
            continue

        try:
            image = Image.open(path).convert("RGB")
            valid_images.append((path, image))

        except Exception as e:
            print(f"[WARNING] Could not open {path}: {e}")

    if not valid_images:
        print(f"[ERROR] No valid images for {name}")
        return

    rows = (len(valid_images) + COLUMNS - 1) // COLUMNS

    sheet_width = COLUMNS * THUMB_WIDTH
    sheet_height = rows * (THUMB_HEIGHT + LABEL_HEIGHT)

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    for index, (path, image) in enumerate(valid_images):

        row = index // COLUMNS
        col = index % COLUMNS

        x = col * THUMB_WIDTH
        y = row * (THUMB_HEIGHT + LABEL_HEIGHT)

        image.thumbnail(
            (THUMB_WIDTH - 10, THUMB_HEIGHT - 10)
        )

        image_x = x + (THUMB_WIDTH - image.width) // 2
        image_y = y + (THUMB_HEIGHT - image.height) // 2

        sheet.paste(
            image,
            (image_x, image_y),
        )

        label = path.name

        draw.text(
            (x + 5, y + THUMB_HEIGHT + 5),
            label,
            fill="black",
        )

    output_path = OUTPUT_DIR / f"{name}.jpg"

    sheet.save(
        output_path,
        quality=95,
    )

    print(f"[SAVED] {output_path}")


def main():

    print("=" * 70)
    print("VanDrishti - Re-ID Visual Review Contact Sheets")
    print("=" * 70)

    for name, paths in SEQUENCES.items():

        print(f"\nCreating: {name}")

        create_contact_sheet(
            name,
            paths,
        )

    print("\n" + "=" * 70)
    print("Contact-sheet generation completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()