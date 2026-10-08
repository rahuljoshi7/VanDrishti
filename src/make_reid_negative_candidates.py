from pathlib import Path
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reid_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAY_DIR = Path(
    r"C:\Users\ASUS\Desktop\VanDrishti_Docs\voc_day\voc_day\JPEGImages"
)


# Selected frames from visually different temporal events.
#
# IMPORTANT:
# These are ONLY candidates for inspection.
# They are NOT labelled as different individuals yet.

SELECTED_IMAGES = [
    DAY_DIR / "AmurLeopard_156.jpg",
    DAY_DIR / "AmurLeopard_157.jpg",
    DAY_DIR / "AmurLeopard_167.jpg",
    DAY_DIR / "AmurLeopard_168.jpg",
    DAY_DIR / "AmurLeopard_169.jpg",
    DAY_DIR / "AmurLeopard_170.jpg",
    DAY_DIR / "AmurLeopard_173.jpg",
    DAY_DIR / "AmurLeopard_174.jpg",
    DAY_DIR / "AmurLeopard_175.jpg",
    DAY_DIR / "AmurLeopard_176.jpg",
]


THUMB_WIDTH = 360
THUMB_HEIGHT = 250
LABEL_HEIGHT = 45
COLUMNS = 3


def create_sheet():

    valid_images = []

    for path in SELECTED_IMAGES:

        if not path.exists():
            print(f"[WARNING] Missing: {path}")
            continue

        try:
            image = Image.open(path).convert("RGB")
            valid_images.append((path, image))

        except Exception as e:
            print(f"[WARNING] Could not open {path}: {e}")

    if not valid_images:
        print("[ERROR] No valid images found.")
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

        draw.text(
            (x + 5, y + THUMB_HEIGHT + 5),
            path.name,
            fill="black",
        )

    output_path = (
        OUTPUT_DIR / "amurleopard_negative_candidates.jpg"
    )

    sheet.save(
        output_path,
        quality=95,
    )

    print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("VanDrishti - Re-ID Negative Candidate Review")
    print("=" * 70)

    print("\nThese images are candidate observations only.")
    print("No DIFFERENT labels are being assigned.")

    create_sheet()

    print("\nReview sheet generation completed.")