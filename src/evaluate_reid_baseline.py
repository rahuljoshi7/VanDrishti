from pathlib import Path
import csv

import cv2
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

CROPS_DIR = ROOT / "data" / "reid" / "crops"
METADATA_FILE = ROOT / "data" / "reid" / "metadata" / "verification_log.csv"
OUTPUT_FILE = ROOT / "data" / "reid" / "metadata" / "baseline_results.csv"


# ---------------------------------------------------------
# EXACT VanDrishti BASELINE
# ---------------------------------------------------------


def appearance_embedding(crop_rgb):
    """
    Handcrafted appearance descriptor used by the current
    VanDrishti baseline.

    Features:
        - Global HSV histograms
        - Grayscale histogram
        - Edge / texture histogram
        - Coarse spatial HSV histograms

    Output is L2-normalized.
    """

    crop = np.asarray(crop_rgb)

    if crop.size == 0:
        return np.zeros(1, dtype=np.float32)

    # Same resize used by the current baseline
    crop = cv2.resize(crop, (256, 128))

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    features = []

    # -----------------------------------------------------
    # Global HSV histograms
    # -----------------------------------------------------

    h_hist = cv2.calcHist(
        [hsv], [0], None, [32], [0, 180]
    ).flatten()

    s_hist = cv2.calcHist(
        [hsv], [1], None, [12], [0, 256]
    ).flatten()

    v_hist = cv2.calcHist(
        [hsv], [2], None, [12], [0, 256]
    ).flatten()

    features.extend(h_hist)
    features.extend(s_hist)
    features.extend(v_hist)

    # -----------------------------------------------------
    # Grayscale histogram
    # -----------------------------------------------------

    gray_hist = cv2.calcHist(
        [gray], [0], None, [32], [0, 256]
    ).flatten()

    features.extend(gray_hist)

    # -----------------------------------------------------
    # Edge / texture histogram
    # -----------------------------------------------------

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude, angle = cv2.cartToPolar(
        gx,
        gy,
        angleInDegrees=True,
    )

    edge_hist, _ = np.histogram(
        angle,
        bins=16,
        range=(0, 360),
        weights=magnitude,
    )

    features.extend(edge_hist)

    # -----------------------------------------------------
    # Coarse spatial HSV histograms
    # -----------------------------------------------------

    h, w = hsv.shape[:2]

    for rows, cols in [(2, 2)]:

        for r in range(rows):

            for c in range(cols):

                y1 = r * h // rows
                y2 = (r + 1) * h // rows

                x1 = c * w // cols
                x2 = (c + 1) * w // cols

                region = hsv[y1:y2, x1:x2]

                rh = cv2.calcHist(
                    [region],
                    [0],
                    None,
                    [32],
                    [0, 180],
                ).flatten()

                rs = cv2.calcHist(
                    [region],
                    [1],
                    None,
                    [12],
                    [0, 256],
                ).flatten()

                rv = cv2.calcHist(
                    [region],
                    [2],
                    None,
                    [12],
                    [0, 256],
                ).flatten()

                features.extend(rh)
                features.extend(rs)
                features.extend(rv)

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    vec = np.asarray(
        features,
        dtype=np.float32,
    )

    vec /= np.linalg.norm(vec) + 1e-8

    return vec


def cosine_similarity(a, b):

    denom = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    ) + 1e-8

    return float(
        np.dot(a, b) / denom
    )


def local_pattern_similarity(crop_a, crop_b):

    if crop_a is None or crop_b is None:
        return 0.0

    a = cv2.cvtColor(
        cv2.resize(crop_a, (320, 160)),
        cv2.COLOR_RGB2GRAY,
    )

    b = cv2.cvtColor(
        cv2.resize(crop_b, (320, 160)),
        cv2.COLOR_RGB2GRAY,
    )

    orb = cv2.ORB_create(
        nfeatures=600
    )

    kp_a, des_a = orb.detectAndCompute(
        a,
        None,
    )

    kp_b, des_b = orb.detectAndCompute(
        b,
        None,
    )

    if (
        des_a is None
        or des_b is None
        or len(kp_a) < 8
        or len(kp_b) < 8
    ):
        return 0.0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    raw = matcher.knnMatch(
        des_a,
        des_b,
        k=2,
    )

    good = []

    for pair in raw:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    score = min(
        1.0,
        len(good)
        / max(
            1,
            min(
                len(kp_a),
                len(kp_b),
            ) * 0.12,
        ),
    )

    return float(score)


def combined_identity_similarity(
    crop_a,
    crop_b,
):

    global_a = appearance_embedding(
        crop_a
    )

    global_b = appearance_embedding(
        crop_b
    )

    global_score = cosine_similarity(
        global_a,
        global_b,
    )

    local_score = local_pattern_similarity(
        crop_a,
        crop_b,
    )

    combined = (
        0.35 * global_score
        + 0.65 * local_score
    )

    return (
        float(combined),
        float(global_score),
        float(local_score),
    )


# ---------------------------------------------------------
# LOAD VERIFIED PAIRS
# ---------------------------------------------------------


def load_pairs():

    pairs = []

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            pairs.append(row)

    return pairs


# ---------------------------------------------------------
# CORRECTED CROP LOADING
# ---------------------------------------------------------


def load_crop(pair_id, image_index, label):
    """
    Load a verified Re-ID crop using the pair ID.

    SAME pairs:
        P001 -> data/reid/crops/same/P001_img1.jpg
        P001 -> data/reid/crops/same/P001_img2.jpg

    DIFFERENT pairs:
        N011 -> data/reid/crops/different/N011_img1.jpg
        N011 -> data/reid/crops/different/N011_img2.jpg
    """

    folder = (
        "same"
        if label == 1
        else "different"
    )

    filename = (
        f"{pair_id}_img{image_index}.jpg"
    )

    path = (
        CROPS_DIR
        / folder
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Crop not found: {path}"
        )

    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read: {path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------


def main():

    pairs = load_pairs()

    print("=" * 70)
    print("VanDrishti Re-ID Baseline Evaluation")
    print("=" * 70)

    print(
        f"Metadata file : {METADATA_FILE}"
    )

    print(
        f"Total pairs   : {len(pairs)}"
    )

    results = []

    for index, row in enumerate(
        pairs,
        start=1,
    ):

        pair_id = row["pair_id"]

        label = int(
            row["label"]
        )

        # -------------------------------------------------
        # Load verified crops using pair ID
        # -------------------------------------------------

        crop_a = load_crop(
            pair_id,
            1,
            label,
        )

        crop_b = load_crop(
            pair_id,
            2,
            label,
        )

        # -------------------------------------------------
        # Calculate Re-ID similarities
        # -------------------------------------------------

        (
            combined,
            global_score,
            local_score,
        ) = combined_identity_similarity(
            crop_a,
            crop_b,
        )

        # -------------------------------------------------
        # Decision thresholds
        # -------------------------------------------------

        if combined >= 0.88:

            prediction = 1
            decision = "LIKELY SAME"

        elif combined <= 0.68:

            prediction = 0
            decision = "LIKELY DIFFERENT"

        else:

            prediction = -1
            decision = "UNCERTAIN"

        # -------------------------------------------------
        # Store result
        # -------------------------------------------------

        results.append(
            {
                "pair_id": pair_id,
                "label": label,
                "combined_similarity": combined,
                "global_similarity": global_score,
                "local_similarity": local_score,
                "prediction": prediction,
                "decision": decision,
                "correct_binary": (
                    int(
                        prediction == label
                    )
                    if prediction != -1
                    else 0
                ),
            }
        )

        print(
            f"{index:02d}. "
            f"{pair_id} | "
            f"GT={label} | "
            f"S={combined:.3f} | "
            f"G={global_score:.3f} | "
            f"L={local_score:.3f} | "
            f"{decision}"
        )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        results[0].keys()
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    y_true = [
        r["label"]
        for r in results
    ]

    # Only definite SAME/DIFFERENT predictions
    evaluated = [
        r
        for r in results
        if r["prediction"] != -1
    ]

    y_eval = [
        r["label"]
        for r in evaluated
    ]

    y_pred = [
        r["prediction"]
        for r in evaluated
    ]

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Total pairs       : {len(results)}"
    )

    print(
        f"Definite decisions: {len(evaluated)}"
    )

    print(
        f"Uncertain         : "
        f"{len(results) - len(evaluated)}"
    )

    # -----------------------------------------------------
    # Classification metrics
    # -----------------------------------------------------

    if evaluated:

        accuracy = accuracy_score(
            y_eval,
            y_pred,
        )

        precision = precision_score(
            y_eval,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_eval,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_eval,
            y_pred,
            zero_division=0,
        )

        cm = confusion_matrix(
            y_eval,
            y_pred,
            labels=[0, 1],
        )

        print()

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1-score  : {f1:.4f}"
        )

        print()

        print("Confusion Matrix")

        print(
            "              Predicted"
        )

        print(
            "              DIFFERENT SAME"
        )

        print(
            f"Actual DIFF   "
            f"{cm[0, 0]:8d} "
            f"{cm[0, 1]:4d}"
        )

        print(
            f"Actual SAME   "
            f"{cm[1, 0]:8d} "
            f"{cm[1, 1]:4d}"
        )

    # -----------------------------------------------------
    # ROC-AUC
    # -----------------------------------------------------

    # ROC-AUC uses continuous similarity,
    # not the thresholded prediction.

    scores = [
        r["combined_similarity"]
        for r in results
    ]

    try:

        auc = roc_auc_score(
            y_true,
            scores,
        )

        print()

        print(
            f"ROC-AUC       : {auc:.4f}"
        )

    except ValueError:

        print()

        print(
            "ROC-AUC       : unavailable"
        )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print()

    print(
        f"Results saved : {OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()