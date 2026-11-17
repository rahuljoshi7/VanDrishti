import os
import tempfile
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# VanDrishti
# AI-Based Multi-Camera Wildlife Monitoring
# ============================================================

st.set_page_config(
    page_title="VanDrishti | Wildlife Intelligence",
    page_icon="🦌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "yolo26n_baseline_v2"
    / "weights"
    / "best.pt"
)


# ============================================================
# CONSTANTS
# ============================================================

TOTAL_IMAGES = 25657
TRAIN_IMAGES = 20634
VAL_IMAGES = 5023
TOTAL_CLASSES = 17

PRECISION = 93.5
RECALL = 90.8
MAP50 = 95.1
MAP50_95 = 81.1

REID_SAME_THRESHOLD = 0.88
REID_DIFFERENT_THRESHOLD = 0.68
QUALITY_THRESHOLD = 0.45


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
# CSS
# ============================================================

st.markdown(
    """
<style>

.vd-header {
    padding: 28px 32px;
    border-radius: 16px;
    background: linear-gradient(
        120deg,
        #0d2818 0%,
        #1b4332 55%,
        #2d6a4f 100%
    );
    margin-bottom: 24px;
    box-shadow: 0 6px 24px rgba(13, 40, 24, 0.25);
}

.vd-eyebrow {
    color: #74c69d;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.vd-title {
    color: #f4fbf7;
    font-size: 42px;
    font-weight: 800;
    margin-top: 5px;
}

.vd-subtitle {
    color: rgba(244, 251, 247, 0.82);
    font-size: 17px;
    margin-top: 4px;
}

.vd-card {
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,.20);
    background: rgba(128,128,128,.035);
    margin-bottom: 14px;
}

.vd-small {
    font-size: 13px;
    opacity: .70;
}

.pipeline {
    padding: 14px 16px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.20);
    background: rgba(128,128,128,.04);
    line-height: 1.9;
}

.pipeline-done {
    color: #2f9e5e;
    font-weight: 600;
}

.pipeline-future {
    color: #999999;
}

.species-chip {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 999px;
    background: rgba(64,145,108,.16);
    color: #2f9e5e;
    margin: 3px 5px 3px 0;
    font-size: 13px;
    font-weight: 600;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="vd-header">
<div class="vd-eyebrow">AI-Based Wildlife Intelligence Platform</div>
<div class="vd-title">🦌 VanDrishti</div>
<div class="vd-subtitle">Multi-Camera Wildlife Monitoring & Unique Animal Identification</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    try:
        import torch

        if torch.cuda.is_available():

            return 0, "NVIDIA GPU"

    except Exception:
        pass

    return "cpu", "CPU"


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"YOLO model not found:\n{MODEL_PATH}"
        )

    return YOLO(
        str(MODEL_PATH)
    )


def get_class_name(
    model,
    class_id
):

    names = model.names

    if isinstance(names, dict):

        return names.get(
            int(class_id),
            str(class_id)
        )

    return names[int(class_id)]


# ============================================================
# NLM PREPROCESSING
# ============================================================

def apply_nlm(
    image_rgb
):

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        image_bgr,
        None,
        h=6,
        hColor=6,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    return cv2.cvtColor(
        denoised_bgr,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# IMAGE DETECTION
# ============================================================

def run_detection(
    model,
    image_rgb,
    use_nlm=True
):

    if use_nlm:

        processed_image = apply_nlm(
            image_rgb
        )

    else:

        processed_image = image_rgb

    device, _ = get_device()

    result = model.predict(
        source=processed_image,
        device=device,
        conf=st.session_state.conf,
        verbose=False,
    )[0]

    annotated = cv2.cvtColor(
        result.plot(),
        cv2.COLOR_BGR2RGB
    )

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls.item()
            )

            confidence = float(
                box.conf.item()
            )

            bbox = [
                int(x)
                for x in box.xyxy[0].tolist()
            ]

            detections.append(
                {
                    "class": get_class_name(
                        model,
                        class_id
                    ),
                    "confidence": confidence,
                    "bbox": bbox,
                }
            )

    return (
        annotated,
        processed_image,
        detections
    )


# ============================================================
# CROP
# ============================================================

def crop_from_bbox(
    image_rgb,
    bbox
):

    x1, y1, x2, y2 = bbox

    height, width = image_rgb.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:

        return np.empty(
            (0, 0, 3),
            dtype=np.uint8
        )

    return image_rgb[
        y1:y2,
        x1:x2
    ]


# ============================================================
# APPEARANCE EMBEDDING
# ============================================================

def appearance_embedding(
    crop_rgb
):

    """
    Handcrafted baseline appearance descriptor.

    Components:
    - HSV histograms
    - grayscale histogram
    - edge histogram
    - spatial HSV histograms

    This is a baseline descriptor, not a trained
    deep Re-ID network.
    """

    if crop_rgb is None:
        return np.zeros(
            1,
            dtype=np.float32
        )

    if crop_rgb.size == 0:
        return np.zeros(
            1,
            dtype=np.float32
        )

    crop = cv2.resize(
        crop_rgb,
        (256, 128)
    )

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_RGB2HSV
    )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_RGB2GRAY
    )

    features = []

    # --------------------------------------------------------
    # Global HSV
    # --------------------------------------------------------

    for channel, bins, value_range in [
        (0, 24, [0, 180]),
        (1, 16, [0, 256]),
        (2, 16, [0, 256]),
    ]:

        hist = cv2.calcHist(
            [hsv],
            [channel],
            None,
            [bins],
            value_range
        ).flatten()

        hist = hist / (
            hist.sum() + 1e-8
        )

        features.extend(
            hist.tolist()
        )

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray_hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [32],
        [0, 256]
    ).flatten()

    gray_hist = gray_hist / (
        gray_hist.sum() + 1e-8
    )

    features.extend(
        gray_hist.tolist()
    )

    # --------------------------------------------------------
    # Edge histogram
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_hist = cv2.calcHist(
        [edges],
        [0],
        None,
        [16],
        [0, 256]
    ).flatten()

    edge_hist = edge_hist / (
        edge_hist.sum() + 1e-8
    )

    features.extend(
        edge_hist.tolist()
    )

    # --------------------------------------------------------
    # Spatial HSV
    # --------------------------------------------------------

    for y in range(2):

        for x in range(4):

            cell = hsv[
                y * hsv.shape[0] // 2:
                (y + 1) * hsv.shape[0] // 2,

                x * hsv.shape[1] // 4:
                (x + 1) * hsv.shape[1] // 4
            ]

            for channel, bins, value_range in [
                (0, 12, [0, 180]),
                (1, 8, [0, 256]),
                (2, 8, [0, 256]),
            ]:

                hist = cv2.calcHist(
                    [cell],
                    [channel],
                    None,
                    [bins],
                    value_range
                ).flatten()

                hist = hist / (
                    hist.sum() + 1e-8
                )

                features.extend(
                    hist.tolist()
                )

    vector = np.asarray(
        features,
        dtype=np.float32
    )

    vector /= (
        np.linalg.norm(vector) + 1e-8
    )

    return vector


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    vector_a,
    vector_b
):

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    ) + 1e-8

    return float(
        np.dot(
            vector_a,
            vector_b
        ) / denominator
    )


# ============================================================
# ORB LOCAL SIMILARITY
# ============================================================

def local_pattern_similarity(
    crop_a,
    crop_b
):

    if (
        crop_a is None
        or crop_b is None
        or crop_a.size == 0
        or crop_b.size == 0
    ):

        return 0.0

    image_a = cv2.cvtColor(
        cv2.resize(
            crop_a,
            (320, 160)
        ),
        cv2.COLOR_RGB2GRAY
    )

    image_b = cv2.cvtColor(
        cv2.resize(
            crop_b,
            (320, 160)
        ),
        cv2.COLOR_RGB2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=600
    )

    keypoints_a, descriptors_a = (
        orb.detectAndCompute(
            image_a,
            None
        )
    )

    keypoints_b, descriptors_b = (
        orb.detectAndCompute(
            image_b,
            None
        )
    )

    if (
        descriptors_a is None
        or descriptors_b is None
        or len(keypoints_a) < 8
        or len(keypoints_b) < 8
    ):

        return 0.0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    matches = matcher.knnMatch(
        descriptors_a,
        descriptors_b,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) != 2:
            continue

        first, second = pair

        if first.distance < (
            0.75 * second.distance
        ):

            good_matches.append(
                first
            )

    denominator = (
        min(
            len(keypoints_a),
            len(keypoints_b)
        ) * 0.12
    )

    score = min(
        1.0,
        len(good_matches)
        / max(1, denominator)
    )

    return float(score)


# ============================================================
# COMBINED RE-ID SIMILARITY
# ============================================================

def combined_identity_similarity(
    crop_a,
    crop_b
):

    embedding_a = appearance_embedding(
        crop_a
    )

    embedding_b = appearance_embedding(
        crop_b
    )

    global_score = cosine_similarity(
        embedding_a,
        embedding_b
    )

    local_score = local_pattern_similarity(
        crop_a,
        crop_b
    )

    combined_score = (
        0.35 * global_score
        + 0.65 * local_score
    )

    return (
        float(combined_score),
        float(global_score),
        float(local_score)
    )


# ============================================================
# IMAGE QUALITY
# ============================================================

def crop_quality(
    image_rgb,
    bbox
):

    height, width = image_rgb.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    crop = image_rgb[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:

        return {
            "score": 0.0,
            "label": "INSUFFICIENT EVIDENCE",
            "area_score": 0.0,
            "sharpness_score": 0.0,
        }

    area_ratio = (
        (x2 - x1)
        * (y2 - y1)
    ) / max(
        1,
        width * height
    )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_RGB2GRAY
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    area_score = min(
        area_ratio / 0.12,
        1.0
    )

    sharpness_score = min(
        sharpness / 180.0,
        1.0
    )

    score = (
        0.55 * area_score
        + 0.45 * sharpness_score
    )

    label = (
        "SUFFICIENT EVIDENCE"
        if score >= QUALITY_THRESHOLD
        else "INSUFFICIENT EVIDENCE"
    )

    return {
        "score": score,
        "label": label,
        "area_score": area_score,
        "sharpness_score": sharpness_score,
    }


# ============================================================
# RE-ID DECISION
# ============================================================

def reid_decision(
    similarity,
    quality_a,
    quality_b,
    species_a=None,
    species_b=None
):

    # Species mismatch = cannot be same animal
    if (
        species_a is not None
        and species_b is not None
        and species_a != species_b
    ):

        return "LIKELY DIFFERENT"

    minimum_quality = min(
        quality_a["score"],
        quality_b["score"]
    )

    if minimum_quality < 0.30:

        return "INSUFFICIENT EVIDENCE"

    if similarity >= REID_SAME_THRESHOLD:

        return "LIKELY SAME"

    if similarity <= REID_DIFFERENT_THRESHOLD:

        return "LIKELY DIFFERENT"

    return "UNCERTAIN"


# ============================================================
# VIDEO TRACKING
# ============================================================

def process_video(
    model,
    video_bytes,
    use_nlm=False
):

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_file.write(
        video_bytes
    )

    input_file.close()

    cap = cv2.VideoCapture(
        input_file.name
    )

    if not cap.isOpened():

        os.unlink(
            input_file.name
        )

        raise RuntimeError(
            "Could not open uploaded video."
        )

    fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 20
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    output_file.close()

    writer = cv2.VideoWriter(
        output_file.name,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height)
    )

    seen_tracks = {}

    frame_count = 0

    device, _ = get_device()

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            processed_frame = frame

            if use_nlm:

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                frame_rgb = apply_nlm(
                    frame_rgb
                )

                processed_frame = cv2.cvtColor(
                    frame_rgb,
                    cv2.COLOR_RGB2BGR
                )

            results = model.track(
                source=processed_frame,
                persist=True,
                tracker="bytetrack.yaml",
                device=device,
                conf=st.session_state.conf,
                verbose=False,
            )

            result = results[0]

            if result.boxes is not None:

                ids = result.boxes.id

                for index, box in enumerate(
                    result.boxes
                ):

                    class_id = int(
                        box.cls.item()
                    )

                    if ids is not None:

                        track_id = int(
                            ids[index].item()
                        )

                    else:

                        track_id = -1

                    if track_id >= 0:

                        key = track_id

                    else:

                        key = (
                            f"frame_"
                            f"{frame_count}_"
                            f"{index}"
                        )

                    seen_tracks[key] = (
                        get_class_name(
                            model,
                            class_id
                        )
                    )

            writer.write(
                result.plot()
            )

            frame_count += 1

    finally:

        cap.release()
        writer.release()

        if os.path.exists(
            input_file.name
        ):

            os.unlink(
                input_file.name
            )

    return (
        output_file.name,
        frame_count,
        seen_tracks
    )


# ============================================================
# MODEL LOAD
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        "Trained YOLO26n model was not found."
    )

    st.code(
        str(MODEL_PATH)
    )

    st.stop()


try:

    model = load_model()

except Exception as error:

    st.error(
        f"Unable to load model:\n{error}"
    )

    st.stop()


device, device_label = get_device()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🦌 VanDrishti"
)

st.sidebar.caption(
    "Wildlife Intelligence Platform"
)

st.sidebar.divider()


mode = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Single Image",
        "Video Tracking",
        "Multi-Camera Preview",
        "Identity Evidence",
        "Re-ID Comparison",
        "Unique Count Demo",
    ]
)


# ============================================================
# SIDEBAR CONFIDENCE
# ============================================================

st.sidebar.divider()

st.sidebar.markdown(
    "### Detection Settings"
)

confidence = st.sidebar.slider(
    "Detection confidence",
    0.10,
    0.90,
    0.25,
    0.05
)

st.session_state.conf = confidence


# ============================================================
# SIDEBAR SYSTEM
# ============================================================

st.sidebar.divider()

st.sidebar.markdown(
    "### 🖥️ System"
)

side_col1, side_col2 = (
    st.sidebar.columns(2)
)

side_col1.metric(
    "Inference",
    device_label
)

side_col2.metric(
    "Classes",
    TOTAL_CLASSES
)

st.sidebar.caption(
    "YOLO26n • ByteTrack • Baseline Re-ID"
)


# ============================================================
# SIDEBAR PIPELINE
# ============================================================

st.sidebar.divider()

st.sidebar.markdown(
    "### 🔗 System Pipeline"
)

pipeline_html = """
<div class="pipeline">

<div class="pipeline-done">
📷 Camera / Image Input ✓
</div>

<div class="pipeline-done">
🧹 NLM Image Preprocessing ✓
</div>

<div class="pipeline-done">
🎯 YOLO26n Detection ✓
</div>

<div class="pipeline-done">
🐾 Species Classification ✓
</div>

<div class="pipeline-done">
🧭 ByteTrack Tracking ✓
</div>

<div class="pipeline-done">
🧬 Baseline Re-ID ✓
</div>

<div class="pipeline-done">
🧠 Identity Decision Logic ✓
</div>

<div class="pipeline-done">
🆔 In-Session Identity Registry ✓
</div>

<div class="pipeline-done">
📊 Unique Count Prototype ✓
</div>

<div class="pipeline-future">
🧠 Learned Re-ID → Future
</div>

<div class="pipeline-future">
🧩 Part-Aware Re-ID → Future
</div>

<div class="pipeline-future">
⚙️ ANFIS Fusion → Future
</div>

</div>
"""

st.sidebar.markdown(
    pipeline_html,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if mode == "Dashboard":

    st.subheader(
        "📊 VanDrishti Dashboard"
    )

    st.caption(
        "AI-based wildlife detection, tracking, "
        "identity evidence and unique-counting prototype."
    )

    # --------------------------------------------------------
    # Dataset metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Wildlife Classes",
        TOTAL_CLASSES
    )

    col2.metric(
        "Total Images",
        f"{TOTAL_IMAGES:,}"
    )

    col3.metric(
        "Training Images",
        f"{TRAIN_IMAGES:,}"
    )

    col4.metric(
        "Validation Images",
        f"{VAL_IMAGES:,}"
    )

    st.divider()

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    left, right = (
        st.columns(
            [1.35, 1]
        )
    )

    with left:

        st.markdown(
            "### 🏗️ VanDrishti Architecture"
        )

        st.markdown(
            """
**📷 Multiple Wildlife Cameras**

↓

**🧹 NLM Noise Filtering + Image Enhancement**

↓

**🎯 YOLO26n Object Detection**

↓

**🐾 Species Classification**

↓

**🧭 ByteTrack Object Tracking**

↓

**🧬 Animal Re-ID**

↓

**🧠 Identity Decision Fusion**

↓

**🆔 Unique Animal Identity**

↓

**📊 Category-wise Unique Population Count**

↓

**🖥️ Monitoring Dashboard**
"""
        )

    with right:

        st.markdown(
            "### 📌 Implementation Status"
        )

        implementation = [
            ("Dataset Preparation", "DONE"),
            ("VOC → YOLO Conversion", "DONE"),
            ("Train / Validation Split", "DONE"),
            ("NLM Preprocessing", "DONE"),
            ("YOLO26n Detector", "DONE"),
            ("ByteTrack Tracking", "DONE"),
            ("Baseline Re-ID", "DONE"),
            ("Identity Evidence", "DONE"),
            ("Identity Registry", "PROTOTYPE"),
            ("Unique Count Demo", "PROTOTYPE"),
            ("Learned Re-ID", "FUTURE"),
            ("Part-Aware Re-ID", "FUTURE"),
            ("ANFIS Fusion", "FUTURE"),
        ]

        for name, status in implementation:

            c1, c2 = st.columns(
                [3, 1]
            )

            with c1:

                st.write(
                    f"**{name}**"
                )

            with c2:

                if status == "DONE":

                    st.success(
                        "DONE"
                    )

                elif status == "PROTOTYPE":

                    st.warning(
                        "PROTO"
                    )

                else:

                    st.info(
                        "FUTURE"
                    )

    # --------------------------------------------------------
    # YOLO metrics
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### 📈 YOLO26n Validation Performance"
    )

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    metric1.metric(
        "Precision",
        f"{PRECISION}%"
    )

    metric2.metric(
        "Recall",
        f"{RECALL}%"
    )

    metric3.metric(
        "mAP@50",
        f"{MAP50}%"
    )

    metric4.metric(
        "mAP@50-95",
        f"{MAP50_95}%"
    )

    st.caption(
        "Custom-trained YOLO26n model evaluated on the project validation split."
    )

    # --------------------------------------------------------
    # Core objective
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### 🎯 Core Objective"
    )

    st.info(
        "Counting animals, not detections."
    )

    st.write(
        "VanDrishti aims to distinguish repeated "
        "observations of the same animal across "
        "multiple cameras and generate more reliable "
        "category-wise unique animal counts."
    )


# ============================================================
# SINGLE IMAGE
# ============================================================

elif mode == "Single Image":

    st.subheader(
        "🔎 Wildlife Detection"
    )

    st.caption(
        "NLM preprocessing → YOLO26n detection → "
        "species classification."
    )

    uploaded = st.file_uploader(
        "Upload wildlife image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    use_nlm = st.checkbox(
        "Apply NLM preprocessing",
        value=True
    )

    if uploaded:

        image = Image.open(
            uploaded
        ).convert("RGB")

        image_rgb = np.array(
            image
        )

        with st.spinner(
            "Running wildlife detection..."
        ):

            (
                annotated,
                processed,
                detections
            ) = run_detection(
                model,
                image_rgb,
                use_nlm
            )

        result_col, info_col = (
            st.columns(
                [2.1, 1]
            )
        )

        with result_col:

            st.markdown(
                "### 🎯 Detection Result"
            )

            st.image(
                annotated,
                use_container_width=True
            )

        with info_col:

            st.metric(
                "Animals Detected",
                len(detections)
            )

            counts = Counter(
                d["class"]
                for d in detections
            )

            st.markdown(
                "### 🐾 Species"
            )

            for species, count in sorted(
                counts.items()
            ):

                st.write(
                    f"**{species}** — {count}"
                )

            st.markdown(
                "### 🎯 Confidence"
            )

            for index, detection in enumerate(
                detections,
                start=1
            ):

                st.progress(
                    detection["confidence"],
                    text=(
                        f"{index}. "
                        f"{detection['class']} "
                        f"— "
                        f"{detection['confidence']:.2f}"
                    )
                )

        st.divider()

        st.markdown(
            "### 🧹 Preprocessing"
        )

        p1, p2 = st.columns(2)

        with p1:

            st.image(
                image_rgb,
                caption="Original",
                use_container_width=True
            )

        with p2:

            st.image(
                processed,
                caption="NLM Preprocessed",
                use_container_width=True
            )


# ============================================================
# VIDEO TRACKING
# ============================================================

elif mode == "Video Tracking":

    st.subheader(
        "🎥 YOLO26n + ByteTrack"
    )

    st.caption(
        "Detect and track wildlife across video frames."
    )

    uploaded_video = st.file_uploader(
        "Upload wildlife video",
        type=[
            "mp4",
            "avi",
            "mov"
        ]
    )

    use_nlm_video = st.checkbox(
        "Apply NLM preprocessing to frames",
        value=False
    )

    if uploaded_video:

        if st.button(
            "▶ Start Tracking",
            type="primary"
        ):

            with st.spinner(
                "Running YOLO26n + ByteTrack..."
            ):

                try:

                    (
                        output_path,
                        frame_count,
                        tracks
                    ) = process_video(
                        model,
                        uploaded_video.getvalue(),
                        use_nlm_video
                    )

                    with open(
                        output_path,
                        "rb"
                    ) as video_file:

                        video_data = (
                            video_file.read()
                        )

                    st.success(
                        f"Processed {frame_count} frames."
                    )

                    st.video(
                        video_data
                    )

                    st.divider()

                    m1, m2, m3 = (
                        st.columns(3)
                    )

                    m1.metric(
                        "Frames",
                        frame_count
                    )

                    m2.metric(
                        "Track IDs",
                        len(tracks)
                    )

                    m3.metric(
                        "Device",
                        device_label
                    )

                    species = Counter(
                        tracks.values()
                    )

                    st.markdown(
                        "### 🐾 Tracked Species"
                    )

                    for name, count in sorted(
                        species.items()
                    ):

                        st.write(
                            f"**{name}** — "
                            f"{count} track(s)"
                        )

                    st.warning(
                        "ByteTrack IDs are temporary "
                        "video-level tracking identities. "
                        "Cross-camera identity requires the Re-ID layer."
                    )

                    if os.path.exists(
                        output_path
                    ):

                        os.unlink(
                            output_path
                        )

                except Exception as error:

                    st.error(
                        f"Tracking failed:\n{error}"
                    )


# ============================================================
# MULTI-CAMERA PREVIEW
# ============================================================

elif mode == "Multi-Camera Preview":

    st.subheader(
        "📷 Multi-Camera Observation"
    )

    st.caption(
        "Simulate observations arriving from multiple "
        "wildlife surveillance cameras."
    )

    files = st.file_uploader(
        "Upload up to 6 camera images",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True
    )

    use_nlm_multi = st.checkbox(
        "Apply NLM preprocessing",
        value=True
    )

    if files:

        files = files[:6]

        columns = st.columns(3)

        total_detections = 0

        species_counts = Counter()

        for index, uploaded in enumerate(
            files
        ):

            image = Image.open(
                uploaded
            ).convert("RGB")

            image_rgb = np.array(
                image
            )

            with st.spinner(
                f"Processing Camera {index + 1}..."
            ):

                (
                    annotated,
                    _,
                    detections
                ) = run_detection(
                    model,
                    image_rgb,
                    use_nlm_multi
                )

            total_detections += len(
                detections
            )

            for detection in detections:

                species_counts[
                    detection["class"]
                ] += 1

            with columns[
                index % 3
            ]:

                st.markdown(
                    f"### 📹 Camera {index + 1}"
                )

                st.image(
                    annotated,
                    use_container_width=True
                )

                st.write(
                    f"Detections: "
                    f"**{len(detections)}**"
                )

        st.divider()

        m1, m2, m3 = (
            st.columns(3)
        )

        m1.metric(
            "Cameras",
            len(files)
        )

        m2.metric(
            "Total Detections",
            total_detections
        )

        m3.metric(
            "Unique Animals",
            "Re-ID Required"
        )

        st.markdown(
            "### 📊 Combined Observations"
        )

        for species, count in sorted(
            species_counts.items()
        ):

            st.write(
                f"**{species}** — {count}"
            )

        st.info(
            "Multiple cameras may observe the same "
            "individual. The Re-ID layer is used to "
            "determine whether observations belong to "
            "the same animal."
        )


# ============================================================
# IDENTITY EVIDENCE
# ============================================================

elif mode == "Identity Evidence":

    st.subheader(
        "🧠 Identity Evidence"
    )

    st.caption(
        "Evaluate whether two wildlife observations "
        "contain sufficient visual evidence for identity comparison."
    )

    uploaded_files = st.file_uploader(
        "Upload two wildlife images",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        if len(uploaded_files) != 2:

            st.warning(
                "Please upload exactly two images."
            )

        else:

            images = []

            for file in uploaded_files:

                image = Image.open(
                    file
                ).convert("RGB")

                images.append(
                    np.array(image)
                )

            detections = []

            for image in images:

                _, _, result = run_detection(
                    model,
                    image,
                    use_nlm=True
                )

                detections.append(
                    result
                )

            c1, c2 = st.columns(2)

            for index in range(2):

                with (
                    c1
                    if index == 0
                    else c2
                ):

                    st.markdown(
                        f"### Observation {index + 1}"
                    )

                    st.image(
                        images[index],
                        use_container_width=True
                    )

                    if detections[index]:

                        for detection in detections[index]:

                            quality = crop_quality(
                                images[index],
                                detection["bbox"]
                            )

                            st.write(
                                f"Species: "
                                f"**{detection['class']}**"
                            )

                            st.write(
                                f"Detection confidence: "
                                f"**{detection['confidence']:.2f}**"
                            )

                            st.write(
                                f"Evidence quality: "
                                f"**{quality['score']:.2f}**"
                            )

                            if quality["score"] >= QUALITY_THRESHOLD:

                                st.success(
                                    "Sufficient visual evidence"
                                )

                            else:

                                st.warning(
                                    "Insufficient visual evidence"
                                )

                    else:

                        st.warning(
                            "No animal detected."
                        )


# ============================================================
# RE-ID COMPARISON
# ============================================================

elif mode == "Re-ID Comparison":

    st.subheader(
        "🧬 Wildlife Re-ID Comparison"
    )

    st.caption(
        "Baseline appearance + local pattern similarity."
    )

    uploaded_files = st.file_uploader(
        "Upload two observations",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        if len(uploaded_files) != 2:

            st.warning(
                "Upload exactly two images."
            )

        else:

            images = []

            for file in uploaded_files:

                image = Image.open(
                    file
                ).convert("RGB")

                images.append(
                    np.array(image)
                )

            detections = []

            for image in images:

                _, _, result = run_detection(
                    model,
                    image,
                    use_nlm=True
                )

                detections.append(
                    result
                )

            if (
                not detections[0]
                or not detections[1]
            ):

                st.error(
                    "An animal must be detected in both images."
                )

            else:

                detection_a = max(
                    detections[0],
                    key=lambda x: x["confidence"]
                )

                detection_b = max(
                    detections[1],
                    key=lambda x: x["confidence"]
                )

                crop_a = crop_from_bbox(
                    images[0],
                    detection_a["bbox"]
                )

                crop_b = crop_from_bbox(
                    images[1],
                    detection_b["bbox"]
                )

                quality_a = crop_quality(
                    images[0],
                    detection_a["bbox"]
                )

                quality_b = crop_quality(
                    images[1],
                    detection_b["bbox"]
                )

                (
                    combined,
                    global_score,
                    local_score
                ) = combined_identity_similarity(
                    crop_a,
                    crop_b
                )

                decision = reid_decision(
                    combined,
                    quality_a,
                    quality_b,
                    detection_a["class"],
                    detection_b["class"]
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.image(
                        images[0],
                        caption="Observation A",
                        use_container_width=True
                    )

                    st.write(
                        f"Species: **{detection_a['class']}**"
                    )

                    st.write(
                        f"Quality: **{quality_a['score']:.2f}**"
                    )

                with c2:

                    st.image(
                        images[1],
                        caption="Observation B",
                        use_container_width=True
                    )

                    st.write(
                        f"Species: **{detection_b['class']}**"
                    )

                    st.write(
                        f"Quality: **{quality_b['score']:.2f}**"
                    )

                st.divider()

                st.markdown(
                    "### 📊 Similarity Analysis"
                )

                m1, m2, m3 = (
                    st.columns(3)
                )

                m1.metric(
                    "Global Appearance",
                    f"{global_score:.3f}"
                )

                m2.metric(
                    "Local Pattern",
                    f"{local_score:.3f}"
                )

                m3.metric(
                    "Combined Similarity",
                    f"{combined:.3f}"
                )

                st.markdown(
                    "### 🧠 Identity Decision"
                )

                if decision == "LIKELY SAME":

                    st.success(
                        "LIKELY SAME INDIVIDUAL"
                    )

                elif decision == "LIKELY DIFFERENT":

                    st.info(
                        "LIKELY DIFFERENT INDIVIDUALS"
                    )

                elif decision == "INSUFFICIENT EVIDENCE":

                    st.error(
                        "INSUFFICIENT EVIDENCE"
                    )

                else:

                    st.warning(
                        "UNCERTAIN"
                    )

                st.caption(
                    "Baseline thresholds: "
                    f"SAME ≥ {REID_SAME_THRESHOLD:.2f}, "
                    f"DIFFERENT ≤ {REID_DIFFERENT_THRESHOLD:.2f}. "
                    "This is an experimental handcrafted baseline, "
                    "not the final learned Re-ID model."
                )


# ============================================================
# UNIQUE COUNT DEMO
# ============================================================

elif mode == "Unique Count Demo":

    st.subheader(
        "🆔 Unique Animal Count Demo"
    )

    st.caption(
        "Prototype for combining multiple camera observations "
        "into an in-session identity registry."
    )

    if "identity_registry" not in st.session_state:

        st.session_state.identity_registry = []

    if "next_identity_id" not in st.session_state:

        st.session_state.next_identity_id = 1

    files = st.file_uploader(
        "Upload multiple wildlife observations",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True
    )

    if files:

        observations = []

        for file in files:

            image = Image.open(
                file
            ).convert("RGB")

            image_rgb = np.array(
                image
            )

            _, _, detections = run_detection(
                model,
                image_rgb,
                use_nlm=True
            )

            if not detections:

                continue

            for detection in detections:

                crop = crop_from_bbox(
                    image_rgb,
                    detection["bbox"]
                )

                quality = crop_quality(
                    image_rgb,
                    detection["bbox"]
                )

                embedding = appearance_embedding(
                    crop
                )

                observations.append(
                    {
                        "image": image_rgb,
                        "species": detection["class"],
                        "confidence": detection["confidence"],
                        "quality": quality,
                        "embedding": embedding,
                        "crop": crop,
                    }
                )

        st.metric(
            "Animal Observations",
            len(observations)
        )

        if st.button(
            "Run Identity Assignment",
            type="primary"
        ):

            # ------------------------------------------------
            # Reset registry for this run
            # ------------------------------------------------

            st.session_state.identity_registry = []

            st.session_state.next_identity_id = 1

            assignments = []

            # ------------------------------------------------
            # Process observations
            # ------------------------------------------------

            for observation in observations:

                best_identity = None

                best_similarity = -1

                for identity in (
                    st.session_state.identity_registry
                ):

                    if identity["species"] != (
                        observation["species"]
                    ):

                        continue

                    similarity = cosine_similarity(
                        observation["embedding"],
                        identity["embedding"]
                    )

                    if similarity > best_similarity:

                        best_similarity = similarity

                        best_identity = identity

                # ------------------------------------------------
                # Match existing identity
                # ------------------------------------------------

                if (
                    best_identity is not None
                    and best_similarity >= REID_SAME_THRESHOLD
                    and observation["quality"]["score"]
                    >= QUALITY_THRESHOLD
                ):

                    best_identity["observations"] += 1

                    assignments.append(
                        {
                            "species": observation["species"],
                            "identity": best_identity["id"],
                            "similarity": best_similarity,
                            "decision": "SAME",
                        }
                    )

                # ------------------------------------------------
                # Create new identity
                # ------------------------------------------------

                else:

                    identity_id = (
                        f"ANIMAL_"
                        f"{st.session_state.next_identity_id:03d}"
                    )

                    st.session_state.next_identity_id += 1

                    new_identity = {
                        "id": identity_id,
                        "species": observation["species"],
                        "embedding": observation["embedding"],
                        "observations": 1,
                    }

                    st.session_state.identity_registry.append(
                        new_identity
                    )

                    assignments.append(
                        {
                            "species": observation["species"],
                            "identity": identity_id,
                            "similarity": best_similarity
                            if best_identity is not None
                            else 0.0,
                            "decision": "NEW",
                        }
                    )

            # ------------------------------------------------
            # Results
            # ------------------------------------------------

            st.divider()

            st.markdown(
                "### 📊 Identity Assignment"
            )

            for index, assignment in enumerate(
                assignments,
                start=1
            ):

                if assignment["decision"] == "SAME":

                    st.success(
                        f"Observation {index}: "
                        f"{assignment['species']} → "
                        f"{assignment['identity']} "
                        f"(similarity "
                        f"{assignment['similarity']:.3f})"
                    )

                else:

                    st.info(
                        f"Observation {index}: "
                        f"{assignment['species']} → "
                        f"{assignment['identity']} "
                        f"(NEW)"
                    )

            # ------------------------------------------------
            # Unique count
            # ------------------------------------------------

            st.divider()

            st.markdown(
                "### 🐾 Unique Population Estimate"
            )

            unique_counts = Counter(
                identity["species"]
                for identity
                in st.session_state.identity_registry
            )

            total_unique = len(
                st.session_state.identity_registry
            )

            st.metric(
                "Total Unique Individuals",
                total_unique
            )

            for species, count in sorted(
                unique_counts.items()
            ):

                st.write(
                    f"**{species}** — {count}"
                )

            st.warning(
                "This is an in-session prototype. "
                "The final system will use a stronger learned "
                "Re-ID model, identity fusion and persistent storage."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VanDrishti • AI-Based Multi-Camera Wildlife Monitoring"
)

st.caption(
    "Counting animals, not detections."
)