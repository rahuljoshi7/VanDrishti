import os
import tempfile
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ------------------------------------------------------------
# VanDrishti SIH 2026 - GUI Prototype
# Detection + Tracking + Multi-Camera Observation Dashboard
# ------------------------------------------------------------

st.set_page_config(
    page_title="VanDrishti | Wildlife Intelligence",
    page_icon="🦌",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT / "runs" / "yolo26n_baseline_v2" / "weights" / "best.pt"
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    :root {
        --vd-forest-900: #0d2818;
        --vd-forest-700: #1b4332;
        --vd-forest-500: #2d6a4f;
        --vd-forest-300: #74c69d;
        --vd-amber: #d99a3d;
        --vd-amber-soft: rgba(217, 154, 61, 0.14);
        --vd-red-soft: rgba(214, 69, 69, 0.14);
        --vd-green-soft: rgba(64, 145, 108, 0.16);
        --vd-border: rgba(128, 128, 128, 0.20);
        --vd-radius: 14px;
    }

    html, body, [class*="css"] {
        font-family: "Source Sans Pro", "Segoe UI", sans-serif;
    }

    .vd-header {
        padding: 26px 30px;
        border-radius: var(--vd-radius);
        background: linear-gradient(120deg, var(--vd-forest-900) 0%, var(--vd-forest-700) 55%, var(--vd-forest-500) 100%);
        margin-bottom: 22px;
        box-shadow: 0 6px 24px rgba(13, 40, 24, 0.25);
    }
    .vd-eyebrow {
        color: var(--vd-forest-300);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0;
        color: #f4fbf7;
        line-height: 1.1;
    }
    .subtitle {
        font-size: 1.05rem;
        color: rgba(244, 251, 247, 0.82);
        margin-top: 6px;
    }

    .status-card {
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid var(--vd-border);
        margin-bottom: 10px;
        background: rgba(128, 128, 128, 0.04);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        transition: border-color .15s ease;
    }
    .status-card:hover { border-color: rgba(128,128,128,.45); }
    .status-name { font-weight: 600; font-size: 0.94rem; }

    .badge {
        display: inline-block;
        padding: 3px 11px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        white-space: nowrap;
    }
    .badge-done   { background: var(--vd-green-soft); color: #2f9e5e; }
    .badge-proto  { background: var(--vd-amber-soft); color: var(--vd-amber); }
    .badge-next   { background: rgba(128,128,128,.14); color: #8a8a8a; }

    .small-note { font-size: .82rem; opacity: .72; }

    .pipeline {
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid var(--vd-border);
        background: rgba(128, 128, 128, 0.04);
        line-height: 2.0;
        font-size: 0.9rem;
    }
    .pipeline-step-done { color: #2f9e5e; font-weight: 600; }
    .pipeline-step-pending { opacity: 0.55; }

    .vd-card {
        padding: 18px 20px;
        border-radius: var(--vd-radius);
        border: 1px solid var(--vd-border);
        background: rgba(128, 128, 128, 0.035);
        margin-bottom: 14px;
    }
    .vd-card h3, .vd-card h4 { margin-top: 0; }

    .species-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 999px;
        background: var(--vd-green-soft);
        color: #2f9e5e;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px 6px 3px 0;
    }

    .decision-chip {
        display: inline-block;
        padding: 4px 13px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: .02em;
    }
    .decision-same   { background: var(--vd-green-soft); color: #2f9e5e; }
    .decision-diff   { background: rgba(128,128,128,.14); color: #8a8a8a; }
    .decision-uncertain { background: var(--vd-amber-soft); color: var(--vd-amber); }
    .decision-insufficient { background: var(--vd-red-soft); color: #c0533f; }

    .evidence-pass { color: #2f9e5e; font-weight: 700; }
    .evidence-fail { color: #c0533f; font-weight: 700; }

    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
    }

    .vd-footer {
        text-align: center;
        opacity: 0.6;
        font-size: 0.82rem;
        padding-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def decision_css_class(decision):
    mapping = {
        "SAME": "decision-same",
        "LIKELY SAME": "decision-same",
        "NEW": "decision-diff",
        "LIKELY DIFFERENT": "decision-diff",
        "UNCERTAIN": "decision-uncertain",
        "INSUFFICIENT EVIDENCE": "decision-insufficient",
        "NO DETECTION": "decision-insufficient",
    }
    return mapping.get(decision, "decision-uncertain")


# ---------- Model / device ----------
def get_device():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return 0, "NVIDIA GPU"
    except Exception:
        pass
    return "cpu", "CPU"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


def class_name(model, idx):
    names = model.names
    return names.get(int(idx), str(idx)) if isinstance(names, dict) else names[int(idx)]


# ---------- Detection ----------
def run_image(model, image):
    device, _ = get_device()
    result = model.predict(
        source=image,
        device=device,
        conf=st.session_state.conf,
        verbose=False,
    )[0]

    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [int(x) for x in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class": class_name(model, cls),
                    "confidence": conf,
                    "bbox": xyxy,
                }
            )
    return annotated, detections


# ---------- Evidence-quality prototype ----------

def appearance_embedding(crop_rgb):
    """
    Transparent baseline identity descriptor.

    It is intentionally NOT presented as a trained Re-ID neural network.
    It combines:
      - HSV color histograms
      - grayscale intensity histogram
      - edge/texture histogram
      - coarse spatial color histograms

    The descriptor is normalized and compared with cosine similarity.
    This gives us a reproducible image-to-image identity baseline while
    the learned part-aware Re-ID model is being developed.
    """
    crop = np.asarray(crop_rgb)
    if crop.size == 0:
        return np.zeros(1, dtype=np.float32)

    crop = cv2.resize(crop, (256, 128))
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    features = []

    # Global HSV histograms
    for channel, bins, rng in [
        (0, 24, [0, 180]),
        (1, 16, [0, 256]),
        (2, 16, [0, 256]),
    ]:
        hist = cv2.calcHist([hsv], [channel], None, [bins], rng).flatten()
        hist = hist / (hist.sum() + 1e-8)
        features.extend(hist.tolist())

    # Grayscale appearance
    gh = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    gh = gh / (gh.sum() + 1e-8)
    features.extend(gh.tolist())

    # Edge/texture distribution
    edges = cv2.Canny(gray, 50, 150)
    eh = cv2.calcHist([edges], [0], None, [16], [0, 256]).flatten()
    eh = eh / (eh.sum() + 1e-8)
    features.extend(eh.tolist())

    # Coarse spatial appearance
    for y in range(2):
        for x in range(4):
            cell = hsv[
                y * hsv.shape[0] // 2:(y + 1) * hsv.shape[0] // 2,
                x * hsv.shape[1] // 4:(x + 1) * hsv.shape[1] // 4,
            ]
            for channel, bins, rng in [
                (0, 12, [0, 180]),
                (1, 8, [0, 256]),
                (2, 8, [0, 256]),
            ]:
                hist = cv2.calcHist([cell], [channel], None, [bins], rng).flatten()
                hist = hist / (hist.sum() + 1e-8)
                features.extend(hist.tolist())

    vec = np.asarray(features, dtype=np.float32)
    vec /= (np.linalg.norm(vec) + 1e-8)
    return vec


def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def crop_from_detection(image_rgb, detection):
    x1, y1, x2, y2 = detection["bbox"]
    h, w = image_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image_rgb[y1:y2, x1:x2]


def reid_decision(similarity, quality_a, quality_b, species_a=None, species_b=None):
    """
    Conservative prototype decision.

    Species mismatch is a hard constraint: two observations belonging to
    different detected species must never be labelled as the same animal.
    """
    if species_a is not None and species_b is not None and species_a != species_b:
        return "LIKELY DIFFERENT"

    min_quality = min(quality_a["score"], quality_b["score"])

    if min_quality < 0.30:
        return "INSUFFICIENT EVIDENCE"
    if similarity >= 0.88:
        return "LIKELY SAME"
    if similarity <= 0.68:
        return "LIKELY DIFFERENT"
    return "UNCERTAIN"


def crop_quality(image_rgb, bbox):
    """Simple quality gate: crop size + blur estimate.
    This is NOT animal Re-ID. It only indicates whether an observation
    contains enough visual evidence for a later identity module.
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return {"score": 0.0, "label": "INSUFFICIENT EVIDENCE"}

    area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, w * h)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Conservative demo-only quality score.
    area_score = min(area_ratio / 0.12, 1.0)
    sharp_score = min(sharpness / 180.0, 1.0)
    score = 0.55 * area_score + 0.45 * sharp_score

    label = "SUFFICIENT EVIDENCE" if score >= 0.45 else "INSUFFICIENT EVIDENCE"
    return {"score": score, "label": label}


# ---------- Video tracking ----------
def process_video(model, video_bytes):
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_in.write(video_bytes)
    temp_in.close()

    cap = cv2.VideoCapture(temp_in.name)
    if not cap.isOpened():
        os.unlink(temp_in.name)
        raise RuntimeError("Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_out.close()

    writer = cv2.VideoWriter(
        temp_out.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    seen_tracks = {}
    frame_count = 0
    device, _ = get_device()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                device=device,
                conf=st.session_state.conf,
                verbose=False,
            )
            result = results[0]

            if result.boxes is not None:
                ids = result.boxes.id
                for i, box in enumerate(result.boxes):
                    cls = int(box.cls[i].item())
                    track_id = (
                        int(ids[i].item())
                        if ids is not None
                        else -1
                    )
                    key = (
                        track_id
                        if track_id >= 0
                        else f"frame_{frame_count}_{i}"
                    )
                    seen_tracks[key] = class_name(model, cls)

            writer.write(result.plot())
            frame_count += 1

    finally:
        cap.release()
        writer.release()
        os.unlink(temp_in.name)

    return temp_out.name, frame_count, seen_tracks




def local_pattern_similarity(crop_a, crop_b):
    """
    Local pattern baseline using ORB keypoints and descriptor matching.

    This is deliberately a baseline experiment, not a learned Re-ID model.
    Local matches help distinguish visually similar individuals of the same
    species because they depend on localized texture/pattern structure.
    """
    if crop_a is None or crop_b is None:
        return 0.0

    a = cv2.cvtColor(cv2.resize(crop_a, (320, 160)), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(cv2.resize(crop_b, (320, 160)), cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=600)
    kp_a, des_a = orb.detectAndCompute(a, None)
    kp_b, des_b = orb.detectAndCompute(b, None)

    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(des_a, des_b, k=2)

    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Normalize by the smaller keypoint set. Cap at 1.
    score = min(1.0, len(good) / max(1, min(len(kp_a), len(kp_b)) * 0.12))
    return float(score)


def combined_identity_similarity(crop_a, crop_b):
    """
    Combine global appearance and localized pattern evidence.

    The local term receives higher weight so that two same-species animals
    with similar global color/brightness are less likely to be merged.
    """
    global_a = appearance_embedding(crop_a)
    global_b = appearance_embedding(crop_b)
    global_score = cosine_similarity(global_a, global_b)
    local_score = local_pattern_similarity(crop_a, crop_b)

    combined = 0.35 * global_score + 0.65 * local_score
    return float(combined), float(global_score), float(local_score)



# ---------- Persistent identity registry (prototype) ----------
def assign_registry_id(registry, species, embedding, quality_score, crop_rgb=None):
    """
    Conservative in-session identity registry.

    Matching uses both global appearance and local pattern evidence.
    A high global score alone is never sufficient to merge two animals.
    """
    if quality_score < 0.30:
        return None, "INSUFFICIENT EVIDENCE", None

    candidates = []

    for animal_id, record in registry.items():
        if record["species"] != species:
            continue

        if crop_rgb is not None and record.get("crop") is not None:
            combined, global_score, local_score = combined_identity_similarity(
                record["crop"], crop_rgb
            )
        else:
            global_score = cosine_similarity(embedding, record["embedding"])
            local_score = 0.0
            combined = 0.35 * global_score

        candidates.append(
            (combined, animal_id, global_score, local_score)
        )

    if not candidates:
        prefix = species[:3].upper()
        number = 1
        while f"{prefix}{number:03d}" in registry:
            number += 1
        new_id = f"{prefix}{number:03d}"

        registry[new_id] = {
            "species": species,
            "embedding": embedding,
            "crop": crop_rgb,
            "observations": 1,
            "quality": quality_score,
            "best_similarity": None,
        }
        return new_id, "NEW", None

    best_combined, best_id, best_global, best_local = max(
        candidates, key=lambda x: x[0]
    )

    # Same identity requires BOTH strong combined evidence and meaningful
    # local-pattern support. This prevents global appearance from dominating.
    if best_combined >= 0.72 and best_local >= 0.55:
        record = registry[best_id]
        record["embedding"] = (
            record["embedding"] * record["observations"] + embedding
        ) / (record["observations"] + 1)
        record["embedding"] /= np.linalg.norm(record["embedding"]) + 1e-8
        record["crop"] = crop_rgb
        record["observations"] += 1
        record["quality"] = max(record["quality"], quality_score)
        record["best_similarity"] = max(
            best_combined,
            record["best_similarity"] or 0.0,
        )
        return best_id, "SAME", best_combined

    # A high global appearance score with weak local evidence is ambiguous,
    # not a reason to merge.
    if best_combined >= 0.55:
        return None, "UNCERTAIN", best_combined

    # Strongly different visual evidence -> create a new identity.
    prefix = species[:3].upper()
    number = 1
    while f"{prefix}{number:03d}" in registry:
        number += 1
    new_id = f"{prefix}{number:03d}"

    registry[new_id] = {
        "species": species,
        "embedding": embedding,
        "crop": crop_rgb,
        "observations": 1,
        "quality": quality_score,
        "best_similarity": None,
    }
    return new_id, "NEW", best_combined



def registry_summary(registry):
    counts = Counter(record["species"] for record in registry.values())
    return counts



# ---------- Header ----------
st.markdown(
    """
    <div class="vd-header">
        <div class="vd-eyebrow">SIH 2026 · Wildlife Intelligence Prototype</div>
        <div class="main-title">🦌 VanDrishti</div>
        <div class="subtitle">
            AI-Based Multi-Camera Wildlife Monitoring &amp; Unique Animal Counting
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("Trained model not found:")
    st.code(str(MODEL_PATH))
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load trained model: {e}")
    st.stop()

device, device_label = get_device()


if "identity_registry" not in st.session_state:
    st.session_state.identity_registry = {}

# ---------- Sidebar ----------
st.sidebar.markdown("### ⚙️ Controls")
mode = st.sidebar.radio(
    "Analysis Mode",
    [
        "Dashboard",
        "Single Image",
        "Video Tracking",
        "Multi-Camera Preview",
        "Identity Evidence",
        "Re-ID Comparison",
        "Unique Count Demo",
    ],
    label_visibility="collapsed",
)

conf = st.sidebar.slider(
    "Detection confidence",
    0.10,
    0.90,
    0.25,
    0.05,
)
st.session_state.conf = conf

st.sidebar.divider()
st.sidebar.markdown("### 🖥️ System")
sys_col1, sys_col2 = st.sidebar.columns(2)
sys_col1.metric("Inference", device_label)
sys_col2.metric("Classes", "17")
st.sidebar.caption("**Model:** VanDrishti YOLO")

st.sidebar.divider()
st.sidebar.markdown("### 🧭 Current Pipeline")

_pipeline_steps = [
    ("📷", "Camera / Image Input", True),
    ("🧹", "Pre-processing", True),
    ("🎯", "YOLO Detection", True),
    ("🐾", "Species Classification", True),
    ("🧭", "ByteTrack Tracking", True),
    ("🧬", "Re-ID / Identity Fusion", True),
    ("🆔", "Persistent Animal ID", True),
    ("📊", "Unique Population Count", True),
]
_pipeline_html = ""
for icon, label, done in _pipeline_steps:
    css_class = "pipeline-step-done" if done else "pipeline-step-pending"
    check = "✓" if done else "…"
    _pipeline_html += f'<span class="{css_class}">{icon} {label} {check}</span><br>'

st.sidebar.markdown(f'<div class="pipeline">{_pipeline_html}</div>', unsafe_allow_html=True)

st.sidebar.caption(
    "Implemented now: detection + species classification + video tracking. "
    "Cross-camera Re-ID and final identity fusion are under development."
)


# ---------- Dashboard ----------
if mode == "Dashboard":
    st.subheader("System Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wildlife Classes", "17")
    c2.metric("Training Images", "20,634")
    c3.metric("Validation Images", "5,023")
    c4.metric("Detector mAP50", "0.951")

    st.divider()

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown('<div class="vd-card">', unsafe_allow_html=True)
        st.markdown("#### 🗺️ VanDrishti Architecture")
        st.markdown(
            """
            **1. Capture** → multiple wildlife camera observations

            **2. Pre-process** → denoising, enhancement, normalization

            **3. Detect** → trained YOLO wildlife detector

            **4. Classify** → 17 wildlife categories

            **5. Track** → ByteTrack maintains IDs within a sequence

            **6. Re-identify** → compare visual identity features across cameras

            **7. Decision fusion** → combine visual similarity with context

            **8. De-duplicate** → repeated observations map to one animal ID

            **9. Count** → category-wise unique population
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown("#### 📌 Implementation Status")

        status = [
            ("Dataset preparation", "DONE"),
            ("Custom YOLO detector", "DONE"),
            ("Species classification", "DONE"),
            ("Video tracking", "DONE"),
            ("Multi-camera workflow", "PROTOTYPE"),
            ("Cross-camera Re-ID", "NEXT"),
            ("Identity fusion / ANFIS", "NEXT"),
            ("Persistent unique counting", "NEXT"),
        ]

        badge_class = {
            "DONE": "badge-done",
            "PROTOTYPE": "badge-proto",
            "NEXT": "badge-next",
        }

        for name, state in status:
            st.markdown(
                f"""
                <div class="status-card">
                    <span class="status-name">{name}</span>
                    <span class="badge {badge_class[state]}">{state}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info(
        "Demo message: VanDrishti is designed to move from counting "
        "detections to counting unique animals across camera observations."
    )


# ---------- Single image ----------
elif mode == "Single Image":
    st.subheader("🔎 Wildlife Detection")
    uploaded = st.file_uploader(
        "Upload a wildlife image",
        type=["jpg", "jpeg", "png"],
        key="single_image",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        with st.spinner("Running VanDrishti detection..."):
            annotated, detections = run_image(model, image_np)

        left, right = st.columns([2.2, 1])

        with left:
            st.image(
                annotated,
                caption="VanDrishti detection result",
                use_container_width=True,
            )

        with right:
            st.metric("Animals detected", len(detections))

            counts = Counter(d["class"] for d in detections)

            st.markdown('<div class="vd-card">', unsafe_allow_html=True)
            st.markdown("##### Species")
            if counts:
                chips = "".join(
                    f'<span class="species-chip">🐾 {species} · {count}</span>'
                    for species, count in sorted(counts.items())
                )
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption("No species detected.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="vd-card">', unsafe_allow_html=True)
            st.markdown("##### Confidence")
            for i, d in enumerate(detections, 1):
                st.progress(
                    d["confidence"],
                    text=f"{i}. {d['class']} — {d['confidence']:.2f}",
                )
            st.markdown('</div>', unsafe_allow_html=True)

            if detections:
                q = crop_quality(image_np, detections[0]["bbox"])
                st.markdown('<div class="vd-card">', unsafe_allow_html=True)
                st.markdown("##### Identity Evidence Gate")
                gate_class = "evidence-pass" if q["label"] == "SUFFICIENT EVIDENCE" else "evidence-fail"
                st.metric("Evidence score", f"{q['score']:.2f}")
                st.markdown(f'<span class="{gate_class}">{q["label"]}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.info(
            "Detection and species classification are implemented. "
            "The evidence gate is only a quality check for a future "
            "identity module; it does not claim cross-camera Re-ID."
        )


# ---------- Video ----------
elif mode == "Video Tracking":
    st.subheader("🎥 Wildlife Tracking")
    uploaded = st.file_uploader(
        "Upload a short wildlife video",
        type=["mp4", "avi", "mov"],
        key="video",
    )
    st.caption("For the jury demo, use a short 5–20 second clip.")

    if uploaded and st.button(
        "▶ Run Detection + ByteTrack",
        type="primary",
    ):
        with st.spinner("Processing video..."):
            try:
                out_path, frames, tracks = process_video(
                    model,
                    uploaded.getvalue(),
                )

                with open(out_path, "rb") as f:
                    video_data = f.read()

                st.success(f"Processed {frames} frames.")
                st.video(video_data)

                species_counts = Counter(tracks.values())

                c1, c2, c3 = st.columns(3)
                c1.metric("Track IDs observed", len(tracks))
                c2.metric("Frames processed", frames)
                c3.metric("Inference device", device_label)

                st.markdown('<div class="vd-card">', unsafe_allow_html=True)
                st.markdown("##### Tracked categories")
                if species_counts:
                    chips = "".join(
                        f'<span class="species-chip">🐾 {species} · {count} track(s)</span>'
                        for species, count in sorted(species_counts.items())
                    )
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption("No tracks recorded.")
                st.markdown('</div>', unsafe_allow_html=True)

                st.warning(
                    "Track IDs are temporary IDs within this video. "
                    "They are not persistent cross-camera animal IDs."
                )

                os.unlink(out_path)

            except Exception as e:
                st.error(f"Video processing failed: {e}")


# ---------- Multi-camera ----------
elif mode == "Multi-Camera Preview":
    st.subheader("📷 Multi-Camera Monitoring")
    st.caption(
        "Upload observations from multiple cameras to demonstrate the "
        "multi-camera acquisition and detection layer."
    )

    files = st.file_uploader(
        "Upload up to 6 camera images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="multi",
    )

    if files:
        files = files[:6]
        cols = st.columns(3)
        all_counts = Counter()
        total_detections = 0

        for idx, uploaded in enumerate(files):
            image = Image.open(uploaded).convert("RGB")
            image_np = np.array(image)

            with st.spinner(f"Analyzing Camera {idx + 1}..."):
                annotated, detections = run_image(
                    model,
                    image_np,
                )

            with cols[idx % 3]:
                st.markdown('<div class="vd-card">', unsafe_allow_html=True)
                st.markdown(f"##### 📹 Camera {idx + 1}")
                st.image(annotated, use_container_width=True)
                st.write(
                    f"Detections: **{len(detections)}**"
                )

                for d in detections:
                    all_counts[d["class"]] += 1

                if detections:
                    best = max(
                        detections,
                        key=lambda x: x["confidence"],
                    )
                    st.caption(
                        f"Top observation: {best['class']} "
                        f"({best['confidence']:.2f})"
                    )
                st.markdown('</div>', unsafe_allow_html=True)

            total_detections += len(detections)

        st.divider()

        c1, c2, c3 = st.columns(3)
        c1.metric("Cameras", len(files))
        c2.metric("Total detections", total_detections)
        c3.metric("Unique count", "Pending Re-ID")

        st.markdown('<div class="vd-card">', unsafe_allow_html=True)
        st.markdown("##### Observation Summary")
        if all_counts:
            chips = "".join(
                f'<span class="species-chip">🐾 {species} · {count}</span>'
                for species, count in sorted(all_counts.items())
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("No detections across uploaded cameras.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.info(
            "Important: adding detections from different cameras can "
            "double-count the same animal. The Re-ID + identity-fusion "
            "layer is designed to convert repeated observations into "
            "persistent animal identities."
        )


# ---------- Identity evidence ----------
elif mode == "Identity Evidence":
    st.subheader("🧬 Identity Evidence Prototype")
    st.caption(
        "This screen demonstrates the design response to partial/poor "
        "observations. It does not claim a completed Re-ID model."
    )

    uploaded = st.file_uploader(
        "Upload an observation image",
        type=["jpg", "jpeg", "png"],
        key="identity_image",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        with st.spinner("Detecting animal and evaluating visual evidence..."):
            annotated, detections = run_image(model, image_np)

        st.image(
            annotated,
            caption="Observation used for identity-evidence assessment",
            use_container_width=True,
        )

        if not detections:
            st.error("No animal detected — identity evidence unavailable.")
        else:
            rows = []
            for i, d in enumerate(detections, 1):
                q = crop_quality(image_np, d["bbox"])
                rows.append(
                    (
                        i,
                        d["class"],
                        f"{d['confidence']:.2f}",
                        f"{q['score']:.2f}",
                        q["label"],
                    )
                )

            st.markdown('<div class="vd-card">', unsafe_allow_html=True)
            st.markdown("##### Evidence assessment")
            st.table(
                {
                    "Observation": [r[0] for r in rows],
                    "Species": [r[1] for r in rows],
                    "Detection confidence": [r[2] for r in rows],
                    "Evidence score": [r[3] for r in rows],
                    "Identity gate": [r[4] for r in rows],
                }
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.warning(
                "If only a small or heavily blurred body part is visible, "
                "VanDrishti should not force a SAME/NEW identity decision. "
                "The intended identity layer will use an explicit "
                "INSUFFICIENT EVIDENCE state and wait for a better observation."
            )


elif mode == "Re-ID Comparison":
    st.subheader("🧬 Image-to-Image Re-ID Baseline")
    st.caption(
        "VanDrishti baseline identity experiment: compare two animal "
        "observations using a transparent global + local appearance baseline. "
        "This is a real baseline, not a claimed final Re-ID neural model."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        file_a = st.file_uploader(
            "Observation A",
            type=["jpg", "jpeg", "png"],
            key="reid_a",
        )
    with col_b:
        file_b = st.file_uploader(
            "Observation B",
            type=["jpg", "jpeg", "png"],
            key="reid_b",
        )

    if file_a and file_b:
        img_a = np.array(Image.open(file_a).convert("RGB"))
        img_b = np.array(Image.open(file_b).convert("RGB"))

        with st.spinner("Detecting animals and comparing appearance..."):
            _, det_a = run_image(model, img_a)
            _, det_b = run_image(model, img_b)

        if not det_a or not det_b:
            st.error(
                "Both observations must contain a detected animal before "
                "the identity baseline can compare them."
            )
        else:
            # Use the highest-confidence detection in each observation.
            best_a = max(det_a, key=lambda d: d["confidence"])
            best_b = max(det_b, key=lambda d: d["confidence"])

            crop_a = crop_from_detection(img_a, best_a)
            crop_b = crop_from_detection(img_b, best_b)

            qa = crop_quality(img_a, best_a["bbox"])
            qb = crop_quality(img_b, best_b["bbox"])

            emb_a = appearance_embedding(crop_a)
            emb_b = appearance_embedding(crop_b)
            similarity, global_similarity, local_similarity = (
                combined_identity_similarity(crop_a, crop_b)
            )

            decision = reid_decision(
                similarity,
                qa,
                qb,
                best_a["class"],
                best_b["class"],
            )

            ca, cb = st.columns(2)
            with ca:
                st.image(crop_a, caption=f"A: {best_a['class']} ({best_a['confidence']:.2f})")
                gate_class_a = "evidence-pass" if qa["label"] == "SUFFICIENT EVIDENCE" else "evidence-fail"
                st.markdown(f'Evidence: <span class="{gate_class_a}">{qa["label"]}</span>', unsafe_allow_html=True)
            with cb:
                st.image(crop_b, caption=f"B: {best_b['class']} ({best_b['confidence']:.2f})")
                gate_class_b = "evidence-pass" if qb["label"] == "SUFFICIENT EVIDENCE" else "evidence-fail"
                st.markdown(f'Evidence: <span class="{gate_class_b}">{qb["label"]}</span>', unsafe_allow_html=True)

            st.divider()

            m1, m2, m3 = st.columns(3)
            m1.metric("Combined similarity", f"{similarity:.3f}")
            m2.metric("Local pattern similarity", f"{local_similarity:.3f}")
            with m3:
                st.markdown(
                    f'<div class="small-note">Identity decision</div>'
                    f'<span class="decision-chip {decision_css_class(decision)}">{decision}</span>',
                    unsafe_allow_html=True,
                )

            if best_a["class"] != best_b["class"]:
                st.warning(
                    "Species differ, so these observations cannot represent "
                    "the same animal. Species mismatch overrides appearance similarity."
                )
            elif decision == "LIKELY SAME":
                st.success(
                    "Baseline result: the two observations are visually "
                    "similar enough to be candidates for the same animal."
                )
            elif decision == "LIKELY DIFFERENT":
                st.info(
                    "Baseline result: the two observations are visually "
                    "different enough to be candidates for different animals."
                )
            elif decision == "INSUFFICIENT EVIDENCE":
                st.warning(
                    "Do not force an identity decision. A future part-aware "
                    "Re-ID model should wait for a more informative observation."
                )
            else:
                st.warning(
                    "Similarity falls in the uncertain band. Keep the "
                    "observations separate until additional visual or "
                    "temporal/spatial evidence is available."
                )

            st.markdown("### Decision logic")
            st.code(
                "Appearance features + species + evidence quality\n"
                "                     ↓\n"
                "             Similarity comparison\n"
                "                     ↓\n"
                "      SAME / DIFFERENT / UNCERTAIN\n"
                "                     ↓\n"
                "       insufficient evidence → wait",
                language="text",
            )

            st.info(
                "Research status: this appearance descriptor is our "
                "baseline experiment. The next learned component is a "
                "part-aware metric-learning Re-ID model trained on "
                "sequence-derived positive/negative pairs."
            )



elif mode == "Unique Count Demo":
    st.subheader("🆔 Unique Animal Counting")
    st.caption(
        "End-to-end prototype: each uploaded observation is detected, "
        "converted to an appearance descriptor, compared with the current "
        "identity registry, and assigned a persistent in-session ID."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        files = st.file_uploader(
            "Upload observations in camera/observation order",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="registry_images",
        )
    with col2:
        if st.button("🗑 Reset Identity Registry"):
            st.session_state.identity_registry = {}
            st.rerun()

    st.info(
        "Demo rule: same species + appearance similarity ≥ 0.88 → SAME; "
        "0.68–0.88 → UNCERTAIN; lower → NEW. These are baseline prototype "
        "thresholds, not final research thresholds."
    )

    if files:
        registry = st.session_state.identity_registry
        results = []

        for idx, uploaded in enumerate(files[:12], start=1):
            image = Image.open(uploaded).convert("RGB")
            image_np = np.array(image)

            with st.spinner(f"Analyzing observation {idx}..."):
                annotated, detections = run_image(model, image_np)

            if not detections:
                results.append({
                    "file": uploaded.name,
                    "species": "—",
                    "animal_id": "—",
                    "decision": "NO DETECTION",
                    "similarity": None,
                    "image": annotated,
                })
                continue

            # For the demo, use the highest-confidence animal in each image.
            det = max(detections, key=lambda d: d["confidence"])
            crop = crop_from_detection(image_np, det)
            quality = crop_quality(image_np, det["bbox"])
            embedding = appearance_embedding(crop)

            animal_id, decision, similarity = assign_registry_id(
                registry,
                det["class"],
                embedding,
                quality["score"],
                crop_rgb=crop,
            )

            results.append({
                "file": uploaded.name,
                "species": det["class"],
                "animal_id": animal_id or "PENDING",
                "decision": decision,
                "similarity": similarity,
                "image": annotated,
                "confidence": det["confidence"],
            })

        st.divider()
        st.markdown("### Camera / Observation Results")

        cols = st.columns(3)
        for i, r in enumerate(results):
            with cols[i % 3]:
                st.markdown('<div class="vd-card">', unsafe_allow_html=True)
                st.markdown(f"**Observation {i+1}**")
                st.image(r["image"], use_container_width=True)
                st.write(f"File: `{r['file']}`")
                st.write(f"Species: **{r['species']}**")
                st.write(f"Animal ID: **{r['animal_id']}**")
                st.markdown(
                    f'<span class="decision-chip {decision_css_class(r["decision"])}">{r["decision"]}</span>',
                    unsafe_allow_html=True,
                )
                if r["similarity"] is not None:
                    st.write(f"Best similarity: **{r['similarity']:.3f}**")
                if "confidence" in r:
                    st.caption(f"Detection confidence: {r['confidence']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        counts = registry_summary(registry)
        total_unique = len(registry)

        m1, m2, m3 = st.columns(3)
        m1.metric("Unique animals", total_unique)
        m2.metric("Observations processed", len(results))
        m3.metric("Species represented", len(counts))

        st.markdown("### 🆔 Identity Registry")
        if registry:
            table_rows = []
            for animal_id, record in registry.items():
                table_rows.append({
                    "Animal ID": animal_id,
                    "Species": record["species"],
                    "Observations": record["observations"],
                    "Best similarity": (
                        f"{record['best_similarity']:.3f}"
                        if record["best_similarity"] is not None
                        else "—"
                    ),
                })
            st.table(table_rows)

            st.markdown('<div class="vd-card">', unsafe_allow_html=True)
            st.markdown("##### 📊 Unique Population")
            chips = "".join(
                f'<span class="species-chip">🐾 {species} · {count} unique</span>'
                for species, count in sorted(counts.items())
            )
            st.markdown(chips, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.success(
            "Core concept demonstrated: repeated observations that match "
            "an existing identity do not increment the unique-animal count."
        )

        st.warning(
            "Prototype limitation: this screen compares the best detection "
            "per image and uses the current appearance baseline. The final "
            "VanDrishti system will replace this with learned part-aware "
            "Re-ID plus temporal/spatial/contextual fusion."
        )


st.divider()
st.markdown(
    '<div class="vd-footer">VanDrishti • SIH 2026 Prototype • '
    'Detection + tracking implemented • '
    'Cross-camera Re-ID and identity fusion under development</div>',
    unsafe_allow_html=True,
)