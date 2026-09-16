\# VanDrishti - Prototype Results



\## Dataset



VanDrishti currently uses a combined day and night wildlife dataset.



\- Total images: 25,657

\- Annotated animal instances: 26,946

\- Wildlife categories: 17

\- Day images: 15,313

\- Night images: 10,344



\## Dataset Preparation



The original Pascal VOC XML annotations were converted to YOLO format.



The preparation pipeline includes:



1\. Dataset audit

2\. Metadata and filename analysis

3\. Sequence analysis

4\. Pascal VOC to YOLO conversion

5\. YOLO label verification

6\. Sequence-aware and class-balanced train/validation splitting



\## Detection Model



The current detector is a custom-trained YOLO26n model for 17 wildlife categories.



\### Validation Results



| Metric | Result |

|---|---:|

| Precision | 93.5% |

| Recall | 90.8% |

| mAP@50 | 95.1% |

| mAP@50-95 | 81.1% |



These metrics describe the object detection model and are not Re-ID metrics.



\## Tracking



ByteTrack is used to maintain temporary identities across consecutive frames within a camera sequence.



\## Re-Identification Baseline



The current prototype contains a baseline image-to-image animal identity comparison using appearance features.



The baseline can produce:



\- SAME

\- NEW

\- UNCERTAIN

\- INSUFFICIENT EVIDENCE



The current identity system is a prototype and is not claimed to be the final learned wildlife Re-ID model.



\## Unique Counting



The prototype maintains an in-session identity registry so that repeated observations associated with the same identity are not counted as new animals.



\## GUI



A Streamlit interface currently provides:



\- Single image detection

\- Video tracking

\- Multi-camera observation preview

\- Re-ID comparison

\- Unique animal counting



\## Current Limitations



Individual-level identity annotations are not explicitly available in the current detection dataset.



The baseline Re-ID system can struggle with:



\- Similar-looking individuals

\- Motion blur

\- Viewpoint changes

\- Occlusion

\- Partial visibility

\- Different camera conditions



\## Next Development Stage



The next research stage focuses on:



\- Verified individual-ID pairs

\- Wildlife-specific part-aware Re-ID

\- Better handling of partial observations

\- Evidence-aware identity decisions

\- Decision fusion using visual and contextual information

\- Cross-camera evaluation

\- Duplicate-count reduction evaluation

