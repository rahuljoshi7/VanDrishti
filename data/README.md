\# Dataset



VanDrishti uses day and night wildlife imagery for detection and identity-association experiments.



\## Dataset Statistics



\- Total images: 25,657

\- Annotated animal instances: 26,946

\- Wildlife categories: 17

\- Day images: 15,313

\- Night images: 10,344



\## Annotation



The original dataset uses Pascal VOC XML annotations.



The preprocessing pipeline converts these annotations into YOLO format for model training.



\## Preparation Pipeline



1\. Dataset audit

2\. Metadata and filename analysis

3\. Sequence analysis

4\. Pascal VOC → YOLO conversion

5\. YOLO label verification

6\. Sequence-aware and class-balanced train/validation split



The raw dataset and generated image/label files are not included in this repository because of their size.

