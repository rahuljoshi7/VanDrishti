\# VanDrishti



\### AI-Based Multi-Camera Wildlife Monitoring for Unique Animal Counting



> \*\*Counting animals, not detections.\*\*



\## Overview



VanDrishti is an AI-based wildlife monitoring prototype designed to reduce duplicate counting of animals observed across multiple wildlife cameras.



A conventional detection system can identify an animal in a frame, but repeated observations of the same individual can be counted multiple times. VanDrishti extends the detection pipeline with tracking, animal Re-ID and identity registration to move toward unique-animal population counting.



\## Problem



Wildlife surveillance cameras can repeatedly capture the same animal across:



\- multiple frames

\- different cameras

\- different time periods



Manual analysis of these observations is time-consuming and can lead to duplicate counting.



\## System Flow



Camera / Image / Video  

↓  

Image Preprocessing  

↓  

YOLO Wildlife Detection  

↓  

Species + Bounding Box  

↓  

ByteTrack Tracking  

↓  

Animal Crop  

↓  

Individual Re-ID  

↓  

Identity Decision  

↓  

Identity Registry  

↓  

Unique Animal Count  

↓  

Dashboard



\## Current Prototype



\### Implemented



\- Wildlife dataset audit and preparation

\- Pascal VOC → YOLO conversion

\- Sequence-aware and class-balanced dataset split

\- Custom-trained YOLO26n detector

\- ByteTrack tracking

\- Baseline image-to-image Re-ID

\- Identity registry

\- Unique-count prototype

\- Streamlit GUI

\- Multi-camera observation workflow



\### Under Development



\- Wildlife-specific part-aware Re-ID

\- Individual-level Re-ID training data

\- Evidence-aware identity fusion

\- ANFIS-based decision fusion

\- Cross-camera evaluation



\## Dataset



\- 25,657 images

\- 26,946 annotated animal instances

\- 17 wildlife categories

\- 15,313 day images

\- 10,344 night images



The raw dataset is not included in this repository because of its size.



See \[`data/README.md`](data/README.md) for dataset preparation details.



\## Detection Results



| Metric | Result |

|---|---:|

| Precision | 93.5% |

| Recall | 90.8% |

| mAP@50 | 95.1% |

| mAP@50-95 | 81.1% |



\## Architecture



!\[VanDrishti Architecture](docs/architecture.png)



\## Prototype Screenshots



\### Wildlife Detection



!\[Detection](docs/gui-detection.png)



\### Multi-Camera Monitoring



!\[Multi-camera](docs/gui-multicamera.png)



\### Re-ID Baseline



!\[Re-ID](docs/gui-reid.png)



\### Unique Animal Counting



!\[Unique Count](docs/gui-unique-count.png)



\## Technology Stack



\- Python 3.11.9

\- PyTorch 2.8.0

\- Ultralytics 8.4.143

\- YOLO26n

\- OpenCV

\- ByteTrack

\- Streamlit

\- NumPy

\- Pandas

\- Pillow

\- Git



\## Installation



```bash

git clone <repository-url>

cd VanDrishti



python -m venv venv



\# Windows

venv\\Scripts\\activate



pip install -r requirements.txt

